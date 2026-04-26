"""Local model loading and intermediate-activation capture.

Loads a Hugging Face causal-LM checkpoint from a local directory (no network),
selects an appropriate device and dtype, and provides a context manager that
captures the per-layer outputs of attention and MLP blocks during a forward
pass.

The model loader is intentionally restricted to local files — `set_offline_mode`
sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` so that nothing reaches
the Hugging Face Hub at runtime. Download the weights once with
`huggingface-cli download` (see README), then point `MYELIN_MODEL_PATH` (or
the `--model-path` CLI flag) at the local directory.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device_and_dtype(
    device_arg: Optional[str] = None,
    dtype_arg: Optional[str] = None,
) -> tuple[str, torch.dtype]:
    """Pick a sensible (device, dtype) pair for the host machine.

    Device priority when `device_arg` is None: CUDA -> MPS (Apple Silicon) -> CPU.
    Set `MYELIN_DEVICE` or `MYELIN_DTYPE` to override defaults process-wide.
    Dtype defaults: bf16 on CUDA when supported else fp16; fp16 on MPS;
    bf16 on CPU when supported else fp32.
    """
    device = device_arg or os.environ.get("MYELIN_DEVICE")
    if device is not None:
        device = device.lower()
        if device == "auto":
            device = None

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device not in {"cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device; choose from auto|cpu|cuda|mps.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MYELIN_DEVICE=cuda was requested, but CUDA is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MYELIN_DEVICE=mps was requested, but PyTorch reports MPS is not available. "
            "Run `python -c \"import torch; print(torch.backends.mps.is_available())\"` "
            "from the same environment to confirm."
        )

    dtype_arg = dtype_arg or os.environ.get("MYELIN_DTYPE")
    if dtype_arg is not None:
        mapping = {
            "float32": torch.float32, "fp32": torch.float32,
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        }
        key = dtype_arg.lower()
        if key not in mapping:
            raise ValueError(
                f"Unsupported dtype {dtype_arg!r}. Choose from float32|float16|bfloat16."
            )
        return device, mapping[key]

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        # CPU: bf16 is fine on most modern x86/arm64; fall back to fp32 otherwise.
        try:
            torch.tensor(1.0, dtype=torch.bfloat16)
            dtype = torch.bfloat16
        except (RuntimeError, TypeError):
            dtype = torch.float32
    return device, dtype


def set_offline_mode(enable: bool = True) -> None:
    """Toggle Hugging Face offline flags for the current process."""
    if enable:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


def load_local_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: torch.dtype,
):
    """Load a causal LM and tokenizer from a local directory.

    Raises FileNotFoundError if `model_path` is not a directory containing a
    Hugging Face checkpoint.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path not found or not a directory: {model_path}\n"
            "Set MYELIN_MODEL_PATH or pass --model-path. See README for download instructions."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    if device in ("cuda", "mps"):
        model.to(device)
    model.eval()
    return model, tokenizer


def _get_decoder_layers(model):
    """Locate the iterable of decoder layers in a causal-LM model.

    Works for Qwen2/LLaMA-family architectures where the layers live at
    `model.model.layers`. Falls back to `model.layers` for less common shapes.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Could not locate decoder layers on this model.")


class ActivationCapture:
    """Context manager that captures per-layer attention and MLP outputs.

    By default captures only the last token's activation per layer to keep
    memory bounded. Pass `last_token_only=False` to capture full sequences
    (used by the prepare step, which wants to read activations at a specific
    position within a longer prompt).
    """

    def __init__(
        self,
        model,
        capture_attn: bool = True,
        capture_mlp: bool = True,
        last_token_only: bool = True,
    ):
        self.model = model
        self.capture_attn = capture_attn
        self.capture_mlp = capture_mlp
        self.last_token_only = last_token_only
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.data: Dict[str, Dict[int, torch.Tensor]] = {"attn": {}, "mlp": {}}

    def _maybe_slice_last(self, t: torch.Tensor) -> torch.Tensor:
        if self.last_token_only and t.dim() == 3 and t.size(1) >= 1:
            return t[:, -1, :].detach().cpu()
        return t.detach().cpu()

    def __enter__(self):
        layers = _get_decoder_layers(self.model)

        for idx, layer in enumerate(layers):
            if self.capture_mlp and hasattr(layer, "mlp"):
                def mlp_hook(module, inputs, output, idx=idx):
                    out = output[0] if isinstance(output, (tuple, list)) else output
                    self.data["mlp"][idx] = self._maybe_slice_last(out)
                self.handles.append(layer.mlp.register_forward_hook(mlp_hook))

            if self.capture_attn and (hasattr(layer, "self_attn") or hasattr(layer, "attn")):
                attn_mod = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
                def attn_hook(module, inputs, output, idx=idx):
                    out = output[0] if isinstance(output, (tuple, list)) else output
                    self.data["attn"][idx] = self._maybe_slice_last(out)
                self.handles.append(attn_mod.register_forward_hook(attn_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def get(self) -> Dict[str, Dict[int, torch.Tensor]]:
        return self.data
