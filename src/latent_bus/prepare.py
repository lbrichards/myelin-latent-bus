"""Build the fact vector v_F from parity-matched contexts.

Given a single missing fact F (e.g. "Bob is the father of Joe.") we construct
two contexts that are identical except for a single controlled span:

    A'  ...  {placeholder_padding_of_equal_token_length}  ...
    B   ...  {fact F}                                     ...

At a chosen site/layer we capture the hidden state at the answer-start
position (the last token of the question prompt) for each context, then
average `h_B - h_A'` across templates and seeds and unit-normalize.

The resulting unit vector v_F is the latent-bus payload: adding alpha * v_F
at answer-start during decoding transmits the fact without any visible tokens.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

from latent_bus.model_io import (
    ActivationCapture,
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)

LOG = logging.getLogger(__name__)

DEFAULT_SEEDS = (101, 102, 103)

# Three light template variations used to average out prompt-specific quirks.
DEFAULT_TEMPLATES = (
    ("", ""),
    ("Please consider: ", ""),
    ("", " Think carefully."),
)

BASE_CONTEXT_WITH_FACT = """Consider this family tree:
- Paul and Mary are the parents of Bob and Carol.
- Bob and Carol are siblings.
- Carol is the mother of Nancy.
- {fact_line}
- Joe exists in the family.

Based on this information, answer questions about family relationships."""

DEFAULT_QUERY = "Nancy is the cousin of whom?"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_placeholder(tokenizer, target_token_count: int) -> str:
    """Pad a neutral sentence to exactly `target_token_count` tokens."""
    base = "The system needs calibration data here."
    tokens = tokenizer.encode(base)
    while len(tokens) < target_token_count:
        base += " now"
        tokens = tokenizer.encode(base)
    while len(tokens) > target_token_count:
        words = base.split()
        if len(words) <= 1:
            break
        base = " ".join(words[:-1])
        tokens = tokenizer.encode(base)
    return base


def _build_contexts(tokenizer, fact: str) -> tuple[str, str, str]:
    """Return (context_without_fact, context_with_fact, placeholder_text).

    The placeholder line in context_without_fact has the same token count as
    the fact line in context_with_fact, so the two prompts differ by content
    but not by length at the controlled span.
    """
    fact_tokens = tokenizer.encode(fact)
    placeholder = _build_placeholder(tokenizer, len(fact_tokens))
    ctx_without = BASE_CONTEXT_WITH_FACT.format(fact_line=placeholder)
    ctx_with = BASE_CONTEXT_WITH_FACT.format(fact_line=fact)
    return ctx_without, ctx_with, placeholder


def _capture_at_answer_start(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    site: str,
    device: str,
) -> np.ndarray:
    """Run the model on `prompt` and return the hidden state at the last position."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    capture = ActivationCapture(
        model,
        capture_attn=(site == "attn_out"),
        capture_mlp=(site == "mlp_out"),
        last_token_only=False,
    )

    with torch.no_grad(), capture:
        _ = model(**inputs)

    data = capture.get()
    key = "mlp" if site == "mlp_out" else "attn"
    if layer_idx not in data[key]:
        raise KeyError(f"Layer {layer_idx} not captured at site {site!r}.")

    act = data[key][layer_idx]
    if act.ndim == 3:
        act = act[0, -1, :]
    elif act.ndim == 2:
        act = act[-1, :]
    return act.to(torch.float32).cpu().numpy()


def prepare_fact_vector(
    model_path: str,
    fact: str,
    out_dir: Path,
    layer: int = 10,
    site: str = "mlp_out",
    query: str = DEFAULT_QUERY,
    templates = DEFAULT_TEMPLATES,
    seeds = DEFAULT_SEEDS,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
) -> Path:
    """Build and save the fact vector v_F. Returns the path to the saved .npy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vectors").mkdir(exist_ok=True)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)

    ctx_without, ctx_with, placeholder_text = _build_contexts(tokenizer, fact)

    deltas: List[np.ndarray] = []
    trials = []

    for template_idx, (prefix, suffix) in enumerate(templates):
        for seed in seeds:
            _set_all_seeds(seed)

            prompt_without = (
                f"{prefix}{ctx_without}{suffix}\n\nQuestion: {query}\nAnswer:"
            )
            prompt_with = (
                f"{prefix}{ctx_with}{suffix}\n\nQuestion: {query}\nAnswer:"
            )

            h_without = _capture_at_answer_start(
                model, tokenizer, prompt_without, layer, site, device
            )
            h_with = _capture_at_answer_start(
                model, tokenizer, prompt_with, layer, site, device
            )

            delta = h_with - h_without
            deltas.append(delta)
            trials.append({
                "template_idx": template_idx,
                "seed": seed,
                "site": site,
                "layer": layer,
                "fact": fact,
                "placeholder_text": placeholder_text,
                "fact_tokens": len(tokenizer.encode(fact)),
                "placeholder_tokens": len(tokenizer.encode(placeholder_text)),
                "delta_norm": float(np.linalg.norm(delta)),
            })
            LOG.info(
                "  template %d, seed %d: ||delta||=%.3f",
                template_idx, seed, np.linalg.norm(delta),
            )

    mean_delta = np.mean(deltas, axis=0)
    norm = float(np.linalg.norm(mean_delta))
    if norm < 1e-10:
        raise RuntimeError(
            "Mean delta has near-zero norm — parity-matched contexts produced "
            "no detectable activation difference."
        )
    v_F = (mean_delta / norm).astype(np.float32)

    vector_path = out_dir / "vectors" / "v_F.npy"
    np.save(vector_path, v_F)
    LOG.info("Saved fact vector: shape=%s, ||v||=%.6f", v_F.shape, np.linalg.norm(v_F))

    trials_path = out_dir / "vectors" / "trials.jsonl"
    with trials_path.open("w") as fh:
        for t in trials:
            fh.write(json.dumps(t) + "\n")

    metadata = {
        "fact": fact,
        "query": query,
        "site": site,
        "layer": layer,
        "n_templates": len(templates),
        "seeds": list(seeds),
        "mean_delta_norm": norm,
        "vector_shape": list(v_F.shape),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "prepare_metadata.json").write_text(json.dumps(metadata, indent=2))

    # Free memory before returning — the caller may load the model again.
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return vector_path
