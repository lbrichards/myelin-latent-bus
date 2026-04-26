"""Run the three-condition demo: Null, Token, Latent.

Null     : query alone, no fact supplied anywhere. Expected: fails.
Token    : full puzzle (definitions + all facts including F) as visible text.
           Expected: succeeds with non-zero visible tokens.
Latent   : query alone, but with the fact vector injected at answer-start.
           Expected: succeeds with zero visible fact tokens.

The harness is deterministic (greedy decoding, fixed seeds), loads the model
once, and writes per-trial logs plus a summary CSV.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from latent_bus.injection import answer_start_injection
from latent_bus.model_io import (
    _get_decoder_layers,
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)

LOG = logging.getLogger(__name__)

TOKEN_PROMPT_TEMPLATE = """DEFINITIONS:
Let Parent(X,Y) mean "X is a parent of Y."
Let Sibling(X,Y) mean "X and Y share at least one parent and X != Y."
Let Cousin(X,Y) mean "exists A,B: Parent(A,X) and Parent(B,Y) and Sibling(A,B) and X != Y."
Let Father(X,Y) mean "X is male and Parent(X,Y)."  From Father(X,Y) infer Parent(X,Y).
Task: Using ONLY the facts below, answer the query with a single PERSON NAME
chosen from {{Alice, Bob, Carol, Dan, Joe, Nancy}}. Do not explain.

FACTS:
- Paul and Mary are the parents of Bob and Carol.
- Bob and Carol are siblings.
- Carol is the mother of Nancy.
- {fact}
- Joe and Nancy exist in the family.

Question: {query}
Answer:"""

EXPECTED_ANSWER_TOKEN = "joe"
DEFAULT_CANDIDATES = ("Alice", "Bob", "Carol", "Dan", "Joe", "Nancy")
BASELINE_ANSWER_CANDIDATES = ("Joe", "Alice", "Dan", "Grace", "Leo", "Max")
BASELINE_TARGETS = ("Joe", "Alice", "Dan", "Grace", "Leo", "Max")
FACT_PROBE_CANDIDATES = ("Bob", "Carol", "Joe", "Alice", "Dan", "Grace", "Leo", "Max")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_condition_prompt(query: str, fact: Optional[str] = None) -> tuple[str, int]:
    if fact is not None:
        return TOKEN_PROMPT_TEMPLATE.format(fact=fact, query=query), -1
    return f"Question: {query}\nAnswer:", 0


def _build_kinship_baseline_prompt(
    target: str,
    candidates: Sequence[str],
    include_fact: bool,
    variant: str = "train",
) -> tuple[str, str]:
    fact = f"Bob is the father of {target}."
    candidate_text = ", ".join(candidates)
    fact_lines = ["Bob and Carol are siblings.", "Carol is Nancy's parent."]
    if include_fact:
        fact_lines.append(fact)
    facts = "\n".join(f"- {line}" for line in fact_lines)
    if variant == "train":
        prompt = f"""Choose the name that completes the sentence.

Facts:
{facts}

A child of Bob is Nancy's cousin, because Bob is Carol's sibling and Carol is Nancy's parent.

Sentence: Nancy's cousin is _____.
Choices: {candidate_text}
Answer:"""
    elif variant == "paraphrase":
        prompt = f"""Use the family facts to answer with one name.

Facts:
{facts}

Bob and Carol are siblings. Carol is Nancy's parent, so Bob's child is Nancy's cousin.

Question: Which child of Bob is Nancy's cousin?
Choices: {candidate_text}
Answer:"""
    elif variant == "minimal":
        prompt = f"""Facts:
{facts}

If Bob is Carol's sibling and Carol is Nancy's parent, Bob's child is Nancy's cousin.
Complete with one choice: Nancy's cousin is
Choices: {candidate_text}
Answer:"""
    else:
        raise ValueError(f"Unknown kinship prompt variant {variant!r}.")
    return prompt, fact


def _build_fact_probe_prompt(target: str, include_fact: bool) -> tuple[str, str]:
    fact = f"Bob is the father of {target}."
    fact_lines = ["Bob and Carol are siblings.", "Carol is Nancy's parent."]
    if include_fact:
        fact_lines.append(fact)
    facts = "\n".join(f"- {line}" for line in fact_lines)
    prompt = f"""Use the family facts to answer with one name.

Facts:
{facts}

Question: Who is the father of {target}?
Choices: {", ".join(FACT_PROBE_CANDIDATES)}
Answer:"""
    return prompt, fact


def _save_prompt(
    prompts_dir: Path,
    condition: str,
    prompt: str,
    tokenizer,
) -> dict:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{condition.lower()}.txt"
    prompt_path.write_text(prompt)
    return {
        "prompt_path": str(prompt_path),
        "prompt_sha256": _sha256_text(prompt),
        "prompt_tokens": len(tokenizer.encode(prompt)),
    }


def _attach_prompt_artifact(
    trial: dict,
    prompts_dir: Path,
    tokenizer,
) -> dict:
    prompt = trial.pop("prompt")
    trial.update(_save_prompt(prompts_dir, trial["condition"], prompt, tokenizer))
    return trial


def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 20) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _evaluate(
    model,
    tokenizer,
    query: str,
    device: str,
    fact: Optional[str] = None,
    vector: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    layer: int = 10,
    site: str = "mlp_out",
    burst_steps: int = 1,
    inject_mode: str = "add",
) -> dict:
    """Run one trial. Returns answer, correctness, visible_tokens, effect data."""
    prompt, visible_tokens = _build_condition_prompt(query, fact)
    if fact is not None:
        visible_tokens = len(tokenizer.encode(fact))

    hook = None
    effect: dict = {}
    if vector is not None and alpha != 0.0:
        hook, effect = answer_start_injection(
            model, vector, alpha, layer, site=site, burst_steps=burst_steps,
            mode=inject_mode,
        )
    try:
        answer = _generate(model, tokenizer, prompt, device)
    finally:
        if hook is not None:
            hook.remove()

    parsed_answer = _extract_first_candidate(answer, DEFAULT_CANDIDATES)
    result = {
        "answer": answer,
        "parsed_answer": parsed_answer,
        "correct": parsed_answer == "Joe",
        "visible_tokens": visible_tokens,
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "prompt_tokens": len(tokenizer.encode(prompt)),
    }
    if effect:
        result.update(effect)
    return result


def _candidate_token_id(tokenizer, candidate: str) -> int:
    token_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Candidate {candidate!r} encoded to no tokens.")
    return token_ids[0]


def _extract_first_candidate(answer: str, candidates: Sequence[str]) -> Optional[str]:
    lowered = answer.lower()
    earliest: tuple[int, str] | None = None
    for candidate in candidates:
        idx = lowered.find(candidate.lower())
        if idx == -1:
            continue
        if earliest is None or idx < earliest[0]:
            earliest = (idx, candidate)
    return earliest[1] if earliest is not None else None


def _score_next_token_candidates(
    model,
    tokenizer,
    prompt: str,
    device: str,
    candidates: Sequence[str],
    vector: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    layer: int = 10,
    site: str = "mlp_out",
    burst_steps: int = 1,
    inject_mode: str = "add",
) -> tuple[list[dict], dict]:
    """Score candidate names by their next-token logit after `prompt`."""
    hook = None
    effect: dict = {}
    if vector is not None and alpha != 0.0:
        hook, effect = answer_start_injection(
            model, vector, alpha, layer, site=site, burst_steps=burst_steps,
            mode=inject_mode,
        )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        if hook is not None:
            hook.remove()

    logits = outputs.logits[0, -1, :].detach().to(torch.float32)
    logprobs = torch.log_softmax(logits, dim=-1)

    rows = []
    for candidate in candidates:
        token_id = _candidate_token_id(tokenizer, candidate)
        logit = float(logits[token_id].cpu())
        rows.append({
            "candidate": candidate,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "logit": logit,
            "logprob": float(logprobs[token_id].cpu()),
            "rank": int(torch.sum(logits > logits[token_id]).item()) + 1,
        })

    best_candidate_logit = max(row["logit"] for row in rows)
    for row in rows:
        best_other = max(
            other["logit"] for other in rows if other["candidate"] != row["candidate"]
        )
        row["candidate_margin"] = row["logit"] - best_other
        row["top_candidate"] = row["logit"] == best_candidate_logit
    rows.sort(key=lambda row: row["logit"], reverse=True)
    return rows, effect


def _register_payload_hook(
    model,
    payload: torch.Tensor,
    layer_idx: int,
    site: str = "mlp_out",
    alpha: float = 1.0,
    mode: str = "add",
    burst_steps: int = 1,
):
    layers = _get_decoder_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {len(layers)} layers."
        )

    target_layer = layers[layer_idx]
    if mode not in {"add", "replace"}:
        raise ValueError(f"Unsupported injection mode {mode!r}; choose 'add' or 'replace'.")
    call_count = {"n": 0}

    def hook_fn(module, inputs, outputs):
        call_count["n"] += 1
        if call_count["n"] > burst_steps:
            return outputs
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.dim() != 3:
            return outputs
        edited = hidden.clone()
        vector = payload.to(device=edited.device, dtype=edited.dtype)
        if mode == "replace":
            edited[:, -1, :] = vector
        else:
            edited[:, -1, :] = edited[:, -1, :] + alpha * vector
        if isinstance(outputs, tuple):
            return (edited,) + outputs[1:]
        return edited

    if site == "mlp_out":
        if not hasattr(target_layer, "mlp"):
            raise RuntimeError(f"No `mlp` sub-module on layer {layer_idx}.")
        return target_layer.mlp.register_forward_hook(hook_fn)
    if site == "attn_out":
        if not hasattr(target_layer, "self_attn"):
            raise RuntimeError(f"No `self_attn` sub-module on layer {layer_idx}.")
        return target_layer.self_attn.register_forward_hook(hook_fn)
    raise ValueError(f"Unsupported site {site!r}; choose 'mlp_out' or 'attn_out'.")


def _next_token_logits_with_payload(
    model,
    tokenizer,
    prompt: str,
    device: str,
    payload: torch.Tensor,
    layer: int,
    site: str,
    alpha: float = 1.0,
    mode: str = "add",
) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = _register_payload_hook(
        model,
        payload,
        layer_idx=layer,
        site=site,
        alpha=alpha,
        mode=mode,
    )
    try:
        outputs = model(**inputs)
    finally:
        handle.remove()
    return outputs.logits[0, -1, :]


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return the first index of ``needle`` in ``haystack`` or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for idx in range(len(haystack) - len(needle) + 1):
        if list(haystack[idx:idx + len(needle)]) == list(needle):
            return idx
    return -1


def _find_target_token_position(tokenizer, prompt: str, target: str) -> tuple[int, list[int]]:
    """Find the target-name token span inside the donor prompt.

    In the donor string ``Bob is the father of {target}.``, Qwen tokenizes the
    name as a leading-space token such as ``" Joe"``. Searching that form keeps
    the trailing period intact while avoiding the period-position washout.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(f" {target}", add_special_tokens=False)
    start = _find_subsequence(prompt_ids, target_ids)
    if start < 0:
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        start = _find_subsequence(prompt_ids, target_ids)
    if start < 0:
        raise ValueError(
            f"Could not locate target token(s) for {target!r} in donor prompt."
        )
    return start + len(target_ids) - 1, prompt_ids


def _find_fact_target_token_position(tokenizer, prompt: str, target: str) -> tuple[int, list[int]]:
    """Find the target-name token inside the explicit father fact in a prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    fact_prompt = f"Bob is the father of {target}."
    fact_ids = tokenizer.encode(fact_prompt, add_special_tokens=False)
    fact_start = _find_subsequence(prompt_ids, fact_ids)
    if fact_start < 0:
        raise ValueError(f"Could not locate father fact for {target!r} in prompt.")
    fact_target_pos, _ = _find_target_token_position(tokenizer, fact_prompt, target)
    return fact_start + fact_target_pos, prompt_ids


def _capture_prompt_activation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    layer: int,
    site: str,
    position: int,
) -> torch.Tensor:
    layers = _get_decoder_layers(model)
    if layer >= len(layers):
        raise ValueError(
            f"layer={layer} out of range for model with {len(layers)} layers."
        )
    target_layer = layers[layer]
    if site == "mlp_out":
        if not hasattr(target_layer, "mlp"):
            raise RuntimeError(f"No `mlp` sub-module on layer {layer}.")
        module = target_layer.mlp
    elif site == "attn_out":
        if not hasattr(target_layer, "self_attn"):
            raise RuntimeError(f"No `self_attn` sub-module on layer {layer}.")
        module = target_layer.self_attn
    elif site == "layer_out":
        module = target_layer
    else:
        raise ValueError(
            f"Unsupported site {site!r}; choose 'mlp_out', 'attn_out', or 'layer_out'."
        )

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.dim() == 3 and position < hidden.size(1):
            captured["payload"] = (
                hidden[0, position, :]
                .detach()
                .clone()
                .to(dtype=torch.float32, device="cpu")
            )

    handle = module.register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if position >= inputs["input_ids"].shape[-1]:
            raise ValueError(
                f"Patch position {position} out of range for prompt length "
                f"{inputs['input_ids'].shape[-1]}."
            )
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if "payload" not in captured:
        raise RuntimeError("Forward pass did not capture the requested activation.")
    return captured["payload"]


def _register_position_patch_hook(
    model,
    payload: torch.Tensor,
    layer_idx: int,
    position: int,
    site: str = "mlp_out",
):
    layers = _get_decoder_layers(model)
    if layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range for model with {len(layers)} layers."
        )

    target_layer = layers[layer_idx]

    def hook_fn(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.dim() != 3 or position >= hidden.size(1):
            return outputs
        edited = hidden.clone()
        edited[:, position, :] = payload.to(device=edited.device, dtype=edited.dtype)
        if isinstance(outputs, tuple):
            return (edited,) + outputs[1:]
        return edited

    if site == "mlp_out":
        if not hasattr(target_layer, "mlp"):
            raise RuntimeError(f"No `mlp` sub-module on layer {layer_idx}.")
        return target_layer.mlp.register_forward_hook(hook_fn)
    if site == "attn_out":
        if not hasattr(target_layer, "self_attn"):
            raise RuntimeError(f"No `self_attn` sub-module on layer {layer_idx}.")
        return target_layer.self_attn.register_forward_hook(hook_fn)
    if site == "layer_out":
        return target_layer.register_forward_hook(hook_fn)
    raise ValueError(
        f"Unsupported site {site!r}; choose 'mlp_out', 'attn_out', or 'layer_out'."
    )


def _generate_with_position_patch(
    model,
    tokenizer,
    prompt: str,
    device: str,
    payload: torch.Tensor,
    layer: int,
    site: str,
    position: int,
    max_new_tokens: int = 12,
) -> str:
    handle = _register_position_patch_hook(
        model, payload, layer_idx=layer, position=position, site=site
    )
    try:
        return _generate(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
    finally:
        handle.remove()


def _score_next_token_candidates_with_position_patch(
    model,
    tokenizer,
    prompt: str,
    device: str,
    candidates: Sequence[str],
    payload: torch.Tensor,
    layer: int,
    site: str,
    position: int,
) -> list[dict]:
    handle = _register_position_patch_hook(
        model, payload, layer_idx=layer, position=position, site=site
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
    finally:
        handle.remove()

    logits = outputs.logits[0, -1, :].detach().to(torch.float32)
    logprobs = torch.log_softmax(logits, dim=-1)
    rows = []
    for candidate in candidates:
        token_id = _candidate_token_id(tokenizer, candidate)
        rows.append({
            "candidate": candidate,
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "logit": float(logits[token_id].cpu()),
            "logprob": float(logprobs[token_id].cpu()),
            "rank": int(torch.sum(logits > logits[token_id]).item()) + 1,
        })
    best_candidate_logit = max(row["logit"] for row in rows)
    for row in rows:
        best_other = max(
            other["logit"] for other in rows if other["candidate"] != row["candidate"]
        )
        row["candidate_margin"] = row["logit"] - best_other
        row["top_candidate"] = row["logit"] == best_candidate_logit
    rows.sort(key=lambda row: row["logit"], reverse=True)
    return rows


def extract_donor_payload(
    model,
    tokenizer,
    target: str,
    device: str,
    layer: int = 10,
    site: str = "mlp_out",
) -> tuple[torch.Tensor, str, dict]:
    """Capture the target-token donor hidden state for a fact prompt."""
    if site != "mlp_out":
        raise ValueError("Donor extraction is defined for the layer MLP output site.")

    layers = _get_decoder_layers(model)
    if layer >= len(layers):
        raise ValueError(
            f"layer={layer} out of range for model with {len(layers)} layers."
        )
    target_layer = layers[layer]
    if not hasattr(target_layer, "mlp"):
        raise RuntimeError(f"No `mlp` sub-module on layer {layer}.")

    prompt = f"Bob is the father of {target}."
    extraction_pos, prompt_ids = _find_target_token_position(tokenizer, prompt, target)
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.dim() == 3:
            captured["payload"] = (
                hidden[0, extraction_pos, :]
                .detach()
                .clone()
                .to(dtype=torch.float32, device="cpu")
            )

    handle = target_layer.mlp.register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    if "payload" not in captured:
        raise RuntimeError("Donor forward pass did not capture an MLP hidden state.")
    meta = {
        "extraction_position": extraction_pos,
        "extraction_token_id": prompt_ids[extraction_pos],
        "extraction_token_text": tokenizer.decode([prompt_ids[extraction_pos]]),
        "prompt_token_ids": prompt_ids,
    }
    return captured["payload"], prompt, meta


def cache_patched_payloads(
    model_path: str,
    out_dir: Path,
    targets: Sequence[str] = BASELINE_TARGETS,
    layer: int = 10,
    site: str = "mlp_out",
    device_arg: str | None = None,
    dtype_arg: str | None = None,
) -> Path:
    """Cache donor-run hidden states as ``{target}_patched_payload.npy`` files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)

    rows = []
    for target in targets:
        LOG.info("Caching donor payload target=%s", target)
        model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)
        for param in model.parameters():
            param.requires_grad_(False)

        target_payloads_dir = out_dir / target.lower() / "payloads"
        target_prompts_dir = out_dir / target.lower() / "prompts"
        target_payloads_dir.mkdir(parents=True, exist_ok=True)
        target_prompts_dir.mkdir(parents=True, exist_ok=True)

        payload, donor_prompt, donor_meta = extract_donor_payload(
            model, tokenizer, target, device, layer=layer, site=site
        )
        payload_path = target_payloads_dir / f"{target.lower()}_patched_payload.npy"
        np.save(payload_path, payload.numpy().astype(np.float32))
        compatibility_payload_path = target_payloads_dir / f"{target.lower()}_payload.npy"
        np.save(compatibility_payload_path, payload.numpy().astype(np.float32))
        prompt_meta = _save_prompt(
            target_prompts_dir, f"{target.lower()}_donor", donor_prompt, tokenizer
        )
        rows.append({
            "target": target,
            "donor_prompt": donor_prompt,
            "donor_prompt_sha256": prompt_meta["prompt_sha256"],
            "extraction_position": donor_meta["extraction_position"],
            "extraction_token_id": donor_meta["extraction_token_id"],
            "extraction_token_text": donor_meta["extraction_token_text"],
            "payload_path": str(payload_path),
            "payload_sha256": _sha256_file(payload_path),
            "payload_norm": float(torch.linalg.vector_norm(payload).cpu()),
        })
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    manifest_path = out_dir / "patched_payload_cache_manifest.json"
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "targets": list(targets),
        "layer": layer,
        "site": site,
        "method": "donor_activation_patch",
        "rows": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest_path


def _render_report(
    out_dir: Path,
    model_path: str,
    site: str,
    primary_layer: int,
    summary_rows: list,
    chosen_latent: dict,
) -> None:
    lines = []
    lines.append("# Latent-bus demo report\n")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Injection site: `{site}` at layer {primary_layer}")
    if chosen_latent:
        lines.append(
            f"- Chosen alpha: {chosen_latent.get('alpha')} "
            f"(layer {chosen_latent.get('layer')})"
        )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Condition | Accuracy | Visible tokens | Alpha | Layer |")
    lines.append("|-----------|----------|----------------|-------|-------|")
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['acc']:.2f} | {row['visible_tokens']} | "
            f"{row.get('chosen_alpha', 0.0)} | {row.get('layer', primary_layer)} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Decoding is greedy (`do_sample=False`). Each condition runs once per configuration; "
        "accuracy is 0 or 1 per trial. Multi-seed bootstraps are left as future work."
    )
    lines.append(
        "- Drift and crosstalk gating are part of the design but are not implemented "
        "in this reference harness."
    )
    (out_dir / "report.md").write_text("\n".join(lines))


def run_demo(
    model_path: str,
    in_dir: Path,
    alpha_grid: Iterable[float] = (0.5, 1.0),
    site: str = "mlp_out",
    layer: int = 10,
    fallback_layer: Optional[int] = 12,
    burst_steps: int = 1,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
) -> Path:
    """Run Null / Token / Latent conditions and write outputs into `in_dir/results/`.

    The Latent condition tries each alpha at `layer`; if none produce the
    expected answer, it retries at `fallback_layer` (set to None to skip).
    Returns the path to the summary CSV.
    """
    in_dir = Path(in_dir)
    results_dir = in_dir / "results"
    prompts_dir = results_dir / "prompts"
    results_dir.mkdir(parents=True, exist_ok=True)

    query = (in_dir / "query.txt").read_text().strip()
    fact = (in_dir / "fact_F.txt").read_text().strip()
    vector_path = in_dir / "vectors" / "v_F.npy"
    vector = torch.from_numpy(np.load(vector_path))

    alpha_grid = list(alpha_grid)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)
    trials = []
    trial_prompts = {}

    LOG.info("Running NULL condition")
    null_r = _evaluate(model, tokenizer, query, device)
    null_trial = {"condition": "Null", "alpha": 0.0, "layer": layer, "site": site,
                  "inject_mode": "none", **null_r}
    trial_prompts["Null"] = null_trial["prompt"]
    trials.append(_attach_prompt_artifact(null_trial, prompts_dir, tokenizer))

    LOG.info("Running TOKEN condition")
    token_r = _evaluate(model, tokenizer, query, device, fact=fact)
    token_trial = {"condition": "Token", "alpha": 0.0, "layer": layer, "site": site,
                   "inject_mode": "none", **token_r}
    trial_prompts["Token"] = token_trial["prompt"]
    trials.append(_attach_prompt_artifact(token_trial, prompts_dir, tokenizer))

    LOG.info("Running LATENT condition (alpha grid: %s)", alpha_grid)
    best: Optional[dict] = None
    best_alpha: float = 0.0
    best_layer: int = layer

    layers_to_try = [layer] + ([fallback_layer] if fallback_layer is not None else [])
    for L in layers_to_try:
        for alpha in alpha_grid:
            r = _evaluate(model, tokenizer, query, device,
                          vector=vector, alpha=alpha, layer=L, site=site,
                          burst_steps=burst_steps)
            LOG.info("  layer %d, alpha %.2f -> answer=%r correct=%s",
                     L, alpha, r["answer"], r["correct"])
            if r["correct"] and (best is None or alpha < best_alpha):
                best = r
                best_alpha = alpha
                best_layer = L
        if best is not None:
            break

    if best is None:
        # No alpha produced the expected answer — record the last trial anyway.
        best = r
        best_alpha = alpha
        best_layer = L

    latent_trial = {
        "condition": "Latent",
        "alpha": best_alpha,
        "layer": best_layer,
        "site": site,
        "inject_mode": "answer_start",
        "burst_steps": burst_steps,
        "visible_tokens": 0,
        "answer": best["answer"],
        "correct": best["correct"],
        "prompt": best["prompt"],
        "prompt_sha256": best["prompt_sha256"],
        "prompt_tokens": best["prompt_tokens"],
    }
    for k in ("proj_before0", "proj_after0", "delta_proj0"):
        if k in best:
            latent_trial[k] = best[k]
    trial_prompts["Latent"] = latent_trial["prompt"]
    trials.append(_attach_prompt_artifact(latent_trial, prompts_dir, tokenizer))

    logit_rows = []
    logit_specs = {
        "Null": (trial_prompts["Null"], None, 0.0, layer, "none"),
        "Token": (trial_prompts["Token"], None, 0.0, layer, "none"),
        "Latent": (trial_prompts["Latent"], vector, best_alpha, best_layer, "answer_start"),
    }
    for condition, (prompt, condition_vector, condition_alpha, condition_layer, inject_mode) in logit_specs.items():
        scored, effect = _score_next_token_candidates(
            model,
            tokenizer,
            prompt,
            device,
            DEFAULT_CANDIDATES,
            vector=condition_vector,
            alpha=condition_alpha,
            layer=condition_layer,
            site=site,
            burst_steps=burst_steps,
        )
        for row in scored:
            logit_rows.append({
                "condition": condition,
                "candidate": row["candidate"],
                "token_id": row["token_id"],
                "token_text": row["token_text"],
                "logit": row["logit"],
                "logprob": row["logprob"],
                "rank": row["rank"],
                "candidate_margin": row["candidate_margin"],
                "top_candidate": row["top_candidate"],
                "alpha": condition_alpha,
                "layer": condition_layer,
                "site": site,
                "inject_mode": inject_mode,
                "prompt_sha256": _sha256_text(prompt),
                "proj_before0": effect.get("proj_before0"),
                "proj_after0": effect.get("proj_after0"),
                "delta_proj0": effect.get("delta_proj0"),
            })

    logit_csv = results_dir / "candidate_logits.csv"
    with logit_csv.open("w", newline="") as fh:
        fieldnames = [
            "condition", "candidate", "token_id", "token_text", "logit",
            "logprob", "rank", "candidate_margin", "top_candidate", "alpha",
            "layer", "site", "inject_mode", "prompt_sha256", "proj_before0",
            "proj_after0", "delta_proj0",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logit_rows)

    # Write per-trial JSONL.
    with (results_dir / "trials.jsonl").open("w") as fh:
        for t in trials:
            fh.write(json.dumps(t) + "\n")

    # Summary CSV.
    summary_rows = []
    for cond in ("Null", "Token", "Latent"):
        t = next(tr for tr in trials if tr["condition"] == cond)
        summary_rows.append({
            "condition": cond,
            "acc": 1.0 if t["correct"] else 0.0,
            "visible_tokens": t["visible_tokens"],
            "chosen_alpha": t.get("alpha", 0.0),
            "layer": t.get("layer", layer),
            "site": site,
            "inject_mode": t.get("inject_mode", "none"),
        })
    summary_csv = results_dir / "summary.csv"
    with summary_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Manifest.
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "site": site,
        "primary_layer": layer,
        "fallback_layer": fallback_layer,
        "alpha_grid": alpha_grid,
        "burst_steps": burst_steps,
        "decoding": "greedy",
        "chosen_alpha": best_alpha,
        "chosen_layer": best_layer,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "vector_path": str(vector_path),
        "vector_sha256": _sha256_file(vector_path),
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "prompt_hashes": {
            condition: _sha256_text(prompt)
            for condition, prompt in trial_prompts.items()
        },
        "candidate_logits_path": str(logit_csv),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    _render_report(in_dir, model_path, site, layer, summary_rows, {
        "alpha": best_alpha, "layer": best_layer,
    })

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return summary_csv


def run_logit_probe(
    model_path: str,
    in_dir: Path,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    alpha: float = 1.0,
    site: str = "mlp_out",
    layer: int = 10,
    burst_steps: int = 1,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
) -> Path:
    """Write candidate next-token logits for Null, Token, and Latent prompts."""
    in_dir = Path(in_dir)
    results_dir = in_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    query = (in_dir / "query.txt").read_text().strip()
    fact = (in_dir / "fact_F.txt").read_text().strip()
    vector = torch.from_numpy(np.load(in_dir / "vectors" / "v_F.npy"))

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)

    prompts = {
        "Null": (f"Question: {query}\nAnswer:", None, 0.0, "none"),
        "Token": (TOKEN_PROMPT_TEMPLATE.format(fact=fact, query=query), None, 0.0, "none"),
        "Latent": (f"Question: {query}\nAnswer:", vector, alpha, "answer_start"),
    }

    rows = []
    for condition, (prompt, condition_vector, condition_alpha, inject_mode) in prompts.items():
        LOG.info("Scoring %s candidate logits", condition.upper())
        scored, effect = _score_next_token_candidates(
            model,
            tokenizer,
            prompt,
            device,
            candidates,
            vector=condition_vector,
            alpha=condition_alpha,
            layer=layer,
            site=site,
            burst_steps=burst_steps,
        )
        for row in scored:
            rows.append({
                "condition": condition,
                "candidate": row["candidate"],
                "token_id": row["token_id"],
                "token_text": row["token_text"],
                "logit": row["logit"],
                "logprob": row["logprob"],
                "rank": row["rank"],
                "candidate_margin": row["candidate_margin"],
                "top_candidate": row["top_candidate"],
                "alpha": condition_alpha,
                "layer": layer,
                "site": site,
                "inject_mode": inject_mode,
                "proj_before0": effect.get("proj_before0"),
                "proj_after0": effect.get("proj_after0"),
                "delta_proj0": effect.get("delta_proj0"),
            })

    probe_csv = results_dir / "logit_probe.csv"
    with probe_csv.open("w", newline="") as fh:
        fieldnames = [
            "condition",
            "candidate",
            "token_id",
            "token_text",
            "logit",
            "logprob",
            "rank",
            "candidate_margin",
            "top_candidate",
            "alpha",
            "layer",
            "site",
            "inject_mode",
            "proj_before0",
            "proj_after0",
            "delta_proj0",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return probe_csv


def run_kinship_baseline(
    model_path: str,
    out_dir: Path,
    targets: Sequence[str] = BASELINE_TARGETS,
    candidates: Sequence[str] = BASELINE_ANSWER_CANDIDATES,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    max_new_tokens: int = 12,
) -> Path:
    """Evaluate whether visible facts solve a small kinship family."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)
    multi_token_candidates = [
        candidate for candidate in candidates
        if len(tokenizer.encode(candidate, add_special_tokens=False)) != 1
    ]
    if multi_token_candidates:
        raise ValueError(
            "Baseline candidate logits require single-token names; multi-token: "
            + ", ".join(multi_token_candidates)
        )

    trials = []
    logit_rows = []
    for target in targets:
        for condition, include_fact in (("Null", False), ("Token", True)):
            prompt, fact = _build_kinship_baseline_prompt(target, candidates, include_fact)
            prompt_name = f"{target.lower()}_{condition.lower()}"
            prompt_meta = _save_prompt(prompts_dir, prompt_name, prompt, tokenizer)
            answer = _generate(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
            )
            parsed_answer = _extract_first_candidate(answer, candidates)
            scored, _ = _score_next_token_candidates(
                model,
                tokenizer,
                prompt,
                device,
                candidates,
            )
            target_row = next(row for row in scored if row["candidate"] == target)
            top_row = scored[0]
            trial = {
                "condition": condition,
                "target": target,
                "fact": fact if include_fact else "",
                "answer": answer,
                "parsed_answer": parsed_answer,
                "correct": parsed_answer == target,
                "top_candidate": top_row["candidate"],
                "top_candidate_correct": top_row["candidate"] == target,
                "target_rank": target_row["rank"],
                "target_margin": target_row["candidate_margin"],
                "visible_tokens": len(tokenizer.encode(fact)) if include_fact else 0,
                **prompt_meta,
            }
            trials.append(trial)
            for row in scored:
                logit_rows.append({
                    "condition": condition,
                    "target": target,
                    "candidate": row["candidate"],
                    "token_id": row["token_id"],
                    "token_text": row["token_text"],
                    "logit": row["logit"],
                    "logprob": row["logprob"],
                    "rank": row["rank"],
                    "candidate_margin": row["candidate_margin"],
                    "top_candidate": row["top_candidate"],
                    "prompt_sha256": prompt_meta["prompt_sha256"],
                })

    trials_path = out_dir / "baseline_trials.jsonl"
    with trials_path.open("w") as fh:
        for trial in trials:
            fh.write(json.dumps(trial) + "\n")

    summary_rows = []
    for condition in ("Null", "Token"):
        condition_trials = [trial for trial in trials if trial["condition"] == condition]
        n = len(condition_trials)
        summary_rows.append({
            "condition": condition,
            "n": n,
            "generated_acc": sum(trial["correct"] for trial in condition_trials) / n,
            "top_candidate_acc": sum(trial["top_candidate_correct"] for trial in condition_trials) / n,
            "mean_target_rank": sum(trial["target_rank"] for trial in condition_trials) / n,
            "mean_target_margin": sum(trial["target_margin"] for trial in condition_trials) / n,
            "mean_visible_tokens": sum(trial["visible_tokens"] for trial in condition_trials) / n,
        })

    summary_path = out_dir / "baseline_summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    logits_path = out_dir / "baseline_logits.csv"
    with logits_path.open("w", newline="") as fh:
        fieldnames = [
            "condition", "target", "candidate", "token_id", "token_text",
            "logit", "logprob", "rank", "candidate_margin", "top_candidate",
            "prompt_sha256",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logit_rows)

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "targets": list(targets),
        "candidates": list(candidates),
        "decoding": "greedy",
        "max_new_tokens": max_new_tokens,
        "trials_path": str(trials_path),
        "summary_path": str(summary_path),
        "logits_path": str(logits_path),
    }
    (out_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2))

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return summary_path


def run_aligned_patch_control(
    model_path: str,
    out_dir: Path,
    targets: Sequence[str] = BASELINE_TARGETS,
    candidates: Sequence[str] = BASELINE_ANSWER_CANDIDATES,
    layer: int = 10,
    site: str = "mlp_out",
    variants: Sequence[str] = ("train",),
    patch_position: str = "answer",
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    max_new_tokens: int = 12,
) -> Path:
    """Run a standard clean/corrupted aligned activation-patching control.

    Clean and corrupted prompts are full Token-condition kinship prompts that
    differ only in the father-fact target name. By default, the clean
    answer-start activation is patched into the aligned answer-start position
    in the corrupted prompt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "aligned_patch_prompts"
    prompts_dir.mkdir(exist_ok=True)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    multi_token_candidates = [
        candidate for candidate in candidates
        if len(tokenizer.encode(candidate, add_special_tokens=False)) != 1
    ]
    if multi_token_candidates:
        raise ValueError(
            "Aligned patch candidates require single-token names; multi-token: "
            + ", ".join(multi_token_candidates)
        )

    if patch_position not in {"answer", "fact_target"}:
        raise ValueError("patch_position must be 'answer' or 'fact_target'.")

    target_list = list(targets)
    target_pairs = [
        (clean_target, corrupt_target)
        for clean_target in target_list
        for corrupt_target in target_list
        if clean_target != corrupt_target
    ]
    rows = []
    logit_rows = []
    for clean_target, corrupt_target in target_pairs:
        for variant in variants:
            clean_prompt, clean_fact = _build_kinship_baseline_prompt(
                clean_target, candidates, include_fact=True, variant=variant
            )
            corrupt_prompt, corrupt_fact = _build_kinship_baseline_prompt(
                corrupt_target, candidates, include_fact=True, variant=variant
            )
            clean_meta = _save_prompt(
                prompts_dir,
                f"{variant}_{clean_target.lower()}_clean",
                clean_prompt,
                tokenizer,
            )
            corrupt_meta = _save_prompt(
                prompts_dir,
                f"{variant}_{clean_target.lower()}_corrupt_{corrupt_target.lower()}",
                corrupt_prompt,
                tokenizer,
            )
            clean_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
            corrupt_ids = tokenizer.encode(corrupt_prompt, add_special_tokens=False)
            if patch_position == "answer":
                clean_pos = len(clean_ids) - 1
                corrupt_pos = len(corrupt_ids) - 1
            else:
                clean_pos, clean_ids = _find_fact_target_token_position(
                    tokenizer, clean_prompt, clean_target
                )
                corrupt_pos, corrupt_ids = _find_fact_target_token_position(
                    tokenizer, corrupt_prompt, corrupt_target
                )
            if clean_pos != corrupt_pos or len(clean_ids) != len(corrupt_ids):
                raise ValueError(
                    "Aligned patch prompts must have matching fact target positions "
                    "and token lengths."
                )

            payload = _capture_prompt_activation(
                model,
                tokenizer,
                clean_prompt,
                device,
                layer=layer,
                site=site,
                position=clean_pos,
            )

            clean_answer = _generate(
                model, tokenizer, clean_prompt, device, max_new_tokens=max_new_tokens
            )
            corrupt_answer = _generate(
                model, tokenizer, corrupt_prompt, device, max_new_tokens=max_new_tokens
            )
            patched_answer = _generate_with_position_patch(
                model,
                tokenizer,
                corrupt_prompt,
                device,
                payload,
                layer=layer,
                site=site,
                position=corrupt_pos,
                max_new_tokens=max_new_tokens,
            )

            clean_scores, _ = _score_next_token_candidates(
                model, tokenizer, clean_prompt, device, candidates
            )
            corrupt_scores, _ = _score_next_token_candidates(
                model, tokenizer, corrupt_prompt, device, candidates
            )
            patched_scores = _score_next_token_candidates_with_position_patch(
                model,
                tokenizer,
                corrupt_prompt,
                device,
                candidates,
                payload,
                layer=layer,
                site=site,
                position=corrupt_pos,
            )

            condition_specs = (
                ("Clean", clean_prompt, clean_answer, clean_scores, clean_meta),
                ("Corrupt", corrupt_prompt, corrupt_answer, corrupt_scores, corrupt_meta),
                ("AlignedPatch", corrupt_prompt, patched_answer, patched_scores, corrupt_meta),
            )
            condition_diffs = {}
            condition_payloads = []
            for condition, prompt, answer, scores, meta in condition_specs:
                parsed = _extract_first_candidate(answer, candidates)
                clean_row = next(row for row in scores if row["candidate"] == clean_target)
                corrupt_row = next(row for row in scores if row["candidate"] == corrupt_target)
                logit_diff = clean_row["logit"] - corrupt_row["logit"]
                condition_diffs[condition] = logit_diff
                condition_payloads.append({
                    "variant": variant,
                    "condition": condition,
                    "clean_target": clean_target,
                    "corrupt_target": corrupt_target,
                    "clean_fact": clean_fact,
                    "corrupt_fact": corrupt_fact,
                    "answer": answer,
                    "parsed_answer": parsed,
                    "generated_clean_correct": parsed == clean_target,
                    "generated_corrupt_correct": parsed == corrupt_target,
                    "top_candidate": scores[0]["candidate"],
                    "top_clean_correct": scores[0]["candidate"] == clean_target,
                    "top_corrupt_correct": scores[0]["candidate"] == corrupt_target,
                    "clean_target_rank": clean_row["rank"],
                    "clean_target_margin": clean_row["candidate_margin"],
                    "corrupt_target_rank": corrupt_row["rank"],
                    "corrupt_target_margin": corrupt_row["candidate_margin"],
                    "logit_diff_clean_vs_corrupt": logit_diff,
                    "patch_position": corrupt_pos,
                    "patch_token_text": tokenizer.decode([corrupt_ids[corrupt_pos]]),
                    "patch_position_kind": patch_position,
                    "layer": layer,
                    "site": site,
                    "prompt_sha256": meta["prompt_sha256"],
                })
                for score in scores:
                    logit_rows.append({
                        "variant": variant,
                        "condition": condition,
                        "clean_target": clean_target,
                        "corrupt_target": corrupt_target,
                        "candidate": score["candidate"],
                        "token_id": score["token_id"],
                        "token_text": score["token_text"],
                        "logit": score["logit"],
                        "logprob": score["logprob"],
                        "rank": score["rank"],
                        "candidate_margin": score["candidate_margin"],
                        "top_candidate": score["top_candidate"],
                        "patch_position": corrupt_pos,
                        "layer": layer,
                        "site": site,
                        "prompt_sha256": _sha256_text(prompt),
                    })
            denom = condition_diffs["Clean"] - condition_diffs["Corrupt"]
            for row in condition_payloads:
                if denom == 0:
                    row["recovery"] = None
                else:
                    row["recovery"] = (
                        row["logit_diff_clean_vs_corrupt"] - condition_diffs["Corrupt"]
                    ) / denom
                rows.append(row)

    control_path = out_dir / "aligned_patch_control.csv"
    with control_path.open("w", newline="") as fh:
        fieldnames = [
            "variant", "condition", "clean_target", "corrupt_target",
            "clean_fact", "corrupt_fact", "answer", "parsed_answer",
            "generated_clean_correct", "generated_corrupt_correct",
            "top_candidate", "top_clean_correct", "top_corrupt_correct",
            "clean_target_rank", "clean_target_margin", "corrupt_target_rank",
            "corrupt_target_margin", "logit_diff_clean_vs_corrupt", "recovery",
            "patch_position", "patch_token_text", "patch_position_kind",
            "layer", "site", "prompt_sha256",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logits_path = out_dir / "aligned_patch_logits.csv"
    with logits_path.open("w", newline="") as fh:
        fieldnames = [
            "variant", "condition", "clean_target", "corrupt_target",
            "candidate", "token_id", "token_text", "logit", "logprob",
            "rank", "candidate_margin", "top_candidate", "patch_position",
            "layer", "site", "prompt_sha256",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logit_rows)

    summary_rows = []
    for condition in ("Clean", "Corrupt", "AlignedPatch"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        recoveries = [
            float(row["recovery"])
            for row in condition_rows
            if row["recovery"] is not None
        ]
        n = len(condition_rows)
        summary_rows.append({
            "condition": condition,
            "n": n,
            "generated_clean_acc": (
                sum(row["generated_clean_correct"] for row in condition_rows) / n
                if n else None
            ),
            "generated_corrupt_acc": (
                sum(row["generated_corrupt_correct"] for row in condition_rows) / n
                if n else None
            ),
            "top_clean_acc": (
                sum(row["top_clean_correct"] for row in condition_rows) / n
                if n else None
            ),
            "top_corrupt_acc": (
                sum(row["top_corrupt_correct"] for row in condition_rows) / n
                if n else None
            ),
            "mean_recovery": (
                sum(recoveries) / len(recoveries) if recoveries else None
            ),
        })

    summary_path = out_dir / "aligned_patch_summary.csv"
    with summary_path.open("w", newline="") as fh:
        fieldnames = [
            "condition", "n", "generated_clean_acc", "generated_corrupt_acc",
            "top_clean_acc", "top_corrupt_acc", "mean_recovery",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    patched_rows = [row for row in rows if row["condition"] == "AlignedPatch"]
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "targets": target_list,
        "target_pairs": target_pairs,
        "candidates": list(candidates),
        "variants": list(variants),
        "layer": layer,
        "site": site,
        "patch_position": patch_position,
        "method": "standard_aligned_activation_patching_positive_control",
        "pairing": "all_ordered_clean_corrupt_pairs",
        "max_new_tokens": max_new_tokens,
        "aligned_patch_generated_clean_acc": (
            sum(row["generated_clean_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
        "aligned_patch_top_clean_acc": (
            sum(row["top_clean_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
        "control_path": str(control_path),
        "logits_path": str(logits_path),
        "summary_path": str(summary_path),
    }
    (out_dir / "aligned_patch_manifest.json").write_text(json.dumps(manifest, indent=2))

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return control_path


def train_single_payload(
    model_path: str,
    out_dir: Path,
    target: str = "Joe",
    candidates: Sequence[str] = BASELINE_ANSWER_CANDIDATES,
    layer: int = 10,
    site: str = "mlp_out",
    steps: int = 120,
    lr: float = 0.05,
    norm_lambda: float = 0.001,
    alpha: float = 1.0,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    max_new_tokens: int = 12,
    log_every: int = 10,
    eval_variants: Sequence[str] = ("train", "paraphrase", "minimal"),
) -> Path:
    """Extract one donor activation payload and evaluate it as a patch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    payloads_dir = out_dir / "payloads"
    payloads_dir.mkdir(exist_ok=True)

    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    multi_token_candidates = [
        candidate for candidate in candidates
        if len(tokenizer.encode(candidate, add_special_tokens=False)) != 1
    ]
    if multi_token_candidates:
        raise ValueError(
            "Payload evaluation requires single-token candidates; multi-token: "
            + ", ".join(multi_token_candidates)
        )
    if target not in candidates:
        raise ValueError(f"Target {target!r} must be in candidates.")

    prompt, fact = _build_kinship_baseline_prompt(
        target, candidates, include_fact=False, variant="train"
    )
    token_prompt, _ = _build_kinship_baseline_prompt(
        target, candidates, include_fact=True, variant="train"
    )
    prompt_meta = _save_prompt(prompts_dir, f"{target.lower()}_patched_train", prompt, tokenizer)
    token_prompt_meta = _save_prompt(prompts_dir, f"{target.lower()}_token_reference", token_prompt, tokenizer)

    payload_path = payloads_dir / f"{target.lower()}_patched_payload.npy"
    compatibility_payload_path = payloads_dir / f"{target.lower()}_payload.npy"
    donor_prompt = f"Bob is the father of {target}."
    extraction_pos, prompt_ids = _find_target_token_position(tokenizer, donor_prompt, target)
    donor_meta = {
        "extraction_position": extraction_pos,
        "extraction_token_id": prompt_ids[extraction_pos],
        "extraction_token_text": tokenizer.decode([prompt_ids[extraction_pos]]),
        "prompt_token_ids": prompt_ids,
    }
    if payload_path.exists():
        payload = torch.from_numpy(np.load(payload_path)).to(dtype=torch.float32)
        if not compatibility_payload_path.exists():
            np.save(compatibility_payload_path, payload.numpy().astype(np.float32))
    else:
        payload, donor_prompt, donor_meta = extract_donor_payload(
            model, tokenizer, target, device, layer=layer, site=site
        )
        payload_np = payload.detach().to(dtype=torch.float32, device="cpu").numpy()
        np.save(payload_path, payload_np.astype(np.float32))
        np.save(compatibility_payload_path, payload_np.astype(np.float32))
    donor_prompt_meta = _save_prompt(
        prompts_dir, f"{target.lower()}_donor", donor_prompt, tokenizer
    )

    before_rows, _ = _score_next_token_candidates(
        model,
        tokenizer,
        prompt,
        device,
        candidates,
    )
    patched_rows, effect = _score_next_token_candidates(
        model,
        tokenizer,
        prompt,
        device,
        candidates,
        vector=payload,
        alpha=alpha,
        layer=layer,
        site=site,
        inject_mode="replace",
    )
    token_rows, _ = _score_next_token_candidates(
        model,
        tokenizer,
        token_prompt,
        device,
        candidates,
    )
    before_target = next(row for row in before_rows if row["candidate"] == target)
    patched_target = next(row for row in patched_rows if row["candidate"] == target)
    trace = [{
        "step": 0,
        "loss": "",
        "ce": "",
        "payload_norm": float(torch.linalg.vector_norm(payload).cpu()),
        "target_logit": patched_target["logit"],
        "target_margin": patched_target["candidate_margin"],
        "target_rank": patched_target["rank"],
        "top_candidate": patched_rows[0]["candidate"],
        "donor_prompt_sha256": donor_prompt_meta["prompt_sha256"],
    }]

    trace_path = out_dir / "payload_train_trace.csv"
    with trace_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)

    with torch.no_grad():
        answer = _generate(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        handle = _register_payload_hook(
            model, payload, layer, site=site, alpha=alpha,
            mode="replace", burst_steps=1,
        )
        try:
            train_patched_answer = _generate(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=max_new_tokens,
            )
        finally:
            handle.remove()

    logits_path = out_dir / "payload_logits.csv"
    with logits_path.open("w", newline="") as fh:
        fieldnames = [
            "condition", "target", "candidate", "token_id", "token_text",
            "logit", "logprob", "rank", "candidate_margin", "top_candidate",
            "inject_mode",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for condition, rows in (
            ("Null", before_rows),
            ("Patched", patched_rows),
            ("Token", token_rows),
        ):
            for row in rows:
                writer.writerow({
                    "condition": condition,
                    "target": target,
                    "inject_mode": "replace" if condition == "Patched" else "none",
                    **row,
                })

    eval_rows = []
    for variant in eval_variants:
        eval_prompt, _ = _build_kinship_baseline_prompt(
            target, candidates, include_fact=False, variant=variant
        )
        eval_token_prompt, _ = _build_kinship_baseline_prompt(
            target, candidates, include_fact=True, variant=variant
        )
        eval_prompt_meta = _save_prompt(
            prompts_dir, f"{target.lower()}_{variant}_null", eval_prompt, tokenizer
        )
        eval_token_prompt_meta = _save_prompt(
            prompts_dir, f"{target.lower()}_{variant}_token", eval_token_prompt, tokenizer
        )

        with torch.no_grad():
            null_answer = _generate(
                model,
                tokenizer,
                eval_prompt,
                device,
                max_new_tokens=max_new_tokens,
            )
            handle = _register_payload_hook(
                model, payload, layer, site=site, alpha=alpha,
                mode="replace", burst_steps=1,
            )
            try:
                patched_answer = _generate(
                    model,
                    tokenizer,
                    eval_prompt,
                    device,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                handle.remove()
            token_answer = _generate(
                model,
                tokenizer,
                eval_token_prompt,
                device,
                max_new_tokens=max_new_tokens,
            )

        for condition, condition_prompt, condition_answer, condition_vector, condition_alpha, prompt_meta_row in (
            ("Null", eval_prompt, null_answer, None, 0.0, eval_prompt_meta),
            ("Patched", eval_prompt, patched_answer, payload, alpha, eval_prompt_meta),
            ("Token", eval_token_prompt, token_answer, None, 0.0, eval_token_prompt_meta),
        ):
            scored, _ = _score_next_token_candidates(
                model,
                tokenizer,
                condition_prompt,
                device,
                candidates,
                vector=condition_vector,
                alpha=condition_alpha,
                layer=layer,
                site=site,
                inject_mode="replace" if condition == "Patched" else "add",
            )
            target_row = next(row for row in scored if row["candidate"] == target)
            eval_rows.append({
                "variant": variant,
                "condition": condition,
                "target": target,
                "answer": condition_answer,
                "parsed_answer": _extract_first_candidate(condition_answer, candidates),
                "generated_correct": _extract_first_candidate(condition_answer, candidates) == target,
                "top_candidate": scored[0]["candidate"],
                "top_candidate_correct": scored[0]["candidate"] == target,
                "target_rank": target_row["rank"],
                "target_margin": target_row["candidate_margin"],
                "prompt_sha256": prompt_meta_row["prompt_sha256"],
            })

    eval_path = out_dir / "payload_eval.csv"
    with eval_path.open("w", newline="") as fh:
        fieldnames = [
            "variant", "condition", "target", "answer", "parsed_answer",
            "generated_correct", "top_candidate", "top_candidate_correct",
            "target_rank", "target_margin", "prompt_sha256",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    patched_eval_rows = [row for row in eval_rows if row["condition"] == "Patched"]
    heldout_patched_rows = [
        row for row in patched_eval_rows if row["variant"] != "train"
    ]

    result = {
        "target": target,
        "fact": fact,
        "donor_prompt": donor_prompt,
        "null_answer": answer,
        "patched_answer": train_patched_answer,
        "latent_answer": train_patched_answer,
        "null_parsed": _extract_first_candidate(answer, candidates),
        "patched_parsed": _extract_first_candidate(train_patched_answer, candidates),
        "latent_parsed": _extract_first_candidate(train_patched_answer, candidates),
        "patched_correct": _extract_first_candidate(train_patched_answer, candidates) == target,
        "latent_correct": _extract_first_candidate(train_patched_answer, candidates) == target,
        "initial_target_margin": before_target["candidate_margin"],
        "final_target_margin": patched_target["candidate_margin"],
        "initial_target_rank": before_target["rank"],
        "final_target_rank": patched_target["rank"],
        "payload_norm": trace[-1]["payload_norm"],
        "proj_before0": effect.get("proj_before0"),
        "proj_after0": effect.get("proj_after0"),
        "delta_proj0": effect.get("delta_proj0"),
        "edit_l2_0": effect.get("edit_l2_0"),
        "prompt_sha256": prompt_meta["prompt_sha256"],
        "token_prompt_sha256": token_prompt_meta["prompt_sha256"],
        "donor_prompt_sha256": donor_prompt_meta["prompt_sha256"],
        "donor_extraction_position": donor_meta["extraction_position"],
        "donor_extraction_token_id": donor_meta["extraction_token_id"],
        "donor_extraction_token_text": donor_meta["extraction_token_text"],
        "payload_path": str(payload_path),
        "payload_sha256": _sha256_file(payload_path),
        "compatibility_payload_path": str(compatibility_payload_path),
        "trace_path": str(trace_path),
        "logits_path": str(logits_path),
        "eval_path": str(eval_path),
        "eval_variants": list(eval_variants),
        "patched_eval_generated_acc": (
            sum(row["generated_correct"] for row in patched_eval_rows) / len(patched_eval_rows)
            if patched_eval_rows else None
        ),
        "latent_eval_generated_acc": (
            sum(row["generated_correct"] for row in patched_eval_rows) / len(patched_eval_rows)
            if patched_eval_rows else None
        ),
        "heldout_patched_generated_acc": (
            sum(row["generated_correct"] for row in heldout_patched_rows) / len(heldout_patched_rows)
            if heldout_patched_rows else None
        ),
        "heldout_latent_generated_acc": (
            sum(row["generated_correct"] for row in heldout_patched_rows) / len(heldout_patched_rows)
            if heldout_patched_rows else None
        ),
        "heldout_patched_top_candidate_acc": (
            sum(row["top_candidate_correct"] for row in heldout_patched_rows) / len(heldout_patched_rows)
            if heldout_patched_rows else None
        ),
        "heldout_latent_top_candidate_acc": (
            sum(row["top_candidate_correct"] for row in heldout_patched_rows) / len(heldout_patched_rows)
            if heldout_patched_rows else None
        ),
    }
    result_path = out_dir / "payload_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "target": target,
        "candidates": list(candidates),
        "layer": layer,
        "site": site,
        "method": "activation_patching",
        "donor_prompt": donor_prompt,
        "donor_extraction": donor_meta,
        "optimizer": "disabled",
        "steps": 0,
        "requested_steps_ignored": steps,
        "lr": None,
        "requested_lr_ignored": lr,
        "norm_lambda": None,
        "requested_norm_lambda_ignored": norm_lambda,
        "alpha": alpha,
        "inject_mode": "replace",
        "burst_steps": 1,
        "max_new_tokens": max_new_tokens,
        "log_every": log_every,
        "eval_variants": list(eval_variants),
        "decoding": "greedy",
        "prompt": prompt_meta,
        "token_reference_prompt": token_prompt_meta,
        "donor_prompt_meta": donor_prompt_meta,
        "result_path": str(result_path),
    }
    (out_dir / "payload_manifest.json").write_text(json.dumps(manifest, indent=2))

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result_path


def train_payload_suite(
    model_path: str,
    out_dir: Path,
    targets: Sequence[str] = BASELINE_TARGETS,
    candidates: Sequence[str] = BASELINE_ANSWER_CANDIDATES,
    layer: int = 10,
    site: str = "mlp_out",
    steps: int = 120,
    lr: float = 0.05,
    norm_lambda: float = 0.001,
    alpha: float = 1.0,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    max_new_tokens: int = 12,
    log_every: int = 10,
    eval_variants: Sequence[str] = ("train", "paraphrase", "minimal"),
) -> Path:
    """Cache one patched payload per target and aggregate the evidence table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_patched_payloads(
        model_path=model_path,
        out_dir=out_dir,
        targets=targets,
        layer=layer,
        site=site,
        device_arg=device_arg,
        dtype_arg=dtype_arg,
    )

    rows = []
    for target in targets:
        target_dir = out_dir / target.lower()
        LOG.info("Evaluating patched payload suite target=%s", target)
        try:
            result_path = train_single_payload(
                model_path=model_path,
                out_dir=target_dir,
                target=target,
                candidates=candidates,
                layer=layer,
                site=site,
                steps=steps,
                lr=lr,
                norm_lambda=norm_lambda,
                alpha=alpha,
                device_arg=device_arg,
                dtype_arg=dtype_arg,
                max_new_tokens=max_new_tokens,
                log_every=log_every,
                eval_variants=eval_variants,
            )
            result = json.loads(result_path.read_text())
            rows.append({
                "target": target,
                "status": "ok",
                "latent_correct": result["latent_correct"],
                "initial_target_rank": result["initial_target_rank"],
                "final_target_rank": result["final_target_rank"],
                "initial_target_margin": result["initial_target_margin"],
                "final_target_margin": result["final_target_margin"],
                "payload_norm": result["payload_norm"],
                "latent_eval_generated_acc": result["latent_eval_generated_acc"],
                "heldout_latent_generated_acc": result["heldout_latent_generated_acc"],
                "heldout_latent_top_candidate_acc": result["heldout_latent_top_candidate_acc"],
                "payload_sha256": result["payload_sha256"],
                "result_path": str(result_path),
                "error": "",
            })
        except Exception as exc:
            LOG.exception("Payload suite target=%s failed", target)
            rows.append({
                "target": target,
                "status": "error",
                "latent_correct": False,
                "initial_target_rank": "",
                "final_target_rank": "",
                "initial_target_margin": "",
                "final_target_margin": "",
                "payload_norm": "",
                "latent_eval_generated_acc": "",
                "heldout_latent_generated_acc": "",
                "heldout_latent_top_candidate_acc": "",
                "payload_sha256": "",
                "result_path": "",
                "error": repr(exc),
            })

    summary_path = out_dir / "payload_suite_summary.csv"
    with summary_path.open("w", newline="") as fh:
        fieldnames = [
            "target", "status", "latent_correct", "initial_target_rank",
            "final_target_rank", "initial_target_margin", "final_target_margin",
            "payload_norm", "latent_eval_generated_acc",
            "heldout_latent_generated_acc", "heldout_latent_top_candidate_acc",
            "payload_sha256", "result_path", "error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "targets": list(targets),
        "candidates": list(candidates),
        "layer": layer,
        "site": site,
        "method": "activation_patching",
        "optimizer": "disabled",
        "steps": 0,
        "requested_steps_ignored": steps,
        "lr": None,
        "requested_lr_ignored": lr,
        "norm_lambda": None,
        "requested_norm_lambda_ignored": norm_lambda,
        "alpha": alpha,
        "inject_mode": "replace",
        "burst_steps": 1,
        "max_new_tokens": max_new_tokens,
        "log_every": log_every,
        "eval_variants": list(eval_variants),
        "n_ok": len(ok_rows),
        "n_error": len(rows) - len(ok_rows),
        "single_prompt_latent_acc": (
            sum(bool(row["latent_correct"]) for row in ok_rows) / len(ok_rows)
            if ok_rows else None
        ),
        "heldout_latent_generated_acc": (
            sum(float(row["heldout_latent_generated_acc"]) for row in ok_rows) / len(ok_rows)
            if ok_rows else None
        ),
        "heldout_latent_top_candidate_acc": (
            sum(float(row["heldout_latent_top_candidate_acc"]) for row in ok_rows) / len(ok_rows)
            if ok_rows else None
        ),
        "summary_path": str(summary_path),
        "cache_manifest_path": str(cache_manifest_path),
    }
    (out_dir / "payload_suite_manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary_path


def probe_payload_suite_facts(
    model_path: str,
    suite_dir: Path,
    targets: Sequence[str] = BASELINE_TARGETS,
    layer: int = 10,
    site: str = "mlp_out",
    alpha: float = 1.0,
    device_arg: str | None = None,
    dtype_arg: str | None = None,
    max_new_tokens: int = 12,
) -> Path:
    """Probe whether trained payloads help answer a different fact query."""
    suite_dir = Path(suite_dir)
    set_offline_mode(True)
    device, dtype = select_device_and_dtype(device_arg, dtype_arg)
    LOG.info("Loading model from %s on %s (%s)", model_path, device, dtype)
    model, tokenizer = load_local_model_and_tokenizer(model_path, device, dtype)

    multi_token_candidates = [
        candidate for candidate in FACT_PROBE_CANDIDATES
        if len(tokenizer.encode(candidate, add_special_tokens=False)) != 1
    ]
    if multi_token_candidates:
        raise ValueError(
            "Fact-probe candidates require single-token names; multi-token: "
            + ", ".join(multi_token_candidates)
        )

    rows = []
    for target in targets:
        patched_payload_path = (
            suite_dir / target.lower() / "payloads" / f"{target.lower()}_patched_payload.npy"
        )
        legacy_payload_path = suite_dir / target.lower() / "payloads" / f"{target.lower()}_payload.npy"
        payload_path = patched_payload_path if patched_payload_path.exists() else legacy_payload_path
        if not payload_path.exists():
            raise FileNotFoundError(f"Missing payload file: {payload_path}")
        payload = torch.from_numpy(np.load(payload_path))
        null_prompt, _ = _build_fact_probe_prompt(target, include_fact=False)
        token_prompt, fact = _build_fact_probe_prompt(target, include_fact=True)

        with torch.no_grad():
            null_answer = _generate(
                model, tokenizer, null_prompt, device, max_new_tokens=max_new_tokens
            )
            handle = _register_payload_hook(
                model,
                payload.to(device=device, dtype=torch.float32),
                layer,
                site=site,
                alpha=alpha,
                mode="replace",
                burst_steps=1,
            )
            try:
                patched_answer = _generate(
                    model, tokenizer, null_prompt, device, max_new_tokens=max_new_tokens
                )
            finally:
                handle.remove()
            token_answer = _generate(
                model, tokenizer, token_prompt, device, max_new_tokens=max_new_tokens
            )

        for condition, prompt, answer, vector, condition_alpha in (
            ("Null", null_prompt, null_answer, None, 0.0),
            ("Patched", null_prompt, patched_answer, payload.to(dtype=torch.float32, device="cpu"), alpha),
            ("Token", token_prompt, token_answer, None, 0.0),
        ):
            scored, _ = _score_next_token_candidates(
                model,
                tokenizer,
                prompt,
                device,
                FACT_PROBE_CANDIDATES,
                vector=vector,
                alpha=condition_alpha,
                layer=layer,
                site=site,
                inject_mode="replace" if condition == "Patched" else "add",
            )
            bob_row = next(row for row in scored if row["candidate"] == "Bob")
            target_row = next(row for row in scored if row["candidate"] == target)
            rows.append({
                "target": target,
                "fact": fact,
                "condition": condition,
                "answer": answer,
                "parsed_answer": _extract_first_candidate(answer, FACT_PROBE_CANDIDATES),
                "generated_correct": _extract_first_candidate(answer, FACT_PROBE_CANDIDATES) == "Bob",
                "top_candidate": scored[0]["candidate"],
                "top_candidate_correct": scored[0]["candidate"] == "Bob",
                "bob_rank": bob_row["rank"],
                "bob_margin": bob_row["candidate_margin"],
                "target_rank": target_row["rank"],
                "target_margin": target_row["candidate_margin"],
                "payload_path": str(payload_path),
            })

    probe_path = suite_dir / "payload_fact_probe.csv"
    with probe_path.open("w", newline="") as fh:
        fieldnames = [
            "target", "fact", "condition", "answer", "parsed_answer",
            "generated_correct", "top_candidate", "top_candidate_correct",
            "bob_rank", "bob_margin", "target_rank", "target_margin",
            "payload_path",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    patched_rows = [row for row in rows if row["condition"] == "Patched"]
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "model_config_sha256": _sha256_file(Path(model_path) / "config.json"),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "targets": list(targets),
        "layer": layer,
        "site": site,
        "alpha": alpha,
        "inject_mode": "replace",
        "burst_steps": 1,
        "max_new_tokens": max_new_tokens,
        "probe_path": str(probe_path),
        "patched_generated_acc": (
            sum(row["generated_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
        "latent_generated_acc": (
            sum(row["generated_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
        "patched_top_candidate_acc": (
            sum(row["top_candidate_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
        "latent_top_candidate_acc": (
            sum(row["top_candidate_correct"] for row in patched_rows) / len(patched_rows)
            if patched_rows else None
        ),
    }
    (suite_dir / "payload_fact_probe_manifest.json").write_text(json.dumps(manifest, indent=2))

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return probe_path
