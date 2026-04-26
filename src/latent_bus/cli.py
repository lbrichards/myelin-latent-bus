"""Console-script entry points.

After `pip install -e .` (or `pip install .`) these are runnable as
`latent-bus-prepare ...` and `latent-bus-demo ...` from the shell.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from latent_bus.prepare import DEFAULT_QUERY, prepare_fact_vector
from latent_bus.run import (
    BASELINE_ANSWER_CANDIDATES,
    BASELINE_TARGETS,
    cache_patched_payloads,
    DEFAULT_CANDIDATES,
    FACT_PROBE_CANDIDATES,
    probe_payload_suite_facts,
    run_aligned_patch_control,
    run_demo,
    run_kinship_baseline,
    run_logit_probe,
    train_payload_suite,
    train_single_payload,
)


def _default_model_path() -> str:
    """Resolve the default model location from env or a sensible fallback."""
    env = os.environ.get("MYELIN_MODEL_PATH")
    if env:
        return env
    return str(Path.home() / "Development" / "Qwen2.5-1.5B-Instruct")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Runtime device. Defaults to $MYELIN_DEVICE or auto-detection.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        default=None,
        help="Runtime dtype. Defaults to $MYELIN_DTYPE or a device-specific choice.",
    )


def prepare_main(argv=None) -> int:
    """CLI: build the fact vector v_F for a given fact."""
    p = argparse.ArgumentParser(
        prog="latent-bus-prepare",
        description="Build a fact vector v_F from parity-matched contexts.",
    )
    p.add_argument("--out", required=True, help="Output directory (e.g., examples/kinship)")
    p.add_argument("--fact", default="Bob is the father of Joe.",
                   help="The missing fact F. Default matches the bundled kinship example.")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--model-path", default=_default_model_path(),
                   help="Local HF checkpoint directory. "
                        "Defaults to $MYELIN_MODEL_PATH or ~/Development/Qwen2.5-1.5B-Instruct.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    path = prepare_fact_vector(
        model_path=args.model_path,
        fact=args.fact,
        out_dir=Path(args.out),
        layer=args.layer,
        site=args.site,
        query=args.query,
        device_arg=args.device,
        dtype_arg=args.dtype,
    )
    print(f"Wrote fact vector to {path}")
    return 0


def demo_main(argv=None) -> int:
    """CLI: run Null / Token / Latent conditions using a prepared fact vector."""
    p = argparse.ArgumentParser(
        prog="latent-bus-demo",
        description="Run the three-condition latent-bus demo.",
    )
    p.add_argument("--in", dest="in_dir", required=True,
                   help="Directory produced by latent-bus-prepare.")
    p.add_argument("--alpha-grid", default="0.5,1.0",
                   help="Comma-separated alpha values to try for the Latent condition.")
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--fallback-layer", type=int, default=12,
                   help="Layer to retry if the primary layer yields no correct alpha. "
                        "Pass -1 to disable.")
    p.add_argument("--burst-steps", type=int, default=1)
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    fb = None if args.fallback_layer < 0 else args.fallback_layer
    csv_path = run_demo(
        model_path=args.model_path,
        in_dir=Path(args.in_dir),
        alpha_grid=alphas,
        site=args.site,
        layer=args.layer,
        fallback_layer=fb,
        burst_steps=args.burst_steps,
        device_arg=args.device,
        dtype_arg=args.dtype,
    )
    print(f"Summary written to {csv_path}")
    return 0


def probe_main(argv=None) -> int:
    """CLI: score candidate next-token logits for Null / Token / Latent."""
    p = argparse.ArgumentParser(
        prog="latent-bus-probe",
        description="Score candidate next-token logits for the prepared demo.",
    )
    p.add_argument("--in", dest="in_dir", required=True,
                   help="Directory produced by latent-bus-prepare.")
    p.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES),
                   help="Comma-separated candidate answer strings to score.")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--burst-steps", type=int, default=1)
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    csv_path = run_logit_probe(
        model_path=args.model_path,
        in_dir=Path(args.in_dir),
        candidates=candidates,
        alpha=args.alpha,
        site=args.site,
        layer=args.layer,
        burst_steps=args.burst_steps,
        device_arg=args.device,
        dtype_arg=args.dtype,
    )
    print(f"Logit probe written to {csv_path}")
    return 0


def baseline_main(argv=None) -> int:
    """CLI: evaluate Null/Token baselines over a tiny kinship set."""
    p = argparse.ArgumentParser(
        prog="latent-bus-baseline",
        description="Evaluate Null/Token candidate logits on a kinship baseline set.",
    )
    p.add_argument("--out", required=True, help="Output directory for baseline artifacts.")
    p.add_argument("--targets", default=",".join(BASELINE_TARGETS),
                   help="Comma-separated target answer names.")
    p.add_argument("--candidates", default=",".join(BASELINE_ANSWER_CANDIDATES),
                   help="Comma-separated candidate answer names.")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    summary_path = run_kinship_baseline(
        model_path=args.model_path,
        out_dir=Path(args.out),
        targets=targets,
        candidates=candidates,
        device_arg=args.device,
        dtype_arg=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Baseline summary written to {summary_path}")
    return 0


def cache_payloads_main(argv=None) -> int:
    """CLI: cache donor-run patched payloads for the kinship targets."""
    p = argparse.ArgumentParser(
        prog="latent-bus-cache-payloads",
        description="Extract donor activation payloads for kinship targets.",
    )
    p.add_argument("--out", required=True, help="Output suite directory.")
    p.add_argument("--targets", default=",".join(BASELINE_TARGETS),
                   help="Comma-separated target answer names.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out"])
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    manifest_path = cache_patched_payloads(
        model_path=args.model_path,
        out_dir=Path(args.out),
        targets=targets,
        layer=args.layer,
        site=args.site,
        device_arg=args.device,
        dtype_arg=args.dtype,
    )
    print(f"Patched payload cache manifest written to {manifest_path}")
    return 0


def aligned_patch_main(argv=None) -> int:
    """CLI: run the aligned clean/corrupted activation-patching positive control."""
    p = argparse.ArgumentParser(
        prog="latent-bus-aligned-patch",
        description="Run a standard aligned activation-patching positive control.",
    )
    p.add_argument("--out", required=True, help="Output directory for aligned patch artifacts.")
    p.add_argument("--targets", default=",".join(BASELINE_TARGETS),
                   help="Comma-separated target answer names.")
    p.add_argument("--candidates", default=",".join(BASELINE_ANSWER_CANDIDATES),
                   help="Comma-separated candidate answer names.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out", "layer_out"])
    p.add_argument("--patch-position", default="answer", choices=["answer", "fact_target"],
                   help="Aligned position to patch. Defaults to answer-start.")
    p.add_argument("--variants", default="train",
                   help="Comma-separated prompt variants to evaluate.")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    result_path = run_aligned_patch_control(
        model_path=args.model_path,
        out_dir=Path(args.out),
        targets=targets,
        candidates=candidates,
        layer=args.layer,
        site=args.site,
        variants=variants,
        patch_position=args.patch_position,
        device_arg=args.device,
        dtype_arg=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Aligned patch control written to {result_path}")
    return 0


def train_payload_main(argv=None) -> int:
    """CLI: extract and evaluate one patched payload for a target answer."""
    p = argparse.ArgumentParser(
        prog="latent-bus-train-payload",
        description="Extract and evaluate one donor activation patch for a kinship target.",
    )
    p.add_argument("--out", required=True, help="Output directory for payload artifacts.")
    p.add_argument("--target", default="Joe")
    p.add_argument("--candidates", default=",".join(BASELINE_ANSWER_CANDIDATES),
                   help="Comma-separated candidate answer names.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--norm-lambda", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--log-every", type=int, default=10,
                   help="Log optimization progress every N steps; pass 0 to disable.")
    p.add_argument("--eval-variants", default="train,paraphrase,minimal",
                   help="Comma-separated prompt variants to evaluate after training.")
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    eval_variants = [x.strip() for x in args.eval_variants.split(",") if x.strip()]
    result_path = train_single_payload(
        model_path=args.model_path,
        out_dir=Path(args.out),
        target=args.target,
        candidates=candidates,
        layer=args.layer,
        site=args.site,
        steps=args.steps,
        lr=args.lr,
        norm_lambda=args.norm_lambda,
        alpha=args.alpha,
        device_arg=args.device,
        dtype_arg=args.dtype,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        eval_variants=eval_variants,
    )
    print(f"Payload result written to {result_path}")
    return 0


def train_suite_main(argv=None) -> int:
    """CLI: extract and evaluate one patched payload per target answer."""
    p = argparse.ArgumentParser(
        prog="latent-bus-patch-suite",
        description="Extract and evaluate one donor activation patch per kinship target.",
    )
    p.add_argument("--out", required=True, help="Output directory for suite artifacts.")
    p.add_argument("--targets", default=",".join(BASELINE_TARGETS),
                   help="Comma-separated target answer names.")
    p.add_argument("--candidates", default=",".join(BASELINE_ANSWER_CANDIDATES),
                   help="Comma-separated candidate answer names.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--norm-lambda", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--log-every", type=int, default=10,
                   help="Legacy compatibility option; ignored by patch extraction.")
    p.add_argument("--eval-variants", default="train,paraphrase,minimal",
                   help="Comma-separated prompt variants to evaluate after extraction.")
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    eval_variants = [x.strip() for x in args.eval_variants.split(",") if x.strip()]
    summary_path = train_payload_suite(
        model_path=args.model_path,
        out_dir=Path(args.out),
        targets=targets,
        candidates=candidates,
        layer=args.layer,
        site=args.site,
        steps=args.steps,
        lr=args.lr,
        norm_lambda=args.norm_lambda,
        alpha=args.alpha,
        device_arg=args.device,
        dtype_arg=args.dtype,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        eval_variants=eval_variants,
    )
    print(f"Payload suite summary written to {summary_path}")
    return 0


def fact_probe_main(argv=None) -> int:
    """CLI: probe patched payloads on a different fact question."""
    p = argparse.ArgumentParser(
        prog="latent-bus-fact-probe",
        description="Probe trained payloads on a different fact query whose answer is Bob.",
    )
    p.add_argument("--in", dest="suite_dir", required=True,
                   help="Suite directory produced by latent-bus-train-suite.")
    p.add_argument("--targets", default=",".join(BASELINE_TARGETS),
                   help="Comma-separated target answer names.")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--site", default="mlp_out", choices=["mlp_out", "attn_out"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--model-path", default=_default_model_path())
    _add_runtime_args(p)
    args = p.parse_args(argv)

    _setup_logging()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    probe_path = probe_payload_suite_facts(
        model_path=args.model_path,
        suite_dir=Path(args.suite_dir),
        targets=targets,
        layer=args.layer,
        site=args.site,
        alpha=args.alpha,
        device_arg=args.device,
        dtype_arg=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Payload fact probe written to {probe_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Allow `python -m latent_bus.cli prepare ...` style if someone wants it.
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        sys.exit(prepare_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        sys.exit(demo_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        sys.exit(probe_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        sys.exit(baseline_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "cache-payloads":
        sys.exit(cache_payloads_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "aligned-patch":
        sys.exit(aligned_patch_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "train-payload":
        sys.exit(train_payload_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] in {"train-suite", "patch-suite"}:
        sys.exit(train_suite_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "fact-probe":
        sys.exit(fact_probe_main(sys.argv[2:]))
    print("Usage: python -m latent_bus.cli {prepare|demo|probe|baseline|cache-payloads|aligned-patch|train-payload|train-suite|patch-suite|fact-probe} [args]", file=sys.stderr)
    sys.exit(2)
