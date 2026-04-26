# Architecture

This note explains how the code is organized and how the experiment harnesses
move from prompts to activation edits to CSV/manifest outputs. See
`docs/EXPERIMENTS.md` for the public-facing experiment list and commands.

## Modules

```
src/latent_bus/
├── model_io.py   -- device/dtype selection, offline-mode flags, local HF model loader,
│                    ActivationCapture context manager (forward hooks on attn/MLP blocks).
├── injection.py  -- answer_start_injection(): registers a forward hook that adds
│                    alpha * v_F or exactly replaces the hidden state at the last
│                    position for the first `burst_steps` forward passes, and records
│                    projection/effect sanity checks.
├── prepare.py    -- prepare_fact_vector(): builds parity-matched contexts (A' with a
│                    token-length-matched placeholder, B with the fact), captures the
│                    hidden state at answer-start under each at a chosen layer/site,
│                    averages (h_B - h_A') across templates and seeds, and unit-normalizes
│                    to produce v_F.
├── run.py        -- experiment harnesses:
│                    run_demo() for Null/Token/Latent contrastive vector demo;
│                    train_payload_suite() for donor activation patch evaluation;
│                    probe_payload_suite_facts() for reversed fact-probe;
│                    run_aligned_patch_control() for clean/corrupted positive control.
└── cli.py        -- thin argparse wrappers exposed as console scripts.
```

## Data flow

```
  fact F
    │
    ▼
  prepare_fact_vector  ──────────────────►  v_F.npy
    │  1. build contexts A' (placeholder-padded) and B (with fact)
    │  2. forward-pass each; ActivationCapture reads hidden state at
    │     answer-start position (last token before the model generates)
    │  3. delta = h_B - h_A' for each (template, seed); average; unit-normalize
    │
    ▼
  run_demo
    │  Null  : model(query_only)                                   ─► incorrect
    │  Token : model(full_puzzle_including_fact)                   ─► "Joe"
    │  Latent: model(query_only) + hook adds alpha*v_F at layer 10 ─► "Joe"
    │                                              ↑
    │                                              injection happens
    │                                              at the first decode
    │                                              step, last position
    ▼
  trials.jsonl, summary.csv, run_manifest.json, report.md
```

## Why answer-start injection

A fact added to the prompt as visible text would appear in the hidden state at the position of those tokens. Removing the tokens removes the activation. Injecting at answer-start (the decoder's last position on the first generation step) recreates the "residue" the fact would have left — specifically, the component of that residue that consistently differs between matched-and-without-fact and matched-and-with-fact contexts. The contrastive extraction isolates that component; the injection places it where the model would normally read it when deciding what to generate next.

## Why parity-matched contexts

If A (without fact) is simply shorter than B (with fact), then `h_B - h_A'` captures both the fact content *and* length effects, tokenizer artifacts, and positional shifts. Padding the fact's span in A' with a neutral placeholder of equal token length isolates the semantic delta from these confounds. Averaging across templates and seeds further cancels prompt-wording-specific noise.

## Why greedy decoding and n=1 per condition

Deterministic single runs make the demo maximally legible and reproducible. The tradeoff is that results are 0-or-1 per configuration with no confidence bands. For statistical claims (as opposed to a demonstration), one would want multi-seed sampling, a query panel beyond the single cousin puzzle, and drift/crosstalk measurements.

## Generated artifacts

The harnesses write CSV, JSON, JSONL, prompt text, and small activation arrays
under `examples/`. These outputs are ignored for public commits because
manifests contain machine-local checkpoint paths. Reproduce them by running the
CLI commands in `docs/EXPERIMENTS.md`.
