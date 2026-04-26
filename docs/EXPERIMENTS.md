# Experiment Guide

This repo contains three related activation-intervention experiments. The
common task is a small kinship problem where the relevant fact has the form:

```text
Bob is the father of {target}.
```

Targets are:

```text
Joe, Alice, Dan, Grace, Leo, Max
```

Candidate sets and prompt wording are defined in `src/latent_bus/run.py`.

## Experiment 1: Contrastive Latent-Bus Demo

**Question.** Can a contrastive vector extracted from matched contexts affect a
single answer-start prediction?

**Conditions.**

- `Null`: no visible fact, no intervention.
- `Token`: fact supplied as text.
- `Latent`: no visible fact, vector injected at answer-start.

**Commands.**

```bash
latent-bus-prepare --out examples/kinship
latent-bus-demo --in examples/kinship --alpha-grid 0.5,1.0
latent-bus-probe --in examples/kinship
```

**Primary artifacts.**

```text
examples/kinship/vectors/v_F.npy
examples/kinship/results/summary.csv
examples/kinship/results/candidate_logits.csv
examples/kinship/results/run_manifest.json
```

These are generated artifacts and are not committed.

## Experiment 2: Donor Activation Patch

**Question.** Does an activation extracted from a donor fact sentence act as a
portable role-agnostic fact payload when transplanted into different queries?

**Donor prompt.**

```text
Bob is the father of {target}.
```

**Extraction.**

- Model: Qwen2.5-1.5B-Instruct.
- Site: layer 10, MLP output.
- Position: target-name token, not trailing punctuation.
- Capture detail: clone before CPU copy to avoid MPS async view reuse.

**Injection.**

- Exact replacement at answer-start.
- No optimizer; Adam and CE training are disabled.
- `burst_steps=1`.

**Commands.**

```bash
latent-bus-patch-suite \
  --out examples/patched_suite_qwen15b \
  --model-path ~/Development/Qwen2.5-1.5B-Instruct \
  --device mps --dtype float16 \
  --steps 80 \
  --eval-variants train,paraphrase,minimal

latent-bus-fact-probe \
  --in examples/patched_suite_qwen15b \
  --model-path ~/Development/Qwen2.5-1.5B-Instruct \
  --device mps --dtype float16
```

**Primary artifacts.**

```text
examples/patched_suite_qwen15b/payload_suite_summary.csv
examples/patched_suite_qwen15b/payload_fact_probe.csv
examples/patched_suite_qwen15b/patched_payload_cache_manifest.json
```

**Claim boundary.** This experiment can show whether the tested single-site,
single-token donor activation transfers across prompt roles. It does not rule
out multi-token, multi-layer, residual-stream, attention-path, or KV-cache
interventions.

## Experiment 3: Aligned Patching Positive Control

**Question.** Is the patching apparatus capable of causally recovering a clean
answer when clean and corrupted contexts are structurally aligned?

This is not a new method. It follows the standard clean/corrupted activation
patching setup used in causal tracing and mechanistic interpretability work:
cache activations from a clean run, run a corrupted prompt, replace an aligned
activation with its clean counterpart, and measure recovery.

**Clean/corrupted prompts.** Full Token-style kinship prompts that differ only
in the target of the visible father fact.

```text
Clean:    Bob is the father of Joe.
Corrupt:  Bob is the father of Alice.
```

**Primary positive control.**

- Site: decoder layer output.
- Layer: 27.
- Position: answer-start.
- Patch mode: exact replacement.

**Command.**

```bash
latent-bus-aligned-patch \
  --out examples/aligned_patch_qwen15b_layer27_out \
  --model-path ~/Development/Qwen2.5-1.5B-Instruct \
  --device mps --dtype float16 \
  --layer 27 --site layer_out \
  --patch-position answer \
  --variants train
```

**Primary artifacts.**

```text
examples/aligned_patch_qwen15b_layer27_out/aligned_patch_summary.csv
examples/aligned_patch_qwen15b_layer27_out/aligned_patch_control.csv
examples/aligned_patch_qwen15b_layer27_out/aligned_patch_manifest.json
```

**Expected local result.**

```text
Clean:        30/30 generated clean answer
Corrupt:      30/30 generated corrupt answer
AlignedPatch: 30/30 generated clean answer
```

Layer-10 MLP-output and layer-10 decoder-output aligned answer-start patches are
also runnable, but current local results do not recover the clean answer. This
is useful for the boundary map: late aligned patching works, while the earlier
single-site payload-style interventions fail to provide portable facts.

**Attribution.** This control is borrowing established practice, especially the
clean/corrupted activation restoration protocol from ROME-style causal tracing
and the activation-patching terminology/API popularized by TransformerLens. It
is included to show that the hook and scoring apparatus can produce a causal
answer shift under aligned conditions, not to claim novelty for aligned
patching.

## Public Repo Hygiene

Generated result folders are ignored by `.gitignore` because manifests include
local checkpoint paths. To reproduce a result, rerun the command for that
experiment and inspect the generated CSV/JSON artifacts locally.

## SHA-256 Usage

The manifests record SHA-256 hashes for prompts, payload arrays, contrastive
vectors, and `config.json`. These hashes make a local run auditable: they answer
"which exact prompt string and generated array did this CSV use?"

They do not replace a full checkpoint provenance record. The model config hash
can catch obvious model-family/config drift, but it is not a hash of every model
weight shard. If a result is used in a paper or review artifact, preserve the
full generated manifest together with the model source, revision if known,
device/dtype, layer/site, and command-line parameters.
