# Myelin Latent Bus

Small, reproducible activation-intervention experiments on a kinship fact in
Qwen2.5-Instruct models.

The repo is organized as a boundary map rather than a new method. It asks when
a hidden-state intervention behaves like a portable fact payload, when it
collapses into an answer-token bias, and when standard aligned activation
patching works as a positive control.

## Experiments

### Experiment 1: Contrastive Latent-Bus Demo

The original demo builds a contrastive fact vector from matched contexts and
injects it at answer-start on a small kinship query.

```bash
latent-bus-prepare --out examples/kinship
latent-bus-demo --in examples/kinship --alpha-grid 0.5,1.0
latent-bus-probe --in examples/kinship
```

This is a minimal reference harness for Null / Token / Latent conditions:

- **Null:** query alone.
- **Token:** query plus the missing fact in text.
- **Latent:** query alone plus an injected activation vector.

### Experiment 2: Donor Activation Patch

This experiment replaces optimized payload training with a donor-run activation
patch. For each target in `{Joe, Alice, Dan, Grace, Leo, Max}`, the code runs:

```text
Bob is the father of {target}.
```

It extracts the layer-10 MLP-output activation at the target-name token, then
replaces the answer-start activation in the forward and fact-probe prompts.

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

The legacy alias `latent-bus-train-suite` also works, but this path records
`optimizer: disabled` and does not run Adam. The ignored output directory
contains the cached donor payloads, forward/held-out evaluations, fact-probe
CSV, and manifests.

### Experiment 3: Aligned Patching Positive Control

This is the "tungsten" control. It uses standard clean/corrupted activation
patching on structurally matched Token prompts. Clean and corrupted prompts
differ only in the visible fact target:

```text
Bob is the father of Joe.
Bob is the father of Alice.
```

The positive-control run patches the clean answer-start activation into the
corrupted prompt at the same token position.

```bash
latent-bus-aligned-patch \
  --out examples/aligned_patch_qwen15b_layer27_out \
  --model-path ~/Development/Qwen2.5-1.5B-Instruct \
  --device mps --dtype float16 \
  --layer 27 --site layer_out \
  --patch-position answer \
  --variants train
```

Layer-10 MLP-output aligned patching is also supported, but in current local
runs it does not recover the clean target. The strong positive control is
layer-27 decoder-layer output at answer-start.

## Installation

Requires Python 3.10-3.12.

```bash
git clone https://github.com/lbrichards/myelin-latent-bus.git
cd myelin-latent-bus
pip install -e ".[dev]"
```

Download a local checkpoint. The current follow-up experiments use
Qwen2.5-1.5B-Instruct:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ~/Development/Qwen2.5-1.5B-Instruct \
  --local-dir-use-symlinks False
```

Any local path is fine; set `MYELIN_MODEL_PATH` or pass `--model-path`.

Runtime selection defaults to CUDA, then Apple MPS, then CPU. To pin Apple
Silicon GPU:

```bash
export MYELIN_DEVICE=mps
export MYELIN_DTYPE=float16
```

## What Is Committed

Generated experiment outputs are intentionally ignored. They often include
machine-local paths in manifests and are reproducible from the commands above.
The public repo keeps:

- source code,
- tests,
- static kinship input files,
- experiment documentation,
- packaging metadata.

Draft manuscripts are also ignored.

## Hashes and Provenance

The experiment harness writes SHA-256 hashes into generated manifests at
runtime:

- `prompt_sha256` identifies the exact prompt text used for a trial.
- `payload_sha256` or `vector_sha256` identifies the exact generated activation
  array used by an intervention.
- `model_config_sha256` hashes the local checkpoint's `config.json`.

These hashes are provenance checks, not hard-coded experimental constants. In
particular, `model_config_sha256` is **not** a full model-weight hash and should
not be cited as proof that two machines used byte-identical checkpoint weights.
It is useful for catching obvious model/config mismatches. For publication-grade
reproduction, keep the generated manifest with the local model path, config
hash, prompt hashes, payload hashes, layer/site choices, dtype/device, and
command-line parameters.

Because generated manifests are ignored, the public repo should not contain
stale local SHA values. Re-running an experiment regenerates the hashes for the
checkpoint and artifacts actually used on that machine.

## Development

Run the lightweight offline tests:

```bash
pytest -q
```

The model-dependent experiments require local Hugging Face checkpoints but do
not require network access at runtime.

## Repo Structure

```text
src/latent_bus/
  model_io.py     local model loading, device/dtype selection, activation capture
  injection.py    answer-start add/replace hooks
  prepare.py      contrastive fact-vector extraction
  run.py          experiment harnesses and CSV/manifest writers
  cli.py          console scripts

docs/
  ARCHITECTURE.md
  EXPERIMENTS.md

examples/kinship/
  puzzle.txt
  query.txt
  fact_F.txt

tests/
  lightweight unit tests
```

## Prior Work

The repo uses standard activation-intervention tools; it does not claim aligned
activation patching as a new method. Experiment 3 is deliberately framed as a
positive control using the established clean/corrupted activation-patching
protocol: run a clean prompt and a corrupted prompt, replace an aligned
activation in the corrupted run with the clean activation, and measure recovery
by generated answer, top candidate, and logit difference.

Core prior work and tools:

- Meng, Bau, Andonian, and Belinkov, *Locating and Editing Factual Associations
  in GPT* (ROME), NeurIPS 2022. Introduces causal tracing for factual recall and
  uses clean/corrupted activation restoration to locate causally important
  states.
- TransformerLens activation patching utilities. Provides the standard
  clean/corrupted patching API and terminology used throughout mechanistic
  interpretability practice.
- Wang, Variengien, Conmy, Shlegeris, and Steinhardt, *Interpretability in the
  Wild: a Circuit for Indirect Object Identification in GPT-2 Small*, ICLR 2023.
  Uses activation patching and path patching as causal tools for circuit
  discovery.

Related steering/intervention work:

- Turner et al. and related work on activation addition / steering vectors.
- Todd et al., *Function Vectors in Large Language Models*, ICLR 2024.
- Li et al., *Inference-Time Intervention*, NeurIPS 2023.

The contribution of this repo is not the patching method itself. It is the
bounded comparison between (1) optimized answer-start payloads, (2) donor
activation patches tested for role-agnostic transfer, and (3) an aligned
patching positive control showing that the apparatus can recover clean behavior
when donor and recipient contexts are structurally matched.

## License

MIT. See [LICENSE](LICENSE).
