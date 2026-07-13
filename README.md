# tensor_language

Training **tensor-transformers** (bilinear / squared-attention models, kept polynomial so the
computation is a tensor network) on language and toy languages, and doing weight-based
mechanistic interpretability on them.

## Active research: `jacclust/`

The current program lives in **[`jacclust/`](jacclust/)** — Jacobian clustering and **operator-SAEs**
for bilinear layers. Start with [`jacclust/SUMMARY.md`](jacclust/SUMMARY.md) (authoritative results)
and [`jacclust/README.md`](jacclust/README.md) (index of every experiment: script → tick → result).

Headline: a bilinear layer's per-datapoint operation has an **exact weights-only Jacobian kernel**;
clustering by it recovers mechanism in toys but reduces to input-cosine on real MLPs; the
productive real-model line is the **bilinear-secant SAE** (a sparse dictionary of per-token
operators `M = y·x⁺`, reconstructed via a Dooms expanded loss so it scales to `d=1152`).

## Model / training code

- `model.py`, `deep_model.py` — tensor-transformer definitions (bilinear attention + bilinear MLP; `DeepModel` runs an `[attn, mlp, ...]` spec).
- `lm_train.py`, `hop_train.py`, `clamp_train.py` — training entry points.
- `runs_owt/`, `runs_lm/`, `runs_gen/`, `runs/` — trained checkpoints (small in-repo models).
- `data_text/` — tokenized corpora (`train.bin`, `val.bin`, `tokenizer.json`).
- Larger checkpoints (124M / 500M bilinear) are on HuggingFace: `Elriggs/gpt2-bilinear-*`.

## Earlier work

`mechdecomp/` — mechanistic-decomposition experiments (paused). `archive/` — older explorations.

## Setup

```
source /venv/main/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python jacclust/<script>.py   # from repo root
```
