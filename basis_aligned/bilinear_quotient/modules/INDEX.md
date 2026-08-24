# Module dossiers — read the relevant one BEFORE designing an experiment

One self-contained doc per module of bilin18. Each holds: wiring, established facts (with §refs
into BILIN18_CONNECTION.md and the key numbers inline), how understood on the benchmark
(0=mean-ablate, 1=full), the experiments/files that established it, module-specific method traps,
and what's genuinely open. **Purpose: kill re-derivation** (§1092: four experiments re-derived the
sink arc because the facts were scattered). If you learn something new about a module, UPDATE ITS
DOSSIER in the same commit as the ledger writeup.

| Doc | Covers |
|---|---|
| `attn-sink-5-7.md` | Head 5.7 — the constant/bias head, position-0 mechanism, gain payload |
| `attn-front-routers.md` | L0 bigram-table attention, L0H3/L1H1/L2H5 routers, windows |
| `attn-middle-pooling.md` | The redundant collective pooling band, pooling criterion, static-attention |
| `mlp-front-grammar.md` | mlp0/mlp1 — class writers, gain writer, bilinear conjunction gate |
| `mlp-transition-L3-5.md` | Content onset; L4 the first context MLP; mlp4's position-0 job |
| `mlp-deep-content.md` | The content machine L5-14 — manifold, universality, patching, OOD |
| `readout-L15-17.md` | Near-linear read, mlp16 structure, block-17 calibration, the merge |
| `induction.md` | The induction mechanism — heads, circuit, coupling with content |
| `specialist-heads.md` | Named specialists (10.5/8.1/17.2-3/matchers), three-goal status, extraction ladder, selectivity matrix |
| `channels.md` | value-residual, x0 re-injection, massive dims/gain, embedding dominance, clamps |
| `benchmark.md` | Per-module understanding scores + valid stand-ins + measurement rules |

Model loading + forward idioms: see any recent script (e.g. `l4_variable.py`) — `from
bilin18_joint_removal import m, DEV` (sys.path `/workspace/rspd`), census via
`cl.use_state(PT+'census_state_diverse.pt')`, data via `cl.fineweb_rows(N)` (FineWeb ONLY, LESSONS
rule 10). Model: 18L, D=1152, 9 heads×128, HID=4608, V≈50257; block: `x = λ₀x+λ₁x₀;
x,v1 = attn(rms_norm(x),v1); x+=attn_out; x+=mlp(rms_norm(x))`; logits `30·tanh(lm_head(rms(x))/30)`.

Dedup rule (§1092): before a new thread, grep the ledger with MULTIPLE vocabularies — old arcs
predate the § era and use different terms (sink, cost map, dotted `5.7` head notation).
