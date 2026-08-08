# Handoff — the conventional softmax+GELU baseline (the foldability tax)

Written for a reader with no memory of the work. Started 2026-08-08 23:05 UTC.

## What this is

`GRID.md` carried "same-size softmax+GELU transformer — unclaimed" from the day
the programme started, and `STANDALONE_RESULTS.md` §8 opens with it: **every
result in this programme is relative to another foldable model, so nobody has
measured what the exact fold costs in prediction quality.** This closes that.

## Where everything is

| file | what |
|---|---|
| `tf_baseline_predictions.json` | predictions registered BEFORE the first training step — Logan's three and the analyst's nine, which disagree on two of three |
| `tf_baseline_std.py` | the conventional model, a transcription of `tf_train.train_cell`, and controls C1/C2/C3/C5/C6/C7 |
| `tf_baseline_probe.py` | the induction battery called VERBATIM through a shim, plus control C4 |
| `tf_baseline_report.py` | reads whatever is on disk and scores it |
| **`tf_baseline_table.md`** | **the live scored table — read this for the numbers** |
| `tf_baseline_std.json` | the full machine record behind that table |
| `tf_baseline_selfreview.json` | the self-red-team / fairness assessment |
| `tf_baseline_controls.json`, `tf_baseline_probe_control.json` | control results |
| `tf_baseline_chain.sh` / `tf_baseline_chain.log` | the chain and its log |
| `RESULTS.md` FINDING 17 | the write-up |

## The architecture, in one table

| | family (`tf_model.TinyBilin`, variant `vanilla`) | conventional (`tf_baseline_std.StdTransformer`) |
|---|---|---|
| attention | `(q1·k1/16)·(q2·k2/16)`, causal, **no softmax**, two branches | `softmax(q·k/√16)`, causal, **one branch** |
| feed-forward | `Down(Left(x) ⊙ Right(x))` | `Down(GELU(Up(x)))` |
| body per block | `18W² + W` | `(4 + 2·expansion)W² + W` |

Everything else is the same object. Two arms: `x4` (the conventional shape; the
conventional model is then ~12% SMALLER in total parameters) and `x7` (body
exactly `18W² + W`, total bit-identical at every cell).

## Stem convention

`tfb_std{expansion}_d{depth}_w{width}_b8192_s{seed}{suffix}`
e.g. `tfb_std4_d2_w128_b8192_s0`, `tfb_std7_d3_w256_b8192_s2`,
`tfb_std4_d2_w128_b8192_s0_noqknorm`, `tfb_std4_d1_w128_b8192_s0_lr0.01`.
Each cell writes `{stem}.json`, `{stem}.pt`, `{stem}_heldloss.npy` and
`{stem}_induction.json`.

## What the chain runs, in order

0. controls (`tf_baseline_std.py controls`, `tf_baseline_probe.py --control`)
1. seed 0 — nine cells at expansion 4, then nine at expansion 7
2. seed 1 — the same eighteen
3. query/key-norm control, depth 2 width 128, three seeds
4. learning-rate fairness bound, width 128, depths 1–3, at 0.01 and 0.04
5. seed 2 — the same eighteen
6. final report

Nine cells = depths {1,2,3} × widths {64,128,256}, ordered so depth 2 width 128
(the headline) lands first. The report is rebuilt and pushed after every stage,
so a chain killed half-way still leaves a scored table with per-cell seed
counts. **Anything showing fewer than two seeds is PROVISIONAL.**

## To check progress

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
tail -20 tf_baseline_chain.log
ls tfb_std*_b8192_s*.json | grep -v induction | wc -l    # cells done (of ~62)
python tf_baseline_report.py | head -40
```

## To finish the job if the chain died

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
pgrep -f -x "/bin/bash ./tf_baseline_chain.sh"          # already running?
setsid nohup ./tf_baseline_chain.sh >> tf_baseline_chain.stdout 2>&1 </dev/null &
```
Every stage is idempotent on `{stem}.pt` and `{stem}_induction.json`, so
re-launching resumes exactly where it stopped. Do not run it while another
chain holds the card — the script gates itself on
`tf_geom_control_chain.sh` and on ≥10000 MiB free for three consecutive
checks, but a NEW competing chain would not be seen.

## To run one cell by hand

```bash
python tf_baseline_std.py cell --exp 4 --depth 2 --width 128 --seed 0 --suffix ""
python tf_baseline_probe.py --stem tfb_std4_d2_w128_b8192_s0
python tf_baseline_report.py
```

## The three things a reader must not get wrong

1. **Our family is the BIGGER model at nominal expansion** (18W²+W against
   12W²+W of body per block). Logan's registered prediction L3 — "at matched
   parameter count the gap shrinks by at least a third" — presupposes the
   opposite. Matching parameters makes the CONVENTIONAL model bigger.
2. **Two different cross-entropies exist in this programme.** The number used
   here is `run.final_held_ce` (T=512, held rows [0:1500], the training
   protocol) for BOTH families. The rung-5 ladder CE (T=256, 96 sequences) is
   about 0.09 nats lower and must never be mixed in — that mistake has already
   cost this programme a retraction.
3. **The comparison is a LOWER BOUND on the conventional model.** The learning
   rate, the optimiser, the head dimension and the softmax temperature were all
   fixed by our family's history. Only the learning rate is priced. The softmax
   temperature is the largest unpriced risk: query/key RMSNorm caps `|q·k|` at
   the head dimension, so `1/√16` may be too cold, and a null conventional
   induction result must not be over-read before a temperature sweep is run.

## The obvious next cells, if someone wants them

- softmax-temperature sweep at depth 2 width 128 (closes the largest open
  fairness risk; `--exp 4` with the divisor changed from `sqrt(hd)` to
  `hd/2`, `hd/4`, `1`)
- widen the Muon learning-rate grid for the conventional model beyond
  {0.01, 0.02, 0.04} if either edge wins
- a SwiGLU arm — deliberately excluded here because it is a gated bilinear
  form, but it would separate "softmax" from "gating" as the source of any gap
- head dimension 32 or 64 at width 256, where 16 heads is unusually many
