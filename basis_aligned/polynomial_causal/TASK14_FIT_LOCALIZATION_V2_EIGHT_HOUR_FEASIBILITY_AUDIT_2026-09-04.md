# Task 14 FIT localization v2: preliminary eight-hour feasibility audit

**CPU-only audit:** 2026-09-04 11:20 UTC. **Status:** informative comparison, not an execution license.
The repaired physical compiler, model-facing producer, largest-shape memory canary, and stage-specific p99 timing
receipt remain required before any task-14 queue entry.

## Question

Can the compiler's conservative branch-complete task-14 schedule plausibly finish inside its frozen eight-hour
allowance without shortening the scientific design?

The current compiler candidate reports the following worst-case physical price:

| quantity | task 14 maximum |
|---|---:|
| logical Adam updates | 60,000 |
| physical forward batches | 119,207 |
| autograd backward calls | 60,004 |
| forward graphs contributing to those backwards | 118,004 |
| prompt sequences evaluated | 9,207,984 |
| tokens evaluated | 63,782,508 |

These counts are still conditional on the repaired compiler reproducing them exactly. The first compiler was blocked
for validation and branch-state defects, so this memo does not elevate its artifacts to authority.

## Closest completed timing reference

Rung 522 was a managed projector-fitting run on the same model and GPU environment. Its immutable terminal receipt
reports:

| quantity | rung 522 measured |
|---|---:|
| optimization forwards | 20,600 |
| optimization backwards | 20,600 |
| inference-only forwards actually reached | 5,029 |
| wall time | 4,981.154 s = 83.019 min |

Rung 522 is only a rough comparator. Its optimization batches contained four or six sequences of length 256, or
1,024--1,536 input tokens per optimization forward. Task 14 uses short prompts: its compiled worst case averages

$$
\frac{63{,}782{,}508}{119{,}207}=535.06
$$

tokens per physical forward, with an average prompt length of 6.93 tokens. On the other hand, task 14 accumulates
1.967 forward graphs per backward on average and must execute more Python-level calls, hooks, hashes, and evidence
writes. These differences prevent a defensible one-number extrapolation.

## Three deliberately simple extrapolations

Scaling rung 522 only by optimizer updates gives

$$
4{,}981.154\frac{60{,}000}{20{,}600}=14{,}508\ \text{s}=4.03\ \text{h}.
$$

Scaling only by physical forward calls gives

$$
4{,}981.154\frac{119{,}207}{20{,}600+5{,}029}=23{,}169\ \text{s}=6.44\ \text{h}.
$$

Treating every forward and backward as one event gives

$$
4{,}981.154\frac{119{,}207+60{,}004}{20{,}600+5{,}029+20{,}600}
=19{,}310\ \text{s}=5.36\ \text{h}.
$$

Thus the crude comparators span 4.03--6.44 hours, leaving 1.56--3.97 hours inside the eight-hour limit. This is
encouraging but not a runtime proof. The forward-count estimate is probably conservative about tensor work because
task 14's average token count per call is lower; it may be optimistic about launch, graph-retention, and evidence
overhead because task 14 has many more small calls and two live graphs per logical update.

Equivalently, the eight-hour limit permits at most 0.2416 seconds per compiled forward batch if all elapsed time is
charged against forwards, or 0.4800 seconds per logical optimizer update if all elapsed time is charged against
updates. Rung 522 averaged 0.1944 seconds per reached forward and 0.2418 seconds per optimizer update, but those
averages mix different batch shapes and cannot be substituted for a p99 bound.

## Decision

The eight-hour design remains **plausibly feasible**, so there is no evidence-based reason to weaken its frozen
counterfactual families, five seeds, ranks, sites, necessity tests, or reader tests. It is not yet authorized.

The producer must measure the exact implementation at every distinct registered physical shape, including the
largest `batch=192, length=8, rank=4` two-graph backward canary. The authorization receipt must compute

$$
T_{\mathrm{upper}}
=T_{\mathrm{startup}}+T_{\mathrm{preflight}}+T_{\mathrm{publication}}
+\sum_s N_s\,t_{s,\mathrm{p99}}
\le 28{,}800\ \text{s},
$$

where $N_s$ is the repaired compiler's exact count for physical shape/stage $s$ and $t_{s,\mathrm{p99}}$ is a reviewed
p99 measurement from the exact producer, model, checkpoint, runtime, and GPU. A single mean time is insufficient.
If that inequality fails, execution must hard-abort before task data or model outcomes are opened. The remedy would
be a separately preregistered implementation optimization or scientific redesign, never dropping calls during the
run.
