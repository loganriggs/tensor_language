# Preregistration — how many score templates does the copy TASK need? (CPU probe; parallel lane)

Date: 2026-09-03 01:08 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any per-half rank outcome; zero model forwards, CPU only,
on rung513's published per-implementation task-effect statistics. Grounds the
coverage-credit dictionary claim in TASK space, independent of the write-space
§2647 arc. Codex's nonlinear-reader route untouched.

## Transparency

An exploratory POOLED (all 248 discovery docs) computation of the raw top-1
energy fraction motivated this probe. The SCORED test below is the UNPEEKED
per-document-half reproduction and cross-half direction stability; the pooled
value is NOT a registered bar.

## Object and computation (exact, deterministic)

Rung513's bundle (sha 06118d18594c4b167a3f3d46a2aa282969f6b061835f83a3b3d62b5ca72b8d8a) stores source_task[a,doc,cell] (4 implementations
N/P/Z7/Z8 × 248 docs × 6 context cells: all/near/far/one/multiple/off) and
base_task[doc,cell] (score-absent). The per-implementation copy-TASK effect is
E[a,doc,c] = source_task[a,doc,c] - base_task[doc,c] on the 4 context cells
c in {near,far,one,multiple} (indices 1..4). Rung513's discovery split is docs
500:748 at 624 -> half0 = local docs [0,124), half1 = [124,248).

Per half h, form the 4×(124·4) matrix V_h (rows = implementations, columns =
doc×cell effects). Report, per half: the raw singular values, the top-1 energy
fraction e1_h = S[0]^2 / sum(S^2), and the effective rank
exp(-sum p_i log p_i) with p_i = S_i^2/sum(S^2). Also report the top-right-
singular-vector cosine between half0 and half1 (the shared score direction's
stability). Separately report the CENTERED (mean-removed across the 4
implementations) spectrum as a descriptive companion (the mismatch spread).

## Frozen predictions

### pred_a — exact reproduction
Bundle sha matches; shapes exact (source_task 4×248×6, base_task 248×6); every
scored matrix finite; the pooled top-1 energy recomputes deterministically.

### pred_b — the copy task is near-rank-1 across implementations, both halves
In BOTH document halves independently: raw top-1 energy fraction e1_h >= .90.
(The four implementations produce nearly-collinear copy-task effects: one
dominant shared score direction.)

### pred_c — the shared score direction is stable across halves
The top-right-singular-vector cosine between half0 and half1 is >= .90 (the
one dominant direction is the same on both document halves, not a per-half
artifact).

## Strong null
Fires if pred_a fails, or pred_b fails in either half (task-space effect is NOT
low-rank -> the copy task needs multiple score templates and the §2633
dictionary reduction has no task-space basis), or pred_c fails (the low rank is
a per-half sampling artifact). Reported beside §2633/§2647 either way; no bar
changes.

## Price
Zero model forwards; CPU < 10 s; one receipt JSON. Nothing deployed.
