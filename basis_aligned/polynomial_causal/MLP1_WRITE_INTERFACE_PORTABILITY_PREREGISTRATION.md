# Preregistration — MLP1-write interface portability probe (parallel lane)

Date: 2026-09-02 17:02 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any transplant outcome exists

## Question

Rung493 established that replacing the MLP1 write in place removes .84–.95 of ALL
branch-pair effect distinctions (same-position aligned removed fractions; the
merge intervention removes CE-effect contrast — larger removed fraction = more of
the distinction carried by that site). That made MLP1's write a generic chokepoint
— but a chokepoint is only an INTERFACE if what passes through it is portable.
This probe runs the Beckers–Halpern-flavored sufficiency test my 1607 review
offered (move #2): transplant a DONOR document's branch write-adjustment into a
RECIPIENT document at the same site and token positions, and ask whether it
recovers the recipient branch's effect the way the recipient's own adjustment
does. Prior honestly stated: the adjustment is plausibly context-bound
(token-position-specific), and the null is the expected outcome; either verdict is
a boundary datum — pass makes the chokepoint a portable interface candidate, null
closes interface language at this grain and guards future over-claims.

Main-line note: Codex owns the attention1 arc (495 and its BF16 re-run); this
probe touches only MLP1, imports the hash-pinned rung493 module as a library, and
modifies no registered file.

## Exact arms (all named)

Frozen documents and machinery inherited from rung493 (whose validate chain binds
rung474→492 lineage). Batches of 4 documents; donor pairing FIXED as index
XOR 1 within each batch ((0,1),(2,3)); branches b ∈ {T, I} (the material pair).
Per batch:

- NATIVE: the normal model (per-token CE baseline);
- ABSENT_b: rung493's exact branch-absent trajectory for b ∈ {T, I};
- adj_b = M_N − M_b in float32 (native minus branch-absent MLP1 write);
- OWN_b: rung493's `_merge_forward` in `M_ONLY` mode with edited write = M_N
  (restore the recipient's own native write; the in-run restoration reference);
- DONOR_b: edited write = M_b + adj_b[donor] (the paired document's own-branch
  adjustment, cast to deployed dtype after float32 addition);
- CROSSED_b: edited write = M_b + adj_{b'}[donor] with b' the other branch of
  {T, I} (branch-specificity control);
- SHIFT_b: 16 edited writes = M_b + roll(adj_b, shift) over rung493's frozen
  16 position offsets (position control).

Layers 2–17 always recompute. Per token, x_b = CE(ABSENT_b) − CE(NATIVE) is the
branch effect; for an arm e, recovery_e = CE(ABSENT_b) − CE(e). Report per branch
and document half (0:250 / 250:500): aligned recovered fraction
<recovery_e, x_b>/||x_b||², cosine(recovery_e, x_b), and RMS ratio. LOWER
residual/HIGHER aligned recovery = more restoration; stated per claim.

## Data scope

Documents 0:500 only (rung493's discovery range), reported independently on the
two halves. Single-phase probe: no validation phase, no sealed or final role
opened. These are new intervention outcomes on a previously used corpus — not
new-corpus OOD evidence and not prospective validation; a pass licenses (not
constitutes) a registered validation successor.

## Frozen predictions

### pred_a — exact, lawful, live transplant instrument
All frozen hashes match (rung493 source/result, this preregistration; rung493's
own validator re-runs its 474→492 chain). Rung493 receipt shows A true, B/C/D/E
false, strong null, validation closed. Rung493's identity suite holds at its
registered bounds (native prefix D/A/M and z ≤ 1e-12 relative squared,
mlp0-state max-abs 0.0, S-prefix replay and state-source ≤ 1e-12, edited-write
max-abs error 0.0, analytical branch identity ≤ 1e-8, deployed ≤ 1e-5). All call
counts exact: 125 native + 250 absent + 4,750 merge forwards (38 per batch).
Every adjustment RMS ≥ 1e-4; donor-minus-own adjustment difference RMS ≥ 1e-4 in
every batch (the transplant is a real change); OWN aligned recovered fraction
> 0 for both branches in both halves (live restoration reference).

### pred_b — cross-document write portability
For both branches T and I, in both halves:
- DONOR aligned recovered fraction ≥ .25 × OWN aligned recovered fraction
  (rung492's relative-bar precedent);
- DONOR aligned recovered fraction ≥ the 95th percentile of the 16 SHIFT
  fractions + .05; and
- DONOR recovery cosine with x ≥ .30.

### pred_c — the transplant is branch-specific
For both branches, in both halves: DONOR (matched-branch) aligned recovered
fraction ≥ CROSSED (other-branch) fraction + .05.

Descriptive regardless of verdict: full OWN/DONOR/CROSSED/SHIFT tables per
branch/half, OWN fraction (how much of the effect the write carries in place,
connecting to rung493's .84–.95), donor-vs-own adjustment cosine.

## Strong null and interpretation

Strong null fires if any of pred_a–pred_c fails. On the null: the MLP1-write
adjustment is context-bound — the chokepoint is a SITE, not a portable interface;
rung493's in-place removal result stands unchanged; interface-contract language is
closed at this grain and any future interface claim needs a finer-grain carrier.
Bars will not be relaxed; failed controls will not be dropped. A full pass
identifies a portable-interface CANDIDATE only, and licenses a registered
validation successor (500:1000) plus unrelated-circuit preservation tests before
any adoption. No compression, rank, or quantization claim in any branch of the
outcome.

## Literal price

125 batches × (1 native + 2 absent + 6 same-position arms + 32 shifts) = 5,125
full-model forwards, single phase. Per-token CE stored for native/absent/arms
(float32 bundle); shift arms contracted to fractions. Zero deployed parameters
added or removed.
