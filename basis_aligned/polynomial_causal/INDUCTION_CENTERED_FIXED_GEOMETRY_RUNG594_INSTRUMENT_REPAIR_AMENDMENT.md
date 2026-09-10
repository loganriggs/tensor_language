# R594: observed, correctly rounded centered-factor additions

2026-09-10. Prospective successor; R593's invalid terminal and raw evidence remain
unchanged. No R593 diagnostic scientific logits are used for selection or scoring.

The scientific authority remains the hash-bound R585 replacement protocol and
R592 fixed-geometry formulas: same R578 FIT/SELECT rows, all directed counterfactual
families, sites L5H5/L7H3/L8H3/L8H4, physical width30, batches32 (SELECT tail16),
native/replay/score/payload/joint calls, 2000-draw group bootstrap, thresholds,
FIT-first selection gate and nulls. FINAL_TEST/OOD remain closed. No fitted
predictor, row exclusion, changed scientific bar or rank/head/dose sweep.

R593's recorded predicate compared planned delta with fl(fl(before+delta)-before)
at absolute1e-5. At observed scales around1000–1700, a correct FP32 addition can
fail. Its saved hook array was also planned values plus a sentinel, not measured
physical deltas. R594 changes the instrument and evidence representation:

1. Save FP32 before, actual after and FP32 planned layer-total at every query edit,
   ordered layers5/7/8. The two layer8 heads form one physical transaction.
2. An independent CPU NumPy oracle computes round32(float64(before)+float64(total))
   and requires exact FP32 post-state bits. Save actual delta as float64(after)
   minus float64(before); independently reproduce it from the raw arrays.
3. Require layer totals to equal the registered per-head plan (L8 is the FP32 sum
   of its two components). Report representability residual separately; verify
   the correct directional half-ULP rounding interval, including powers of two.
4. Rename the old nominal hook array to planned_component_deltas. The unchanged
   scorer's insertion_activity/per_site_delta_norms explicitly describe registered
   planned components. Actual physical measurements have a three-layer axis and
   are never fabricated or split into fictitious per-head observations.
5. Persist all four new layer arrays in canonical evidence and receipt hashes.
   Skipped edits, wrong signs/totals/sites, and one-ULP post-state corruptions must
   fail the checker. Existing factor, token, zero-replay and support checks remain.

The exact post-state target is an IEEE-FP32 addition observation, not a claim that
the theoretical real-valued intervention is exactly representable. Scientific
endpoint measurements are those actually executed. No tolerance is widened.

The candidate uses focused model-free review, an independently implemented
arithmetic oracle, Fraction-based nearest-value checks and fake-runtime mutation
tests. It does not claim an independent-agent review. This replaces the older
procedural recommendation for a separate agent under the current no-delegation
session; all scientific constraints above remain unchanged. The prior independent
R593 audit remains evidence for the failure being repaired.

Maximum calls remain961 (FIT639, SELECT322), zero backwards/updates. Added raw
evidence is5616directions ×4arms ×3layers ×1152coordinates ×20bytes =1552711680
bytes. Canonical total9804907008bytes; largest current chunk45652480bytes;
combined9850559488bytes. FIT admission11010562560 free bytes, SELECT4473957888.

Workspace has less than1GB free. Use a fresh create-only public root under
/dev/shm/bilin18_induction_r594 for the full raw evidence; this is RAM-backed,
volatile storage, with15GB initially free and26GB available host RAM. Nothing old
is deleted. Compact result/receipt copies and code are committed in workspace.
Do not claim raw-evidence persistence across instance restart; the durable wrapper
records the actual raw location. A later independent raw audit must occur while
those bytes remain available or after an explicitly verified durable copy.

Managed GPU only. The wrapper verifies its complete source binding before loading
the producer, enforces a1800-second process guard, checks free space and fresh
namespaces, then invokes the existing FIT-first producer. Dry run loads no model.
Literal all545902902native weights/producers remain; no independent extraction,
structural saving, FINAL/OOD or complete-goal claim follows from this instrument.
