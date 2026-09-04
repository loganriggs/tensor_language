# Circuit-only ten-minute operating plan

**Effective:** 2026-09-04 13:51 UTC. **Scope:** Codex and Claude research lanes until the user changes it.

## Goal and unit of throughput

All new research work targets circuits: what information is read, what computation is performed, what is written,
which downstream computation uses it, and whether the proposed unit predicts, transfers, composes, and can be
selectively manipulated. Compression, rank reduction, quantization, and frontier pricing are not independent work
lanes during this directive.

The serial throughput target is one **new causal circuit screen or honest null receipt every ten minutes**. A screen is
not a completed circuit. It promotes a candidate only when the model performs the behavior, a declared intervention
has a task-specific effect, and matched controls rule out the simplest shortcut. Promoted candidates receive deeper
held-out interchange, necessity, sufficiency, interaction, out-of-distribution, and weight-translation work in
parallel. This preserves the evidential standard without making every cheap screen wait for adoption-grade machinery.

## Ten-minute serial loop

| elapsed | operation | automatic output |
|---:|---|---|
| 0:00–1:00 | Query behavior, module, and method dossiers using canonical IDs and aliases. | prior-result/novelty receipt; duplicate work stops here |
| 1:00–3:00 | Fill a small declarative hypothesis: behavior, answer score, information read, proposed operation/write, candidate sites, alternative explanation, and opposing predictions. | `CircuitExperimentSpec` fragment |
| 3:00–5:00 | Instantiate linked answer-changing, state-preserving, unrelated, and shifted examples from reusable transformation templates. | hashed FIT/held-out rows plus capability checks |
| 5:00–7:00 | Run the shared native + whole-module/head/full-state causal screen through the managed runner. | one batched result package |
| 7:00–9:00 | Apply common task-effect, control, saturation, and stability scores. | `screen`, `null`, or `invalid` verdict with reason |
| 9:00–10:00 | Append the result to its dossier and launch the next nonduplicate candidate; promote only if the fixed bars pass. | indexed dossier event and next claim |

The runner should batch several arms in one model load, but the measurement above is serial latency for one candidate,
not total parallel throughput divided by the number of agents or arms.

## Risk-tiered engineering

The failed task-14 path treated discovery screening like final deployment. That produced several hours of compiler
work before the first localization result. The replacement has three explicitly different levels:

1. **Screen:** reversible FIT-only diagnostic, common executor, ordinary deterministic tests, no adoption claim. Target:
   ten serial minutes.
2. **Identification:** only for a screen that passes; frozen discovery/held-out separation, linked counterfactuals,
   interchange plus selective removal, interaction controls, and stability checks. This may take longer and runs in
   parallel with new screens.
3. **Adoption:** only for an identified circuit being installed into the smaller executable model; exact artifact
   closure, pricing, composition, OOD, and publication hardening. This is where deployment-grade compiler review belongs.

A screen does not get a bespoke compiler, hundreds of thousands of pre-enumerated calls, or exhaustive publication
machinery. A promoted circuit does not inherit a screen's evidential status merely because the code is reusable.

## Shared system to build now

Reuse the existing `ops/circuit_experiment_spec.py`, shared artifact publisher, managed entrypoint, behavior bank,
model facade, and circuit registry. Add only the missing vertical interfaces:

- a compact circuit-candidate specification containing the computation and opposing hypotheses;
- a machine-checked prior-result/novelty receipt;
- reusable counterfactual transformation templates;
- a generic batched module/head/full-state intervention screen;
- a common scorer that distinguishes invalid instruments, causal nulls, and promotable screens; and
- an automatic module/behavior dossier append plus serial timing receipt.

The acceptance test is a previously unused behavior going from idea to an indexed, independently reproducible screen
or null in at most ten serial minutes without adding a task-specific compiler, adapter, publisher, or scoring program.

## Hourly systems review

At the first safe boundary each hour, both Codex and Claude inspect repository timestamps for every candidate touched in
the preceding hour. Record time spent in scientific thinking, repeated code, tests, GPU computation, review handoff,
and waiting. If median serial latency exceeds ten minutes, identify the largest repeated step and remove it in one
bounded engineering change before starting another bespoke experiment. The review also checks that every active item
is a circuit question, has passed the prior-result gate, and has either advanced or produced an honest null.

## Immediate disposition

- Stop extending the bespoke task-14 localization compiler. Preserve its defects and reviews as evidence for the
  systems redesign; do not freeze or enqueue it.
- Let the currently running Claude job finish, record its already-incurred result, then open no new compression or
  pricing jobs.
- Direct both lanes to the shared ten-minute circuit engine and candidate backlog.
- Start with a small vertical slice that can reuse an existing behavior dataset and result scorer, then exercise it on
  a genuinely new, prior-art-cleared candidate.
