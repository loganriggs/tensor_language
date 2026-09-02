# Rung 463: direct-residual versus distributed-suffix path factorial

Status: prospective explanatory design, frozen after rung 462 and before running any multi-write patch. It uses the
already-open code role, so it is not a new OOD confirmation or an adoption/compression result.

## Question

Rung 462 patched 19 later attention/MLP writes one at a time. No write recovered 10% of the L8H4 equality effect;
MLP10 was best at 5.63%. Small MLP mediators nevertheless preserved the far>near and one>multiple context law.

The native L8H4 equality term can affect final logits through two broad routes:

1. its own 1,152-dimensional attention8 write remains in the residual stream and is carried forward by residual
   mixing even if later module writes are held fixed;
2. it changes later attention/MLP states, causing a distributed set of later writes that then affect the output.

Rung 463 separates these routes with a fixed path factorial and measures their interaction. This is not a rank
decomposition and does not assume that individual write effects add.

## Frozen object, role, and write list

- documents: all 192 `ood_code` rows, with fixed halves `0:96` and `96:192`;
- source intervention: remove L5H5 in every analytical arm; base also removes L8H4, while reference retains native
  L8H4;
- natural-fit scale and L5H5-score hybrid remain hash-bound companions but are not used to choose a path;
- later public writes, in order:
  `MLP8`, then `attention9`, `MLP9`, ..., `attention17`, `MLP17`;
- cells: near, far, one predecessor, multiple predecessors, all positive, and off target.

Native and empty analytical replay are instrument controls. No other pair, reader, score factor, row role, context
definition, QK branch, rank, or SEALED outcome may be opened.

## Fixed path arms

For each four-document batch, cache every later write on base and reference trajectories, then run:

- `base`: both equality terms removed;
- `reference`: native L8H4 restored, with its naturally induced later writes;
- `direct_only`: native L8H4 restored, but replace **all 19** later public writes by their cached base values;
- `mediated_all`: keep L8H4 removed, but replace **all 19** later writes by cached reference values;
- `mediated_mlp`: keep L8H4 removed and patch the ten MLP writes only;
- `mediated_attention`: keep L8H4 removed and patch the nine attention writes only;
- 19 cumulative suffix arms: for boundary `j`, keep L8H4 removed and patch every cached reference write from `j`
  through MLP17. Thus the MLP17 boundary patches one write; the MLP8 boundary equals `mediated_all`.

An attention patch replaces only its public residual-stream write and retains the current trajectory's first-value
channel. A multi-write patch uses a self-consistent set cached from one reference forward on the same documents and
positions. No activation moves backward in depth.

## Metrics and interaction accounting

For arm `P` and context `C`,

`effect(P,C) = [sum CE_base(C) - sum CE_P(C)] / token_count(C)`

and `recovery(P,C) = effect(P,C) / effect(reference,C)` when the reference stake is positive.

The direct/distributed CE interaction is reported without assuming additivity:

`interaction(C) = effect(reference,C) - effect(direct_only,C) - effect(mediated_all,C)`.

A negative interaction means the two isolated paths redundantly recover the full effect; a positive interaction
means the full trajectory uses their combination more than their separate interventions predict.

For the suffix curve, order boundaries by number of patched writes (one through nineteen) and report Spearman
correlation with recovery, every adjacent increment, and both document halves. A smooth curve without a knee is a
distributed-depth result, not a failure.

## Registered predictions

### A. Instrument

All parent/source/model/row/mask/write-order hashes hold; replay relative-squared error is at most `1e-12`, factor
reconstruction error at most `1e-10`, every patch tensor has exact document/position/module identity, each patch
fires exactly once at each declared write, call census is exact, and SEALED remains closed.

### B. The direct residual route carries most of the equality effect

`direct_only` all-positive recovery is at least `.50`, is positive in both fixed halves, preserves far>near and
one>multiple effects pooled and in both halves, and has absolute off-target effect at most `.01 nat`.

### C. Induced later writes have a coherent cumulative effect

`mediated_all` all-positive recovery is at least `.15` and positive in both halves. Across the 19 suffix arms,
patched-write count versus all-positive recovery has Spearman at least `.70` pooled and positive in both halves.
`mediated_all` exceeds rung 462's frozen best single-write recovery `.0563254` by at least `.05`.

### D. The distributed route is mainly carried by MLP writes

`mediated_mlp` all-positive recovery is at least `.10` and its effect exceeds `mediated_attention` pooled and in
both halves. This prediction comes from rung 462's three best positive single-site mediators all being MLPs
(MLP10/13/11); it is now tested as a combined path rather than inferred by adding them.

### E. The stronger isolated route preserves the context law

Whichever of `direct_only` and `mediated_all` has the larger pooled all-positive effect must have far>near and
one>multiple effects pooled and in both halves. Its absolute off-target effect is at most `.01 nat`. Freeze the
winner by the pooled effect only for this reported decision; do not search boundaries or redefine the route.

The strong null is instrument failure; both direct-only and mediated-all recovery at most `.10`; neither isolated
route preserving both pooled context signs; or the full native reference stake nonpositive in any primary cell.

## Claim boundary and successor

- B pass / C fail: context use is mainly alignment of the original L8H4 write with the residual/readout path.
- B fail / C pass: no single write mediates, but the induced suffix collectively does.
- B and C pass: both routes matter; the measured interaction determines complementarity versus redundancy.
- B and C fail or strong null: this public-write factorial misses a normalized-state or hidden channel and must be
  redesigned before any branch split.

D locates the distributed route by module type; E checks that the chosen route still carries the named context law.
Even a full pass identifies paths, not semantic coordinates, independent OOD confirmation, or saved parameters.
The next test is a held-out targeted removal/interchange of the identified route, or a hidden-channel audit under
the failure branch. Do not answer a path failure with rank reduction.
