# Suffix join: within-layer writers and key/value port factorial

SUFFIX_JOIN_WRITER_V1 nominates L2 (zero-based): remote S→J cuts lose .891–.971
hop3 gold probability while lower-hop/matched-edge controls are small. L0/L1 do
not meet the shared writer bar. Restoring final K1/K2/V after joint L0/L1/L2 cuts
recovers about100% of behavior but misses full-distribution rescue, including
OOD B2-later mean KL .00206. A/B/D true,C false; preserve that miss.

Next intervention cuts only L2 S→J edges. L2 output changes only at J positions,
so all final-layer query inputs at the cold query remain unchanged. Restoring all
three final source ports K1/K2/V at J should therefore restore the **query's**
full logit vector exactly. This statement does not cover logits at J as a query,
whose own residual and Q ports were also altered by the L2 intervention.

Fresh16 IID24-cycle worlds seed13909 and16 two12-cycle OOD worlds seed13910,
counterbalanced through6 orders at binding slots3/11/19, each with queryhops0..3.
32 independent worlds,768 correlated variants. Same semantic S/J source masks and
adjacent nonjoining controls as the parent; no head fitting, rank or seed sweep.

Arms: native; each of4 L2 heads' S→J edges cut; each corresponding control-source
cut; all8 subsets of native final K1/K2/V restored after all-head L2 S→J cut.
The empty subset is the all-head cut. Native donor ports come from the same
world/order's hop0 and are reused across all4 query hops. All reference activation
use is explicitly an intervention; it is not an independently extracted runtime.

A instrument: synthetic masked head/capture/restore controls; all-port query rescue
native agreement1e-9; source-port equality across queryhops1e-9. Native hop3
accuracy>=.8 in each population/orientation. No per-row correctness filtering.
B head localization: for each orientation at least one identical head across both
populations has hop3 target loss>=.5, control-source |loss|<=.1 and lowerhop0/1/2
|loss|<=.1. Report all heads and overlap/difference across orientations; no demand
that native heads must be different semantic operations.
C directional port hypothesis: relative to all-head cut, B2-later (forward join)
V-only rescue>=.8 of base loss and K1+K2-only<=.1; B3-later (backward join)
K1+K2-only rescue>=.8 and V-only<=.1, in both populations. Base loss>=.5 required.
These are separately falsifiable semantic port hypotheses, not assumed from names.
D exact joint port composition: all-port query rescue agrees with native at1e-9
and all8 subset outcomes are recorded; do not assume probability additivity.
Persist per-row query logits for all17 arms. If C fails, retain a coupled join
representation instead of relabeling native K/V fields or altering thresholds.

Passing B nominates native writing heads. Passing C identifies directional port
use but still needs a weight-level middle-entity matching operation and compact
extraction. All400640 parameters remain native and charged. Batch4 FP64,1800s,
<256MiB/tensor, GPU only via managed enqueue. Reuse the existing experiment core.
