# MLP4-induced value response: intervening attention versus MLP computation

Registered10 September2026 before implementation/native execution. The direct
residual fold passed its numerical tests but misses81–88% of the full localV9
effect vector. Next test actual intervening dependencies, not a postcomputed
source-term deletion that hides those same dependencies in its producers.

Reuse the unchanged72 opened pairs and ALL valid MLP4 output positions,
both directions, exact earlier source intervention. Receiving background,
query/key and shared-first-value payload stay native in the reader assay.
No fit, new data, source selection or change to any previous threshold.

During the MLP4 output swap, run four intervening configurations at layers5..8:
full (all modules respond); mlp_only (attention outputs fixed to receiving
native); attention_only (MLP outputs fixed to receiving native); residual_only
(both fixed). Preserve the actual receiving first-value payload in attention
return tuples. Every unfrozen downstream module recomputes its output on
its CURRENT input. Capture induced raw localV9 from each configuration.

Insert each induced localV9 field into receiving-native heads1/4, all valid
source positions, keeping query/key/otherheads/background native. Full suffix
recomputes. Also run identity clamps with MLP4 unmodified. Counts: perpanel
two native captures +eight source/configuration forwards +eight field-reader
forwards +two identityclamp forwards=20; total80forwards/1440seq evaluations.

A: bound parent/source/row/backend/checkpoint; all finite; exactcounts and
hookrestoration. Identityclamp full-logit replay<=1e-3 ANDrelFrob<=1e-5.
Residual_only V9 agrees with independent RMS(u9+gamma4*deltaMLP4) value
projection atsamebars. Full induced V9 effect reproduces parent value_full
margin effects<=1e-3/1e-5 and parent endpoint effectnorm within1e-5relative.
MLP4 swapped outputs and clamped module outputs match registered values;
allunselected localV entries unchanged. Small CPU clamp controls beforeenqueue.

B MLP-chain hypothesis: mlp_only reader effect relativeerror versus full
reader effect<=.10 in BOTH centered full-logit andmargin frames, every
panel/direction. C alternative attention-chain hypothesis: same test for
attention_only. D all three non-residual configurations have nonzero
centered-logit effects>1e-8. Report residual_only too, with no promotion.

A miss invalidates. B/C are separate hypotheses: preserve both outcomes;
no module subset/dose/rank/norm rescue. This decomposes an EXISTING PARTIAL
value path. The prior has/had full-logit carrier threshold failed and remains
failed; a chain match does not promote it to whole-task sufficiency.

All545902902 nativeparameters and receiving native background remain charged;
clamps test causal dependence, not standalone execution or weight elimination.
No learned parameters or modelupdates. A passing chain becomes a candidate
for explicit finite-response weight interpretation and fresh confirmation.
An exact pair-conditioned RMS/bilinear secant tool is available, but supplying
both endpoints does not itself constitute prediction or a shared task operator.
600-second watchdog, managed GPU enqueue only. Save immutable JSON.
