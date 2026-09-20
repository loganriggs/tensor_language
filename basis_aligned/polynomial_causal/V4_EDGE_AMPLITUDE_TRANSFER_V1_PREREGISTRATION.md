# Frozen mixed-edge amplitude transfer

CIRCUIT track. Previous native carry-only control passes8/8 opened unit-edit composition cells with2.91%worstnumber/1.51%worstcontrol error. Test reuse across independently varied signed and strengthened amplitudes, not a newly fitted tensor.

Freeze source selector weights and mixed-edge core compiled from baseline plus unit singleton inputs. Seven (subject,attractor) settings: (1,1),(.5,1),(1,.5),(-1,1),(1,-1),(2,1),(1,2). Both query/key token orders occur; map amplitudes using actual later-query/earlier-key positions. Recompute native singleton backgrounds at each setting and charge them. No refit, no joint state supplied to predictors. Native joint state is validation only. These are prospective amplitude conditions on opened text, not fresh-text OOD.

Compare carry-only, direct linear-numerator response and exact normalized MLP correction. Carry-only uses the existing mixededge after native additive-background MLP; linear uses the directly compiled reduced response with exact denominator. Exact uses nativeMLP on edge-corrected preMLPstate, serving as independent positive control. Source selectors remain contextual oracles, not an extracted selector.

Per family and amplitude setting, error for every observable is divided by min(norm(subject-number-effect),norm(attractor-number-effect)). Number<=.1 and everycontrol<=.05. Report all56cells and no averaging away fails. Halfamplitudes lower signal: retain denominator norms and exactprecision check.

pred_a_instrument:12prefix64native84joined, unitnative anchor<=1e-8, every exact-correctedoutput<=.001 weaknormalizederror. Report localedge discrepancy separately (rounding can matter).
pred_b_carry_transfer: carry-only passes all56cells.
pred_c_linear_transfer: linear-only passes all56cells.
Failure of pred_b closes this coarse carry simplification over the tested amplitude domain; do not change the gates. Failure of exactcontrol is an instrument problem requiring diagnosis before interpreting omissions.

Pricing:56explicitMLP11calls beyondsuffixes. Core31296values/context, carry no additional response coefficients; linear62614/context at each declaredbackground, plus native background/suffix/sourcegeneration. No claim of fixed token-input extraction. No dense duplicate exports.
