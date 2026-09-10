# Coupled lexical/form readout factors, CPU v1

10 September2026. Original handoff/pilot; existing calibration and numerator/RMS
factorials are precedent. New endpoint: lexical-margin drift under fixed all36
form command F, at both lemmas in both A1/A2 frames. All opened rows retained.

Use LEXICAL_FORM_INTERCHANGE_V1_STATES.pt; FB versusB andFX versusX. No native
model/checkpoint/GPU. Recover actual four token weights from fixed H components.
For unit e, h=s e+h_perp; a_t=U_t e, c_t=U_t h_perp. Score is
30*tanh((a_t*s+c_t)/(30*rho)), rho=sqrt(||h||^2/1152+eps32).
Readout lattice independently chooses base/edited s,c,rho (8 corners). Also
predict the physical scalar-only curve with c_base and
rho_physical(s)=sqrt((||h_perp_base||^2+s^2)/1152+eps32).
Its base context is native; it does not predict initial state from tokens.

A instrument: reconstructed endpoint logits maxabs<=1e-3 versus saved native;
FP64 numerator split and physical-state norm identities maxabs<=1e-10; finite
outputs and each lexical-drift reference norm>1e-4. No-op at s_base reproduces
base readout under same tolerance. Raw e-unit error<=1e-10.
B physical scalar-only explains drift: A and relativeL2 error<=.20 on both
lexical margins, separately for FB andFX in BOTH frames, against native change.
C scalar+actual norm explains drift: same B bar for a*s_edited+c_base with actual
rho_edited. This is conditional on supplied native norm, not norm extraction.
D complement+actual norm explains drift: same bar for a*s_base+c_edited with
actual rho_edited. Opposes scalar-reader sufficiency. No refit or axis change.

Report entire8-corner errors, physical prediction, selected four-score effect
errors and interaction-coordinate changes; small lexical denominator checked
separately. Paired4000 resamples seed9111581+cellindex. Controls distinguish
floating point agreement from explanatory sufficiency. Original native-choice,
selectivity and composition failures are neither overwritten nor rescored.

Literal source/native dependencies retained. Program coefficients a_t,c_base,
s_base and perpendicular norm are conditional summaries, not a discovered
upstream producer. CPU follow-up: compare lattice residuals to determine whether
an explicit fixed readout coupling suffices or complementary-state production
must be explained. Preserve nulls and do not add directions/ranks/thresholds.
