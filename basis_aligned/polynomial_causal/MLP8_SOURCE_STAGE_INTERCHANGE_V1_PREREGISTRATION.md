# Full MLP8 mixed-source reuse across sentence layouts

Use all 16 original/fronted lexical pairs and all five-factor corners in the fixed
THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1 bank. Source C_r=Q_oh MLP8_r at every position.
Q_oh projects object-number × attractor-kind interaction, conditional on other
factors. It is produced from observed counterfactuals, not independently extracted.

The proposed interface has three positions: first token where both factors are
causally available, `to`, and action. They are the last three positions in both
layouts. The first position is an attractor noun in original and object noun in
fronted; this is an information-stage map, not a claim of identical noun roles.
Map donor C_p at these positions to recipient positions in the same residual basis.
No fitted alignment, rotary correction, rescaling, head or row selection.

Let B_r=MLP8_r-C_r. Run all downstream computations normally with B_r, B_r+C_r,
and B_r+T(C_p). E_pr=z_r(B_r+T(C_p))-z_r(B_r), with p producer, r recipient.
The diagonal is native-minus-full-source-removal, replaying the full arm of
MATURE_VALUE_MLP8_CONSUMERS_V1. Unlike COMMON_INTERCHANGE_V1, this source reaches
ALL consumers; attention9 is live. No directional clamps are used.

A instrument: native and full-removal three-reader outputs replay parent; self
interchange full vocabulary matches native; source before maturity has relative
Frobenius norm <=1e-8 (zero source fails scientific gates, not mechanics).
Incoming MLP8 output and first-value9 state bitwise checked. Fixed row/corner and
last-two-token alignment, finite outputs, restored hooks. All output bridges use
maxabs<=1e-3 AND relative Frobenius<=1e-5. CPU stage-map, original intervention and
producer/reader accounting controls pass. Preserve invalid instruments separately.

On Q_oh effects, measure correct margin, centered three-answer vector and centered
full-vocabulary vector. Every rule requires all three readouts and both directions
in EVERY pair. Zero denominators fail; no post-result normalization or fallback.
B producer-following: E10 matches E11 and E01 matches E00, relative error <=.10.
C reader-following: E10 matches E00 and E01 matches E11, relative error <=.10.
D interaction: ||E11-E10-E01+E00||/max(||E00||,||E11||)<=.10.
Report all errors and separate task-only counts, without promoting failed rules.

Native capture4 + source removal4 + cross replacement4 + self replacement4 forwards
per pair =256 forwards /4096 sequence instances, zero fits, 600-second cap.
Managed GPU runner only. All 545902902 native weights and counterfactual producer
inputs retained, saving0. This is a reuse screen on previously opened layouts,
not fresh OOD or a complete semantic circuit. If B/C fail, close this fixed
stage-invariant source/reader description; do not open a position/head/rank sweep.
