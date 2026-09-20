# Native mixed attention edge: registered diagnostic

Opened OODv2 composition data; no new OOD claim. Frozen subject/attractor swap5 source edits, four contexts, sixteen settings plus baseline. No parameter fitting.

Claim: token-local input/head RMS and causal unnormalized two-QK attention imply that the only mixed block11 write is at the later edited query, from the earlier edited key. Compile its two score products into a 3x3 coefficient core per head, retaining query/head/key/value norms, rounded RoPE, first-value cache and fixed baseline downstream readers. Price is 300 scalars per context for nine heads and four readers; source states and reader generators remain charged.

Instrument: 12 prefix captures, 104 float64 suffixes, 68 first-attention writes, 16 reader reverse calls. Saved float64 composition replay <=1e-10. Folded versus independently evaluated projected mixed write <=1e-10, and mixed write outside nominated query <=1e-10. Existing planted independent edge/dead-axis control passes ~1e-15. This uses the model's explicit float64 operations; it is not a new float32 replay claim.

Finite causal removal: subtract A(s,t)-A(s,0)-A(0,t)+A(0,0) from attention11 before continuing the full nonlinear suffix. Original interaction I=Y(s,t)-Y(s,0)-Y(0,t), for signed baseline-minus-edit Y. Compare its norm to the remaining interaction after removal. Strong preregistered hypothesis: remaining number interaction norm <=50% of original in ALL thirteen previously failed attractor-increment cells. Failure rejects this sufficient-localization hypothesis, not the exact algebra. Also report fixed-reader versus actual finite removal response; do not conflate first-order transport with finite causal effects. This attenuation measure does not itself prove correction of the predictor's error.

No changes to frozen composition runner or dependencies. Retain all outcomes and failed gates.
