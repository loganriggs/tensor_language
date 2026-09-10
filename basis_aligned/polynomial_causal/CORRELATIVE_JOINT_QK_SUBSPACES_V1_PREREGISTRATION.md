# Joint QK1×QK2 input-product subspaces

Both tasks use both factors. The old whole-half ablation and OV rowspace audit did not test this question. Define phi_q=vec(q1 q2^T), phi_k=vec(k1 k2^T). Their inner product/128^2 is the exact native pattern, with the real head RMS, rounded rotary transforms and causal mask retained.

FIT first8 A1 pairs (family A) and first8 C pairs (family B). Per each of26 fixed parent heads, span donor-minus-base joint query changes (8 columns) and all visible key changes (<=48 columns) using FP64 SVD, singular tolerance1e-6 of largest, no rank search. Zero change gives empty basis. EVAL last8 A1/A2/C, excluded from fitting but previously opened examples, not new OOD evidence.

Replace joint features by phi+BB^T(phi_donor-phi), separately at query and keys, retaining their mixed product. A projected feature need not factor into one q1/q2 pair: this is a compiled product-port edit, not native input-vector or weight ablation. Use identical full donor values in value-only, full-routing, and both subspace arms. Project the resulting live head-output change through existing P for A1/A2 and R for C. Reference is full-routing-plus-value minus value-only: isolate conditional routing rather than crediting already strong value transfer.

- pred_a: exact28 forwards224 sequences; finite outputs; native endpoint capability8/8 each panel; factor reconstruction rel<=1e-5,max<=1e-3+1e-5*maxabs(native); no-op and independent whole-head-swap readouts max<=1e-3,rel<=1e-5; FIT basis orthogonality<=1e-9 and residual<=1e-5; joint-feature coordinate versus explicit bilinear contraction rel<=1e-10. Existing planted algebra fixtures<=1e-10.
- pred_b: own-family subspace predicts centered full-logit conditional routing correction with relative error<=.20 and signed projection recovery>=.80 in all3panels, reference norm>1e-4.
- pred_c: other-family subspace produces conditional routing effect with relative norm<=.20 in all3panels, same denominator floor.

Null insufficient or overlapping family spaces. No row filtering, rank tuning or head selection. Geometric overlap is reported separately from causal interchange. Preserve all prior failed removal results.

A basis column reshaped W reads (R_pos Q1 u)^T W (R_pos Q2 u)/(rho1*rho2), an explicit bilinear input function involving BOTH trained query matrices; keys analogously use K1/K2. Head denominators and positions remain live. This screen still uses native input producers and all weights; success does not imply independent extraction.

Price4 FIT forwards32seq plus8 forwards64seq per EVAL panel (base,donor,value,full,Aspace,Bspace,no-op,independent full reference)=28/224. Each basis bank<=48columns, each new tensor<=256MiB. Save bases, per-row errors and effect norms. Managed lane1. No simultaneous claim that these product-port edits are realized by native raw-input or weight edits.
