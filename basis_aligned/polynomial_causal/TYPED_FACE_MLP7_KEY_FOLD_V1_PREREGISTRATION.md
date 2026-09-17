# Native head8.2 key numerators through MLP7, V1

Normalization response screen passed on opened contexts (both-frozen target
error .0589, cell max .0713) despite the failed write-level .10 gate. Preserve
both results. No omission adopted without fresh confirmation.

Exact native path: head8.2 K1,K2 readers backward through block8 residual
mixing and MLP7. Let g be block7 residual after attention, z=N(g),
h=(L7 z)*(R7 z), m=D7 h+b7, and r8=l80(g+m)+l81 e.
For j in{1,2}, Kj r8=l80 Kj g + l80(Kj D7)h + l80 Kj b7 + l81 Kj e.
Keep three named sources carry(g), MLP7(m), initial(e), self and both ordered
cross terms in ||Kj r8||². Carry actual residual RMS rho8 and native head-key
RMS; no polynomial-only approximation or fitted proxy.

Use eight existing context/cue sequences, opened. Eight native forwards,
300s maximum, no intervention yet. Hook native g,z,m,r8 and head8 key channels;
compile both Kj D7 matrices and bias readers from weights in float64.
pred_a: normalized-source reconstruction vs native head keys <=1e-5 relative;
pred_b: folded/direct MLP7 key products <=1e-12 in float64 and native projected
MLP7 <=1e-5; pred_c: full ordered3x3 Gram sum reconstructs key squared norm
within1e-12 in float64. All nonfinite/dead zero denominators are invalid.
Report per-source paired-change norms AND aligned fractions/cosines, with no
causal dominance claim and no fitted source selection. Retain all three sources.

Null: the replay or source interfaces are wrong; no new causal interpretation.
Native g and residual normalizers remain open; L7/R7 native input maps remain
charged. This is a computation-path fold, not fresh identification or a simpler
executable circuit. The next causal check, only after a valid fold, freezes the
MLP7 city-key contribution versus carry while preserving both QK factors and
source interactions. No native suffix proposal precedes response measurement.

Literal price: two128x4608 folded reader maps, two128 bias vectors and two
block8 residual scalars, plus externally charged L7/R7 and state/normalizer
interfaces. Do not claim savings when adding folded maps while keeping D7 for
exact residual norm generation. Reuse generic paired-write runtime for followup.
