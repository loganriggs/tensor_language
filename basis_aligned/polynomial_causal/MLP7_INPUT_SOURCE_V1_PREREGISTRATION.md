# Earlier MLP7 input: exact three-source intervention

Now-opened 16 sequences/8contexts from KEY_SOURCE_FRESH_V1. Let
`g7 = lambda7[0]*residual6 + attention7_write + lambda7[1]*initial` at the city.
For all eight donor/recipient source masks, feed the raw hybrid into the exact
MLP7 donor generator; recompute MLP7, residual8 RMS and both key-head RMS stages.
Donor token supplies initial8 and inherited value throughout. Both QK factors
and their product are retained. The zero corner thus equals old source4, not
inherited-only. Full corner equals the compiled full face. Keep recipient
queries/current value and the head9 odd-value descendant unchanged.

Native + eight corners + direct full =160 forwards,300s. No fit or support search.
pred_a: native source reconstruction <=1e-5 relative; old-source4/zero corner and
old-full/full corner anchors <=1e-5 absolute and1e-6 relative Frobenius, target
increment RMS >=1e-5. pred_b: earlier residual6 alone predicts the full key
increment (corner7-corner0) within .35 relative L2 in each cue and endpoint.
pred_c: residual6+attention7 within .20 in each cell. pred_d: all-source Mobius
closure <=1e-12 and finite. Report all source single/pair/third effects and four
unrelated readers, without source-selectivity claims. Null: source omission
exceeds the registered effect error. Existing random-split composition failure
remains. This changes the exact path specification and identifies which upstream
computation needs a next fold; raw weight/state magnitude cannot select it.
