# Discarded arms

`tf_slots_d2_w128_b8192_s{0,1}_nslots2` — a partition DOSE-RESPONSE arm
(`--n-slots 2` at depth 2) run during the round-2 review. **Invalid and not
reported.** With `n_slots < 2*depth` the masked decoder's write masks are built
by a python slice that runs off the end of the row and is legally empty, so the
write masks came out `[64, 64, 0, 0]`: the entire second block — its attention
*and* its MLP — wrote nothing into the residual stream, and nothing reported it.
The models trained happily (CE 4.8898). See `tf_reviewer_round_2.json` R8; a
guard assertion now lives in `tf_model.TinyBilin` so this cannot recur silently.
