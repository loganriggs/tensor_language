# One residual6 sequence supplies the full city-removal generator

Generate attention7 at every position from native residual6 and token IDs,
using all heads and both QK factors. Generate normalized MLP7 inputs, then
fold five complete head8.2 maps through MLP7 Down: Q1,K1,Q2,K2,currentV.
Generate all head8 query/key RMS factors and rotary internally. No supplied
query fields or mixed8 RMS. Approximate mixed8 RMS by the other-source norm,
retaining every MLP7 numerator term. For Q/K, input RMS cancels except for the
native epsilon contribution; retain epsilon, do not silently change the model.

Inputs: one residual6[B,T,1152] array, token IDs, city index, destination mask.
Output: complete city-removal delta at attention8. Prefix0–6 and native MLP8+
suffix remain external. One array does not imply less state: fullT32 requires
36,864 native floats versus23,169 at the prefix-plus-query interface. Charge all
attention7 weights/tables, L7/R7, five folded readers and remaining head8 maps.

CPU preflight on40opened Pile sequences: generated query-field relative error
<=1e-4 on every sequence; native city-write error<=.10 on every sequence;
unsupported tokens fail, all outputs finite/support zero. No fits or new rows.

Full-model CPU screen:19padded batches,760sequence-equivalent forwards,
342batched block calls, two threads,600second cap. Arms: native, native complete
city-removal reference, generated one-input removal,16same-site norm-matched
random edits. Random CPU seeds18093000+1000*k+context_id, paired seed/opposite
cue signs. Native attention8 injection; recompute full MLP8 and later layers.

Gates:
- a: CPU native/full-reference scores replay saved GPU native/full removal
  <=1e-4absolute AND<=1e-5relative; finite outputs; support zero; null norm error
  <=1e-5; exactly760sequence-equivalents/342block calls; CUDA uninitialized.
- b: generated target-effect error<=.35; target RMS>=1e-5;>=90/120 native capable
  contrasts (native UK−US margin>=.1).
- c: each of four control-reader RMS movements<=.5 generated target RMS.
- d: positive attenuation>=.90 among capable contrasts, mean>=.02.
- e: generated target RMS>=2times random median and beats all16nulls.

Report errors on untouched/substituted arms separately with no new subgroup
gates. These are opened selection/implementation data; no independent OOD or
composition claim. Passing motivates fresh confirmation. The earlier exact
native-RMS export and its certificates remain unchanged.
