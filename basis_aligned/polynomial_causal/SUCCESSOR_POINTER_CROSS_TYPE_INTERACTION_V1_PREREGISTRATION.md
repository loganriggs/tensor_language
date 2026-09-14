# Successor pointer-depth cross-type interaction V1

The prospective length-six confirmation established a native-oracle mediator spanning attention and MLP responses at layers 8–12. On the original screen, neither `pointer_attention` nor `pointer_mlp` passed both families, while their joint group did. This experiment tests whether the joint group is an additive union or a cross-type causal interaction. It reuses every frozen length-six row without row selection and adds the two component arms required by the factorial.

For coherent-to-late and coherent-to-early transplants, run three fixed arms: attention8–12 only (`A`), MLP8–12 only (`M`), and their union (`AM`). Preserve the exact self and all-module ceiling. Cache the final residual for each arm. At both the 1152D final residual and the pre-softcap normalized unembedding score, define the interaction by inclusion–exclusion: `I = effect(AM) - effect(A) - effect(M)`, with the native recipient as zero. Post-softcap answer margins are reported separately. This distinguishes internal downstream interaction from the final tanh softcap, although final RMS normalization remains part of the pre-softcap readout.

Per family, report component and joint target projection/cosine, additive reconstruction of the joint, interaction projection on the coherent-minus-late backward target, final-residual interaction/joint RMS, and interaction controls for late-forward and early-backward margins. The already confirmed `AM` arm must continue to pass the original projection/cosine/control bars or the instrument is invalid.

- `additive_union`: in both families, final-residual interaction/joint RMS <= .10 and pre-softcap additive reconstruction of the joint backward effect has projection >= .90 and cosine >= .95.
- `cross_type_interaction`: in both families, final-residual interaction/joint RMS >= .20 and the pre-softcap interaction accounts for absolute target projection >= .10, while its late-forward and early-backward post-softcap control RMS are each <= .50 of backward-target RMS.
- otherwise `mixed_or_unresolved`.

No gain, fit, rank, subset, depth split, head split, row change, threshold rescue, or quantization is allowed. Price is 11 forwards over eight sequences, 88 sequence evaluations, zero updates.
