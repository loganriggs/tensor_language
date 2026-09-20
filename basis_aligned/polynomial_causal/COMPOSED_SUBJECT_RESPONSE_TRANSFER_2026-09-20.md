# Composition and longer-structure transfer

The frozen width8 response program now accepts explicit attention response writes before each MLP. Its shared-product runtime agrees with independent joint-core execution; no basis was refit for these tests.

| Panel / cell | Error vs dense attention12-restored suffix | Error vs full native | Native accuracy |
| --- | --- | --- | --- |
| v658 pre_beside|singular | 1.927% | 4.698% | 100.0% |
| v658 pre_beside|plural | 1.195% | 3.208% | 100.0% |
| v658 pre_beyond|singular | 2.024% | 3.518% | 100.0% |
| v658 pre_beyond|plural | 2.507% | 4.592% | 100.0% |
| v658 post_beside|singular | 2.156% | 8.595% | 100.0% |
| v658 post_beside|plural | 2.909% | 10.002% | 100.0% |
| v658 post_beyond|singular | 1.387% | 7.123% | 100.0% |
| v658 post_beyond|plural | 1.532% | 7.449% | 100.0% |
| v660 post_room|singular | 2.763% | 9.874% | 100.0% |
| v660 post_room|plural | 1.500% | 3.717% | 100.0% |
| v660 post_door|singular | 3.826% | 10.585% | 100.0% |
| v660 post_door|plural | 0.744% | 5.517% | 100.0% |
| v660 pre_room|singular | 0.929% | 4.168% | 100.0% |
| v660 pre_room|plural | 1.977% | 3.951% | 100.0% |
| v660 pre_door|singular | 2.356% | 6.523% | 100.0% |
| v660 pre_door|plural | 1.497% | 2.309% | 100.0% |

v658 uses the already opened48-row position panel. It passes the5% conditional gate, but post-beside plural is10.002% against a10% full-native limit: preserve the failure. Baseline replay is exact; unchanged exported-chain replay is2.64e-7.

v660 freezes48 longer prompts before execution: four templates, six noun pairs disjoint from calibration and v644, eight template-number cells. Historical use of the same vocabulary is not excluded. The basis and producer remain frozen; new native contexts are explicitly generated. Conditional5% passes in all cells. Native capability is100%, but post-door singular10.585% fails10%. This is structural transfer of a conditional surrogate, not full native circuit adoption. Baseline/shared-versus-joint replay is5.20e-6.

Positive red-team: both evaluations still depend on native baseline generation and initial post11 response. Neither selective removal, unrelated-behavior preservation nor cross-task reuse is established. A successful local fold cannot stand in for these tests.

Negative red-team: frozen native attention itself has approximation error; the dense restored comparator separates that error from reduced-chain error. Native capability and independent replay pass, so these misses are not explained by incapability or an obvious score/contraction bug. Do not relax the thresholds or fit a gain.

## Export correction

v655 saved enough information for prepared-context replay but omitted the six output encoders used to prepare new projected MLP contexts. v659 adds55,296 values. Existing response tensors are bitwise unchanged; max encoder duality error4.97e-14. The physical producer now has577,536 values (logical577,542 including six runtime scales), plus2,582 runtime values. New artifact size5,974,215 bytes includes saved cases. Native MLP factors and the baseline generator remain external dependencies; this accounting is not a full token-input program price.

[Encoder audit](CONTEXT_ENCODER_EXPORT_CPU_2026-09-20.json), [v659 artifact](../bilinear_quotient/circuits/followups/subject_response_v659_program.pt).

Next: v661 restores all attention12–17 responses using full-sequence latent states, with frozen bases and native baseline contexts. This discriminates omitted attention dynamics from an inadequate response space. No rank sweep or refit.

## All six attention responses (v661)

Restoring all attention12–17 responses within full-sequence reduced states passes the10% native-effect gate in all eight opened cells (0.85–3.32%). Native attention projection replay has maximum relative error4.55e-7. All bases remain frozen; native baseline contexts, first-layer values and initial post11 response remain external ports. This is a conditional suffix candidate, not a complete extracted semantic circuit. The previous layer12-only failure therefore does not refute the reduced response space; omitted later attention explains most of its gap on this panel.

Compiled storage includes sequence/background-dependent arrays and must not be summarized by the2,582-value MLP consumer alone. Per12-row batch: [{"sequence_length": 8, "batch": 12, "attention_values": 2144832, "mlp_context_values": 46656}, {"sequence_length": 8, "batch": 12, "attention_values": 2144832, "mlp_context_values": 46656}, {"sequence_length": 10, "batch": 12, "attention_values": 3255504, "mlp_context_values": 58320}, {"sequence_length": 8, "batch": 12, "attention_values": 2144832, "mlp_context_values": 46656}]. These count physically repeated arrays as stored; no unmeasured deduplication discount. Fresh syntactic transfer is registered asv662 without refitting.

## Fresh syntax (v662)

The unchanged all-dynamic program passes on48 newly frozen relative-clause/fronted-comma prompts: full-native effect error0.93–3.66%, all eight cells nativeaccuracy100%, localattention replay3.52e-7. Vocabulary is reused fromv660; syntax is new relative to selection and calibration. This strengthens conditional OOD fidelity, not selective-manipulation or closed-port simplicity. v663 registers three number-invariant modal readers and eight matched random site edits to test selectivity and control-reader prediction.

## Selectivity and a failed control prediction

v663 native modal collateral is3.52%,3.32%,3.23% of target-effect norm for can/will, may/might and should/could; target effect is12.93 times median of eight equal-L2 random same-site edits. This supports native selectivity under these limited controls. Reduced collateral prediction errors are5.17%,2.73%,7.45%, failing the5% gate for two readers. Exploratory cell disaggregation keeps native collateral below6.96%, but reduced prediction error reaches10.84%; pooling does not rescue the candidate.

v664 isolates final-decoder loss using the true final-state delta, projected through the same frozen encoder/decoder. This oracle retains92.5% and107.1% of the errors on the two failed readers. Thus the negative is largely already present in final projection, rather than being explained by recurrent attention dynamics. Oracle execution is not a predictor and is never counted as circuit success.

v665 is registered as a new same-width candidate: use four equal-weight finite-reader contrasts on the unchanged256 calibration rows when constructing balanced frames. Rank, response snapshots and runtime form are unchanged. This changes the identified observable space instead of adding capacity. Fresh validation and extraction/port closure remain outstanding.
