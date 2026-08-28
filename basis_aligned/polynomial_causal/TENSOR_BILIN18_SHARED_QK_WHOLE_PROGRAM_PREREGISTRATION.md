# Preregistration: first compressed standalone bilin18 program

Date: 2026-08-28

Status: composition replication on already-opened roles. Rank 384 was selected by the
parent attention frontier, so this run opens no fresh selection or OOD authority.

## Candidate

The complete standalone program keeps the exact embedding, residual shell, dense
bilinear MLPs, unembedding, RMSNorms, and softcap. At each attention site, Q, K, Q2, and
K2 share one activation-weighted rank-384 input encoder with four typed decoders. V,
output projection, rotary state, attention lambda, causal score-product contraction,
and the first-value bus remain exact.

The shared attention bank is refitted bottom-up on the frozen 480-row fit role. The
complete program is then scored after the checkpoint object is destroyed.

## Registered measurements and gates

1. **Complete price:** exactly 490,165,686 stored float32 values, a saving of 55,738,368
   values (10.2103%) from the 545,904,054-value dense program. Fitted token tables,
   native calls, and native module references are zero; total token support is true.
2. **Covered predictive composition:** CE harm versus native is at most 0.03 nat on
   both skip7000 and skip11000 roles. Replayed covered CE differs from the parent
   shared-QK attention result by at most 0.003 nat.
3. **All-position composition:** CE harm versus native from position 64 onward is at
   most 0.05 nat on both roles. Seen and unseen-current-token subsets are reported
   separately; no subset is omitted from the all-position gate.
4. **Context transport:** on the frozen prefix intervention, downstream current tokens
   remain fixed, the compressed effect is nonzero, context-delta recovery

   $$
   1-\frac{\lVert\Delta z_{\rm compressed}-\Delta z_{\rm native}\rVert_2^2}
           {\lVert\Delta z_{\rm native}\rVert_2^2}
   $$

   is at least 0.90, and delta cosine is at least 0.95.
5. **Independence:** native/program storage is disjoint, the checkpoint model is
   garbage-collected before compressed scoring, and every retained module comes from
   the owned tensor-program implementation.
6. **Replication:** CE harm differs by at most 0.01 nat between the two roles.
7. Sources, tests, role hashes, exact-parent and attention-parent receipts are bound in
   a create-only artifact. Code and tests are committed before the single GPU run.

Passing licenses the first compressed complete-program point on these already-opened
roles. It does not establish fresh OOD generalization, semantic circuit explanation,
editability, or optimality of rank 384.
