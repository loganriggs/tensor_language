# Terminal-copy attention checkpoint check v1

**Purpose:** an engineering identity check, not an E4 scientific outcome. It is frozen
before loading the bilin18 checkpoint and may only determine whether the owned
per-head adapter is faithful enough to become part of a later E4 authority.

The check loads the pinned local checkpoint in bfloat16 on CUDA. With seed
`2026082917`, it generates one `[2,32,1152]` random state for block 0, obtains the
native shared value bus, then generates independent states for the five distinct
layers containing the preregistered six copy heads: 5, 7, 8, 13, and 14.

For every layer it compares:

1. the native attention write with the adapter's unpartitioned full write;
2. the native write with the sum of all nine separately projected head writes;
3. the native and adapter shared value buses;
4. the adapter's literal storage and zero-native-call price receipt.

The check passes only if the unpartitioned replay and value bus are bit-identical,
the all-head relative error is at most $2\times10^{-3}$, the all-head maximum error
is at most `0.02`, all outputs are finite, and the source/checkpoint/create-only
lifecycle closes. These loose recomposition limits cover bfloat16 accumulation order;
the observed errors are reported and become the evidence for any tighter execution
authority. A failure is preserved and spends v1. Passing closes only the checkpoint
identity portion of the per-head-adapter blocker. It does not authorize rows, label
selection, behavioral scoring, extraction, removal, or an E4 claim.

