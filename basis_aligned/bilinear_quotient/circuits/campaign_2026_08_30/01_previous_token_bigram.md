# Previous-token and bigram lookup

## CURRENT tier: 5

The layer-0 computation has been replayed from token embeddings, weights, and rotary
position to logits.  Fresh terminal removal/OOD certification remains separate.

## Behavior and tensor program

Endpoint: next-token CE where L0H3's native top read is offset `-1`.  Matched cells are
self (`0`), other-offset, and diffuse queries.

Fixed tensor form: each QK pattern is a product of two token-and-RoPE bilinear forms;
values are exact `c_v(rms_norm(wte(token)))` lookup rows followed by native projection.
There is **no learned router**: token identity and position deterministically index the
fixed program.  The offset-`-1` behavioral mask is an analysis/removal gate, not part of
the extraction executor.

## Evidence

- [`head_0_3_exact.py`](../../head_0_3_exact.py) and
  [`head_0_3_exact_results.json`](../../head_0_3_exact_results.json): head replacement
  `-0.0` nat; shuffled table `+0.14675`.
- [`layer0_fold.py`](../../layer0_fold.py) and
  [`layer0_fold_results.json`](../../layer0_fold_results.json): layer-0 replacement
  `-0.0`, shuffled `+0.23687`, layer-1 boundary `+1.47026`.

## Terminal gates

Extraction replaces L0H3, then all L0 attention, with the exact lookup and requires
`|dCE| <= 0.001` on every split; shuffled cost must be at least `0.05`.  Removal deletes
only L0H3's offset-`-1` contribution and uses the default target/specificity gates.
Collateral includes self/other offsets, induction, and global text.  OOD holds out
bigram pairs, frequency quartiles, domains, and sequence lengths.

Shared-owner caveat: previous-token heads relay induction.  Their parameters and effect
cannot be credited again to induction.

**Next experiment:** fresh-role terminal certification of exact extraction,
offset-conditioned removal, and unseen-bigram OOD.

## 2026-08-30 discovery update

The preregistered 96-document SELECT assay completed. Full analytical replay was
bit-exact and candidate arms made zero native layer-0 attention/Q/K/V/O calls.
Extraction recovered `0.9421` of the deleted head's effect on previous-top positions
(95% interval `[0.9133, 0.9707]`) and transported essentially unchanged to unseen
bigrams (`0.9417`, versus `0.9442` seen). The shift-minus-two null recovered `0.1529`.

The terminal removal claim **failed**. Previous-top removal damage was `+0.06249` nat,
but self-top damage was approximately `+0.06321`, making specificity `-0.00071` with
95% interval `[-0.01311, +0.01156]`. Global collateral was `+0.06191` nat. Therefore
this entry remains mechanistic Tier 5 but has **extraction/OOD: discovery pass** and
**selective removal: discovery fail**. It is not eligible for fresh terminal promotion
under this endpoint, and no post-hoc argmax gate will be added.

Details: [`PREVIOUS_TOKEN_TENSOR_DISCOVERY_FINDINGS.md`](../../../polynomial_causal/PREVIOUS_TOKEN_TENSOR_DISCOVERY_FINDINGS.md).
