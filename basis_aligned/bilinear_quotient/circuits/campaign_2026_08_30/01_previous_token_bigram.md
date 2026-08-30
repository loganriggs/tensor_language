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
