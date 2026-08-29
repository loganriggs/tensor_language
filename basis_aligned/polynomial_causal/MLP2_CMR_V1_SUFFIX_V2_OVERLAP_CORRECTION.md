# MLP2 CMR v1 SUFFIX v2 overlap-summary correction

Frozen immediately after the v2 receipt and before opening `VALIDATION`.

The response collection, score vectors, support tensors, support hashes, probe-half
stability, and gauge/permutation audits are intact. The JSON `support_overlaps` field
is wrong: `support_jaccard` constructed Python sets directly from scalar tensors, so
two numerically equal channel IDs were represented by distinct tensor objects. This
reported all fifteen overlaps as zero.

The correction converts every scalar support entry to a Python integer before set
operations, independently replays the six 512-element support hashes from the frozen
bundle, and publishes intersection counts and Jaccards receipt-last. It does not
recompute responses or selectors and opens no model, tokens, targets, logits,
validation, or replication. The original result remains preserved; only its overlap
summary is superseded. Validation authority must cite the correction receipt as well
as the original selector receipt.

