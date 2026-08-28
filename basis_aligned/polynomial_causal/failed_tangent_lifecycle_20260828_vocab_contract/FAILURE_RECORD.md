# Preserved failed tangent lifecycle: padded-vocabulary contract

Date: 2026-08-28

The final response stage rejected this authority/geometry pair before its first Fisher
target because the transaction incorrectly equated the tokenizer support (50,257 valid
token IDs) with the checkpoint's padded embedding/logit width (50,304 rows). No target,
gradient, response, or outcome artifact was published.

The archived files are retained intact:

- authority receipt: `tensor_bilin18_tangent_authority_receipt.json`;
- geometry artifact: `tensor_bilin18_tangent_geometry.pt`;
- geometry receipt: `tensor_bilin18_tangent_geometry_receipt.json`.

The archived geometry bytes have SHA-256
`5f8aeac18fef087b9217eedfde4fff254275e94f2b1b9716c03a3a1bcd5a40be`,
identical to the later canonical geometry because the corrected vocabulary predicate
did not alter the rank-640 program or natural-write geometry. The old receipts remain
distinct and must not be substituted for the canonical post-fix authorities.
