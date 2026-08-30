# Ordered-successor tensor SELECT v2 prospective repair

Status: prospectively frozen, outcome-blind repair. This amendment grants no row,
checkpoint, model, GPU, authority, or outcome access. Ordered-successor SELECT v1
remains `PROSPECTIVE_NO_GO`; nothing here reinterprets or cures a v1 transaction.

## Exact changes from v1

V2 preserves every v1 tensor formula, rank in `{8,16,32,64,96,128}`, spectral-null
construction, mask definition, sufficient statistic, 20,000-draw seed `2026083013`,
simultaneous order index `18999`, threshold, gate, and lowest-price selection rule.
The existing scorer accepts an exact versioned `arm_names` currency: v1 remains the
17-arm default, while v2 requires exactly the 15 registered arms. No 14-, 16-, or
17-arm metric tensor is accepted under the v2 currency, and no placeholder metrics
are synthesized for the omitted diagnostics.
It changes only the three prospectively recorded readiness blockers:

1. The nonpromotive `CURRENT_ONLY` and `V1_ONLY` source-omission diagnostics are
   omitted. The current physical factor backend necessarily stores both sources, so
   pretending to omit one with a dense zero factor would not earn its registered
   price. V2 has exactly 15 arms: native, full replay, H7 deletion, and the twelve
   true/null rank arms. No omitted diagnostic can promote or rescue an arm.
   The exact arm order, unchanged scorer identity, bootstrap constants, powered cells,
   and omission ledger are content-bound by the v2 protocol registry hash in
   `ordered_successor_tensor_select_registry_v2.py`.
2. The SELECT lexicon is exactly the non-cyclic decimal order `0,1,...,9`. Each item
   has two exact GPT-2 `encode_ordinary` surface strings: the bare ASCII digit and the
   same digit preceded by one ASCII space. A form is included only if the complete
   string encodes to exactly one token and that token decodes exactly to the same
   string. The frozen token IDs, merge-table fingerprint, and canonical registry hash
   live in `ordered_successor_digit_lexicon_v2.py` and are recomputed before selection.
3. SELECT is exactly 192 one-row-per-source-document natural FineWeb rows. Candidate
   documents begin at dataset index `200000`, which the freezer requires to be greater
   than every recursively registered historical index. It scans the next 4096
   collision-free candidate documents in pinned parquet order, taking the first
   collision-free 257-token chunk from each. A deterministic first-pass greedily takes
   the earliest document that contributes to any still-underpowered primary cell;
   after all three cells reach 200 positions and 30 documents it fills remaining slots
   with the earliest unused candidates. If the powered set needs more than 192 rows,
   fewer than 192 rows exist, or the final census fails, the transaction fails without
   a receipt. There is no retry, random draw, or outcome-dependent selection.

The powered cells remain `positive_clean`, `wrong_source_clean`, and
`no_source_clean`. Copy/overlap/exclusion and pair occupancy remain mandatory
descriptive outputs. Pair occupancy is reported in exact non-cyclic order
`0->1,...,8->9` with position and document counts and exact closure to eligible
support. Scored positions remain `64:256`. Rows and masks never gate a model write.

## Frozen row lifecycle

The source-closed freezer uses the pinned FineWeb revision, first parquet identity,
GPT-2 merge-table fingerprint, and recursive registry-wide document/index/full-row/
32-token-prefix exclusions. It binds an independent outcome-blind source audit before
opening the parquet. Publication is create-only under an inode/nonce lock:

1. source/audit/tokenizer/parquet/registry authority and namespace checks;
2. deterministic candidate harvest, powered allocation, semantic replay;
3. staged payload plus manifest, fsync, atomic cache-directory install;
4. installed hash-before/load/hash-after and exact semantic validation;
5. complete re-harvest/reallocation and protected-input replay;
6. receipt JSON semantic replay and hard-link publication last.

The final pre-link operation sequence is protected replay, exact terminal absence,
then lock claim. Failure is create-only, mutually exclusive with a receipt, and carries
no row tensor. An installed cache without a receipt is terminal failure state, never an
authority. The freezer imports no model loader and records zero checkpoint/model/GPU/
outcome access.

Because powered support cannot be known without tokenizing the prospective FineWeb
documents, this source is not self-authorizing and must not be executed until its exact
committed bytes receive a separate independent audit. No freezer is run while adopting
this amendment.
