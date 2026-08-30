# Bracket closure v1 fresh-row amendment

Status: prospective CPU-only row authority. No model, checkpoint, logits, or outcome
may be opened by this lifecycle.

The canary roles are replaced prospectively by three document-disjoint roles:
`fit`, `select`, and `ood`. Each role contains exactly 160 prose and 160 code rows,
one row per source document and source file. Within each role/domain, exactly 32 rows
are allocated to each outcome-blind primary score stratum: compatible closer,
incompatible closer, no opener, quote control, and punctuation control. Primary
stratum is the earliest scored occurrence (positions 64:256), breaking equal-position
ties in that fixed order. Allocation uses a fixed SHA256 order and never model output.

After allocation, every role/domain must have at least 30 distinct documents and 30
positions in each of the five score cells; `all` covers every position 64:256. Empty
or underpowered cells fail before publication. FIT may only finalize delimiter/token
metadata; SELECT is discovery; OOD is sealed for one later confirmation. No role may
be repurposed.

Exact document IDs, source files, source revision/blob identity, normalized Python
hashes, rows, and 32-token prefixes are disjoint across roles and excluded against an
authority-bound historical metadata registry. Code and prose sources are separately
typed. Candidate bytes, tokenizer/source/license identity, delimiter registry, prior
registry bytes, source commit/blobs, allocation realization, support masks, and output
payloads are hash-bound.

An independently authored audit JSON must bind the exact pushed source commit, source
hashes, and authority hash with `status=GO` and `outcome_access=false`. The freezer
cannot mint that audit. Installed role files are semantically replayed, source and
history are rechecked, the owned lock is rechecked, and the success receipt is linked
last. Any failure is terminal and cannot coexist with a success receipt.

