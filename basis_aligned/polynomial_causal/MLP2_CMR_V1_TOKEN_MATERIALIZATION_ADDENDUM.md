# MLP2 CMR v1 token-materialization addendum

Frozen after publishing document identities and before reading their text or token
lengths.  This addendum authorizes only CPU materialization of the four frozen roles;
it authorizes no checkpoint load, activation capture, fit, selector, validation,
replication decision, or scientific outcome.

## Exact transformation

Read the `text` field for each frozen FineWeb document index from the pinned Parquet
bytes.  Encode it with GPT-2 `tiktoken.encode_ordinary`.  For every document:

1. retain at most its first 257 tokens;
2. if it has fewer than 257 tokens, right-pad with GPT-2 EOT token ID 50256;
3. publish a Boolean mask over the 256 input/target positions;
4. a position $q$ is eligible exactly when $64\leq q<256$ and token $q+1$ is
   still inside the original document.

Thus padding only makes a rectangular tensor.  No padded target is fitted or scored.
The transformer is causal, so padding strictly after an eligible target cannot change
that position's state or prediction.

The materializer reads only Parquet row groups containing a frozen index.  Row-group
access order cannot affect the role or row order, which is exactly the order in the
published role manifest.

## Support gates

Each of `FIT_MEAN`, `FIT_SELECTOR`, `VALIDATION`, and `REPLICATION` must contain:

- exactly 192 distinct frozen document identities;
- at least 128 documents with one or more eligible positions;
- at least 16,000 eligible positions in total;
- token IDs in `[0,50256]`;
- exactly EOT after the clipped original-token length and nowhere substituted inside
  the original prefix.

Failure of any gate publishes no token receipt.  It does not permit replacing a
short document, changing a role, lowering a threshold, concatenating documents, or
opening the model.

## Publication and authority

Publish create-only in this order:

1. `mlp2_cmr_v1_token_rows.pt` containing role-keyed rows, eligibility masks,
   original token counts, and frozen document indices;
2. `mlp2_cmr_v1_token_rows_manifest.json` binding every tensor semantically;
3. `mlp2_cmr_v1_token_rows_receipt.json` last.

The receipt licenses these bytes only as token inputs for a later independently
source-closed MLP2 CMR collector.  It does not itself authorize a model forward.

