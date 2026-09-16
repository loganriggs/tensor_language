# Subject-number embedding-decoded writer two-site V1

Registered after freezing the token-embedding grammatical-number decoder and
its disjoint-vocabulary lexical confirmation, and after freezing a new
two-clause prompt manifest, but before opening any model behavior on those
prompts.

This test composes an extracted model-native graph:

`subject token -> checkpoint embedding -> RMS normalization -> frozen number
axis and threshold -> one of two frozen cardinality-4 coefficients -> frozen
rank-one L11H3 write axis`.

The candidate graph may use only each subject token ID, checkpoint embedding
weights, the frozen decoder, and the two fixed rank-one coefficients. It may
not read the row's number or direction labels, prompt activations, logits, or
behavior. An oracle independently uses the row labels solely to audit the
candidate implementation. Candidate and oracle directions must agree at all
32 sites and their writes must differ by at most `1e-7` in maximum absolute
value. This is a validity gate, not an outcome.

The authority contains 16 untouched prompts: all four ordered number pairs
occur four times, crossed with two fixed templates. Both regular and irregular
nouns from the independently held lexical decoder authority occur. Evaluate
native, zero-write, site-1-only, site-2-only, and both-write arms: exactly five
physical forwards over 80 sequences. Positions come from the frozen row
manifest rather than hard-coded indexing.

Instrumentation additionally requires the frozen artifact hashes and checkpoint
identity to agree, finite scores and writes, decoder accuracy `1.0` against the
hidden audit labels on these 32 sites with strictly positive minimum signed
margin, zero-write replay error at most `1e-5`, and native number accuracy at
least `.75` in every site/number/template cell.

For each site and requested direction separately, a live single-site write must
have opposite-minus-native number-margin effect RMS at least `.01`, positive
effect fraction at least `.75`, and registered `can`-minus-`will` control RMS at
most `.75` of target RMS. The later-site-only intervention may change the first
answer margin by at most `1e-5`.

Let `E1`, `E2`, and `E12` be the complete two-answer effect matrices from the
site-1-only, site-2-only, and both-write arms. Additive composition requires
`E1 + E2` to predict `E12` overall and at each answer site with cosine at least
`.95`, relative L2 error at most `.25`, sign agreement at least `.90`, and norm
ratio in `[.80, 1.20]`.

Passing establishes prospective lexical extraction plus reusable two-site
composition for this frozen token-to-write path. It does not establish that the
embedding number coordinate is the model's unique native representation, nor
does it establish selective removal of the complete integrated graph; removal
remains a separate intervention.
