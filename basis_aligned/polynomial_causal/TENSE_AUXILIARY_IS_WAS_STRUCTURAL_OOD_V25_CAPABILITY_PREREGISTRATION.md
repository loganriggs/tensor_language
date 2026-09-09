# `is`/`was` structural-OOD v25 native-capability gate

## Why v25 exists

V23 varied temporal phrases and 16 reporter nouns but retained essentially one parse:
`temporal phrase, the NOUN -> copula`. V24 changed cue lexicon but retained the same defect and also
failed its unrelated-control capability gate. V25 is a new authority, not a filtered repair. It
tests eight target structures and prompts from 8 to 18 tokens long:

1. fronted era phrase;
2. time cue after the grammatical subject;
3. relative clause inside the subject;
4. subordinate-clause prefix;
5. reported-source frame;
6. long coordinated prefix;
7. postnominal time cue;
8. embedded predicate nominal.

There are also four structure-matched same-tense paraphrase controls and four structure-matched
harbor/canyon controls. Sixteen history-disjoint compound profession phrases supply lexical
variation. Each of the 16 construction IDs has four rows, two in each direction. Every base/donor
pair is token-position aligned; no rows are selected after native outcomes.

## Capability-only execution

Run the native model on all 64 base and all 64 donor prompts in exactly two forwards. For every
construction × direction × side cell (two rows), require at least 1/2 correct. For every construction
(four rows), require at least 3/4 rows jointly correct on both base and donor. This combination
ensures every retained construction has native support in both directions without row filtering.

Registered predictions:

A. The hash-bound authority, history disjointness, eight target structures, eight matched controls,
   pairwise token alignment, 8--18 token length range, exact population, and exact price pass.
B. Every one of the eight target constructions passes native capability.
C. Every one of the eight control constructions passes native capability.
D. No causal outcome is accessed and price is exactly two forwards / 128 sequence evaluations.

Any A/D failure is invalid. Any B/C failure is an honest native-capability null and opens no circuit
outcome. Only an all-pass receipt can license a separately preregistered v25 four-head causal test.

