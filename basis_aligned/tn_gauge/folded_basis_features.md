# Folded-basis features: embedding + OV → bilinear → layer-1 query/key

The qualitative decompositions you asked for, all in one place, framed as the pipeline:
the **embedding** (current token) and the **OV output** (attended token's layer-0 value)
feed the **bilinear layer**, whose output feeds the **layer-1 query/key**. Each section
clusters one source by its effect through that path (data-validated: real co-occurring
tokens / real attention), decoded with the GPT-2 tokenizer. Sources: F31 (embedding side),
F32 (OV side), F36 (composed). bilin18.

**The headline contrast:** the *selection* side (embedding → query/key) organizes tokens by
**grammatical function** (syntax); the *content* side (OV value) organizes them by **meaning**
(semantic word families); the **composition** links them into **syntactic dependencies**.

---

## 1. Embedding decomposition — what the current token SELECTS with (→ syntax)
Cluster current tokens by their layer-1 query/key signature (82% of query/key is current-token-
determined, F30). Classes are **grammatical categories**:

- determiners / possessives: ` the ` a ` my ` an ` this ` your ` its ` their ` his`
- prepositions: ` of ` to ` in ` for ` on ` as ` at ` about ` into ` like`
- auxiliary / copula verbs: ` is ` was ` are ` be ` have ` had ` were ` been`
- wh-words / relativizers: ` that ` which ` what ` how ` because ` when`
- sentence-initial pronouns: ` I ` me ` It `I ` In`
- punctuation: `. > : ) ! ..`

So a token's *selection behavior* is set by its part of speech — words in the same
grammatical class attend the same way, regardless of specific meaning.

## 2. OV decomposition — what content the attended token OFFERS (→ semantics)
Cluster the layer-0 OV value table (attended tokens) by its effect re-aggregated through the
REAL block-0 attention into the layer-1 query/key. Classes are **semantic word families**:

- numbers: `1 3 10 8 3 5 2 4 5 6 7 9`
- quantifiers / degree: ` some ` not ` more ` other ` very ` all ` many ` much ` great`
- wh-words / demonstratives: ` that ` what ` this ` how ` where ` which ` who`
- motion verbs: ` going ` went ` goes ` go `go ` heads ` gone`
- survival: `Surv ival ` surviving ` Survival ` survive`
- travel: ` trip ` vacation ` tour ` Trip`
- difficulty: ` hard ` easier ` hardest ` easy`
- development: ` development `Development ` developers ` develop`
- conquest: ` conquer ` conquering ` conquered`
- completeness: ` complete ` fully ` full`

So the *content* the OV moves is organized by **meaning / morphological family** — a different
axis from selection. (This is why OV compresses to coarse ~16–64 classes, F32, while carrying
semantically coherent groups.)

## 3. Composition — current × attended through the bilinear layer (→ dependencies)
Cluster the real (current, attended) pairs by their *joint* layer-1 query/key code. Composed
classes provably beat clustering each side individually (F33/F34), and they decode to
**syntactic dependencies** — a word linked to its grammatical governor/dependent:

- auxiliary/modal verb → subject pronoun: ` had`→` I`, ` can`→` you`, ` couldn`→` I`, `'m`→` I`
- determiner → preposition / copula (noun-phrase attachment): ` a`→` in`, ` a`→` is`, ` a`→` for`, ` a`→` with`
- `of` → head noun (of-PP attachment): ` of`→` method`, ` of`→` analysis`, ` of`→` theory`, ` of`→` value`
- clause-initial word → sentence boundary: ` The`→`.`, `The`→`\n`, ` the`→` when`, ` the`→` if`
- (emergent semantic domains too: a legal cluster ` court`→` trial`, ` defendant`→` a`; biology subwords.)

So the composition is where the two axes meet: a grammatical-function token (current) links to a
specific content/role token (attended) — the dependency. That joint structure is exactly what
individual (marginal) token classes cannot represent.

---

**Summary of the three axes**
| source | decomposed by | axis it reveals |
|---|---|---|
| embedding (current token) | layer-1 query/key signature | grammatical function (syntax) |
| OV value (attended token) | effect through bilinear → query/key | meaning (semantic families) |
| composed pair (current × attended) | joint query/key code | syntactic dependencies |

Files: `qualitative_examples_qk1.md` (1), `crossterm_value_classes.md` (2),
`composed_pair_features_scaled.md` (3).
