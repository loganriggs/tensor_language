# Circuit battery behavior bank — 2026-09-04

## Purpose

This is the CPU-only behavior specification for a reusable circuit battery. It replaces the pattern of writing a new
dataset harness for every circuit. It does not itself claim that the model can perform every behavior, or that twenty
behaviors imply twenty distinct circuits. Shared internal machinery across several behaviors is a desired result.

Each task is a next-token decision generated from a declarative template. Before any internal intervention, the shared
tool must verify tokenization and native capability. A task that fails the capability gate stops there and remains an
honest negative row; it is not repaired by changing its prompts after internal outcomes are inspected.

## Common task contract

Every task entry supplies:

- an operation name and prompt constructor;
- at least two independently generated answer-changing transformations, `A1` and `A2`;
- at least one answer-preserving transformation, `P`;
- one active control, `C`, with a declared expected effect rather than an assumed zero effect;
- disjoint FIT, SELECT, TEST, and OOD generator seeds;
- the correct answer, a task-relevant foil set, and generator checks proving the intended semantic relation;
- joint prompt-plus-candidate tokenization: the prompt encoding must remain an exact prefix, followed by exactly one
  answer token, or the task must declare an explicit full-span scoring policy;
- a `group_id` shared by related A1/A2/P/C rows so uncertainty is resampled by generated example, not row endpoint.

The protocol first measures native accuracy and donor-directed logit margin. FIT may choose sites; SELECT only confirms
that frozen choice; TEST and OOD remain physically unopened until the prior split passes. The expected-effect vector over
A1/A2/P/C is part of the high-level causal model. For example, a component shared with copying may be expected to move the
copy control rather than remain inert.

## Twenty behavior entries

The first six reuse existing authorities. Entries 7–20 are the requested expansion bank. Concrete examples illustrate
the generator; the actual code should sample vocabularies and surface forms.

### 1. Induction: selector and copied payload

- Prompt: a token pair occurs and its first token appears again, such as `... dax blue ... dax` → ` blue`.
- `A1`: change which earlier occurrence matches the final selector while keeping candidate payloads fixed.
- `A2`: change the payload after the matching earlier occurrence while keeping the selector fixed.
- `P`: change an irrelevant occurrence or filler without changing the matching source.
- `C`: use repeated or conflicting matches whose declared result tests the tie-breaking rule.
- OOD: longer gaps, multiple predecessors, code-like separators, and unseen token classes.

### 2. Pending opener to required closer

- Prompt: nested text with one unfinished delimiter, such as `([text]` → `)`.
- `A1`: change the type of the pending opener.
- `A2`: change nesting order so a different opener is currently pending.
- `P`: edit a delimiter pair that is already completed.
- `C`: replace punctuation or content without changing the delimiter stack.
- OOD: deeper nesting, quote/bracket mixtures, and code-like surfaces.

### 3. Numbered-list index successor

- Prompt: `21. red\n22. blue\n` → `23`.
- `A1`: shift every visible list label while retaining item text.
- `A2`: change the list step or insert a preceding label so the required next label changes.
- `P`: change item words or layout while retaining the label sequence.
- `C`: use repeated labels, where copying the visible label rather than adding one is correct.
- OOD: longer labels, different punctuation, and longer lists.

### 4. Numeric-sequence continuation

- Prompt: `14, 17, 20,` → `23`.
- `A1`: change the starting value while fixing the step.
- `A2`: change the step while fixing the start and length.
- `P`: change separators or surrounding prose while preserving the sequence.
- `C`: use a constant sequence or explicit copy rule.
- OOD: negative values, word-number surfaces, longer sequences, and unseen steps.

### 5. Explicit successor-pointer lookup

- Prompt: `A→K; B→M; C→R; B→` → `M`.
- `A1`: change the queried key.
- `A2`: change the value paired with the queried key while keeping the key fixed.
- `P`: permute or alter nonqueried rows.
- `C`: use an identity mapping, making the key itself the correct value.
- OOD: more rows, reordered rows, and alternate separators.

### 6. Equality-based source selection

- Prompt: several labeled payloads followed by a query label, such as `dax:red; mip:blue; dax:` → `red`.
- `A1`: change the final query label while retaining source rows.
- `A2`: change the payload attached to the matching source.
- `P`: alter a nonmatching label or payload.
- `C`: duplicate the query label with a registered nearest/first/last-match rule.
- OOD: longer ranges, code identifiers, and multiple matches.

### 7. Month successor

- Prompt: `March, April,` → `May`.
- `A1`: shift both shown months to a different consecutive pair.
- `A2`: change the registered step from one month to two months using three demonstrations.
- `P`: change capitalization, separators, or a surrounding sentence.
- `C`: explicitly request repetition rather than succession.
- OOD: year wraparound, abbreviations, and non-English-independent numeric month surfaces if tokenizer-valid.

### 8. Weekday successor

- Prompt: `Tuesday, Wednesday,` → `Thursday`.
- `A1`: shift the consecutive weekday pair.
- `A2`: demonstrate a two-day step instead of a one-day step.
- `P`: alter formatting or surrounding event text.
- `C`: a repeated-day schedule whose next value remains unchanged.
- OOD: Sunday/Monday wraparound, abbreviations, and reordered surface templates.

### 9. Alphabetic successor

- Prompt: `q, r,` → `s`.
- `A1`: shift the letters while preserving step one.
- `A2`: change the step with at least three demonstrations.
- `P`: change case or separators without changing the abstract sequence, when answers remain tokenizer-matched.
- `C`: constant-letter sequence.
- OOD: wraparound, uppercase, and longer gaps.

### 10. Ordinal-word successor

- Prompt: `first, second,` → `third`.
- `A1`: shift the ordinal subsequence.
- `A2`: switch between adjacent and every-other ordinal progression.
- `P`: alter surrounding prose and separators.
- `C`: explicitly repeat the last ordinal.
- OOD: higher ordinals and digit-suffix surfaces such as `21st` when single-token valid.

### 11. Quote-style closing

- Prompt: prose ending inside one unmatched quote style, with the next token being the matching quote.
- `A1`: change the pending quote type between single, double, or backtick where tokenization is unambiguous.
- `A2`: change nesting so the inner versus outer quote must close.
- `P`: edit text inside the quote or an already closed quote pair.
- `C`: apostrophe/contraction examples where no quote-closing action is required.
- OOD: longer spans and code/string contexts.

### 12. HTML/XML closing tag

- Prompt: `<div><span>text</span>` → `</div>` under a multi-token answer policy, or score the first distinguishing token.
- `A1`: change the unmatched outer tag.
- `A2`: change nesting order so a different tag is pending.
- `P`: change text or a completed inner element.
- `C`: self-closing and void elements whose registered continuation does not close the apparent tag.
- OOD: unseen tag names, attributes, and deeper nesting.

### 13. Markdown/code-fence closing

- Prompt: an opened fenced block ending after code, with the next token beginning the matching fence.
- `A1`: change fence type or length when tokenizer-valid.
- `A2`: add a nested literal example that changes which fence remains open.
- `P`: edit code contents or language label.
- `C`: inline backticks and escaped fences that should not close the block.
- OOD: longer code, unseen language labels, and four-backtick outer fences.

### 14. Subject–verb number agreement

- Prompt: `The key near the cabinets` → `is` versus `are`.
- `A1`: change the grammatical number of the subject while retaining distractor nouns.
- `A2`: change a relative-clause construction while preserving which noun controls agreement.
- `P`: change an attractor noun that does not control the verb.
- `C`: coordinated subjects or collective nouns with a separately declared agreement rule.
- OOD: longer attractor chains and unseen nouns.

### 15. Indefinite article selection

- Prompt: `She bought` followed by a noun phrase requiring `a` or `an`.
- `A1`: change the following word's initial sound class.
- `A2`: use adjective+noun phrases and change the adjective that immediately controls the article.
- `P`: change later noun content while retaining the controlling initial sound.
- `C`: orthography/phonology conflicts such as `hour` and `university`, with the expected answer declared lexically.
- OOD: unseen words, acronyms, and capitalization.

### 16. Two-number comparison

- Prompt: `Larger of 17 and 24:` → `24`.
- `A1`: swap which operand is larger without changing their positions.
- `A2`: change the requested relation between larger and smaller.
- `P`: swap operand order while preserving the answer.
- `C`: equal operands, where copying either operand is correct.
- OOD: negatives, more digits, and near-equal pairs.

### 17. Positional list retrieval

- Prompt: `Items: red, blue, green. Item 2:` → `blue`.
- `A1`: change the queried index while retaining the list.
- `A2`: swap only the queried value with one fixed distractor while retaining the queried index.
- `P`: replace exactly one unqueried item with a novel value, while preserving the answer.
- `C`: put the same target at two queried positions, retain at least one distinct foil, and change the query between
  those positions. The index changes but the answer does not; this tests an index-sensitive component without forcing
  a behavioral change.
- OOD: six-item lists instead of four-item lists, using a fourth disjoint payload vocabulary. Word ordinals and reverse
  ordering remain future surface families rather than being mixed into this first OOD test.

The first executable adapter is `ops/circuit_battery_task17.py`. It constructs 24 generated groups in each of FIT,
SELECT, TEST, and OOD. Every group contains exactly one A1/A2/P/C row; randomness is derived from SHA-256 rather than
Python's process-randomized `hash()`; all four split vocabularies are disjoint; every base/donor prompt has the same
token length; and every answer is checked at its actual joint continuation boundary. Passing these CPU checks licenses
only a native-capability experiment, not a circuit claim.

### 18. Named-field table retrieval

- Prompt: `name=Ada; city=Lima; color=teal; city=` → `Lima`.
- `A1`: change the queried field.
- `A2`: change the value stored in the queried field.
- `P`: reorder fields or alter an unrelated field.
- `C`: duplicate values across fields, reducing the need for field selection.
- OOD: more fields, JSON-like formatting, and unseen field names.

### 19. List-marker and newline continuation

- Prompt: several bullet or numbered lines ending after an item, where the next token starts the next marker.
- `A1`: change the marker family (`-`, `*`, or a numbered label) while holding content fixed.
- `A2`: change indentation level so the next marker is nested versus outer.
- `P`: change item text while retaining layout state.
- `C`: prose containing marker characters that should continue as prose rather than a list.
- OOD: mixed markers, deeper indentation, and longer items.

### 20. Sentence-boundary capitalization

- Prompt: a completed sentence followed by whitespace, scoring a matched lowercase/capitalized next-word pair.
- `A1`: change terminal punctuation between a true sentence boundary and a comma-like continuation.
- `A2`: change newline/paragraph state while retaining the candidate word.
- `P`: alter earlier sentence content without changing boundary status.
- `C`: abbreviations, initials, and decimal points that resemble boundaries but should not trigger capitalization.
- OOD: quotations after punctuation, multiple newlines, and unseen lexical candidates.

## Automatic triage and reuse analysis

The battery should run cheap gates in order:

1. native capability and generator validity;
2. FIT writer/site localization;
3. frozen SELECT interchange, necessity, and expected-control effects;
4. downstream reader path split for successful writers;
5. TEST/OOD confirmation only for unchanged claims.

At every stage it emits the same typed record, including failures. Across tasks, construct a component-by-behavior
response table from held-out donor-directed effects. Components are grouped only when their complete expected-effect
profiles and interchange relations agree on held-out data. This makes reuse a measured conclusion: month, weekday,
alphabet, and ordinal succession might share a successor operation, while different source readers feed it; HTML,
quotes, brackets, and fences might share a stack-like pending-opener state; list and table retrieval might share an
equality/addressing operation. Those are opposing hypotheses, not assumptions built into the task names.

The twenty entries are therefore a throughput target for tested behaviors. The count of distinct circuit variables is
whatever the downstream-operational grouping supports.
