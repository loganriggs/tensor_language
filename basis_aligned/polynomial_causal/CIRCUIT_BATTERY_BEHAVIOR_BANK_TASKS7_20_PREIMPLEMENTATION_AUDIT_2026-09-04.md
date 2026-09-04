# Circuit-battery behavior bank tasks 7–20: pre-implementation audit

Date: 2026-09-04

Scope: CPU-only, model-free review of behavior-bank commit `36c15d7e7`

Verdict: **REPAIR BEFORE IMPLEMENTATION**

## Bottom line

The common FIT/SELECT/TEST/OOD protocol is sound, but only tasks 7–10, 14, 16,
and 17 are close to executable specifications. Tasks 11–13, 19, and 20 do not
yet determine a unique continuation in every arm. Task 15 asks a causal language
model to choose an article from a noun or adjective that is not in its context.
Task 18 is a surface-form duplicate of the existing key/value and equality lookup
tasks. Several purported one-token tasks also become multi-token under the actual
GPT-2 tokenizer.

These are generator-contract problems, not reasons to abandon the behavior bank.
The smallest repair is to keep the generic causal protocol, make every generator
emit an exact authored target and foil, and merge surface variants that test the
same operation. A capability failure after those repairs remains an informative
negative.

## Audit criteria

For each generated group, the implementation should be able to prove before any
model call that:

1. `A1` and `A2` each change the registered answer and change only one declared
   abstract variable. Multiple token edits are acceptable when they are the
   minimal realization of one abstract edit, such as translating an entire
   coherent sequence.
2. `P` retains the exact semantic answer. If its surface answer token changes,
   it is a surface-transfer arm, not an answer-preserving arm.
3. `C` has a unique registered answer and a non-degenerate foil. “The apparent
   rule does not apply” is not enough to determine the next token.
4. The complete prompt-plus-answer is tokenized jointly. The prompt token IDs
   must be an exact prefix of that joint encoding. This is especially important
   for prompts ending in whitespace.
5. One-token outcomes really are one token at the answer boundary. Multi-token
   candidates are scored autoregressively over the complete, prospectively fixed
   span; alternatives must be token-length matched or report length-normalized
   and unnormalized scores separately. Scoring only a common first token is
   forbidden.
6. Correct answer, foil set, semantic coordinates, changed variable, and all
   group members are fixed by the generator before FIT.

## Task-by-task findings and smallest repairs

| Task | Status | Main validity or redundancy problem | Smallest exact repair |
|---|---|---|---|
| 7. Month successor | **Repair; keep as a successor-family surface** | `P` combines capitalization, separators, and prose; capitalization usually changes the output token, so it is not token-level answer preservation. “Explicitly request repetition” changes both prompt grammar and operation. Wraparound is not a valid assumed rule: archived clean accuracy was 100% for sampled months, but wrap behavior was deliberately not established. | Make separate P families and change one surface factor at a time. Keep answer spelling/case fixed in the primary P. Replace C by an otherwise identical three-item constant sequence. Register non-wrap succession as the in-distribution rule; isolate wrap as OOD with no assumed pass. |
| 8. Weekday successor | **Repair; keep as the same successor family** | The repeated-day C is usable only if it specifies at least three repeated observations and a unique same-day answer. Sunday-to-Monday wrap is unsafe: the archived clean assay failed every sampled cyclic wrap. | Use three-item `x,x,x,` constant sequences for C, forbid wrap in FIT/SELECT/TEST, and reserve all wrap rows as a separately reported OOD capability. Keep A1 as a coherent translation and A2 as a fixed-start step change. |
| 9. Alphabetic successor | **Repair; keep as the same successor family** | Changing case changes the answer surface and therefore is not P. Wraparound is convention-dependent. Single letters are boundary-sensitive tokens even though the sampled ` q`, ` r`, and ` s` forms are each one token. | Remove case from P; use only one separator/prose edit per arm with answer case fixed. Treat uppercase as a surface-transfer/OOD family. Freeze no-wrap ranges and jointly check every prompt/answer boundary. |
| 10. Ordinal-word successor | **Repair** | Adjacent versus every-other progression is valid only with enough terms to identify the step. Higher ordinals and digit suffixes are frequently multi-token (` twenty-first` is three GPT-2 tokens; ` 21st` is two). An explicit repeat instruction is poorly matched to the sequence prompt. | Require three observed terms for both steps, use a frozen no-wrap ordinal lexicon, and use a constant ordinal sequence as C. Keep higher and digit-suffix forms under the explicit full-span policy rather than requiring one token. |
| 11. Quote-style closing | **Merge with task 2 after repair** | Natural prose with an unmatched quote does not force the very next token to be a closer. Quote nesting is convention-dependent, and apostrophe/contraction C rows say only what should *not* happen; they do not define the answer. Single quote, double quote, and backtick also mix English quotation, apostrophe, and code grammars. | Make quote style a registered surface family of pending-opener task 2. Use one formal grammar per family and authored strings whose next span is fixed. Define A2 by changing exactly the top unmatched delimiter in a parser-verified stack. Give C an authored non-closer target and foil, or omit that row until such a matched target exists. Do not pool backticks with prose quotes. |
| 12. HTML/XML closing tag | **Block pending a formal target rule** | `<div><span>text</span>` need not close `div` immediately; a sibling may follow. HTML void elements and XML self-closing elements obey different grammars. All shown closing tags start with the same GPT-2 token `</`, so “first distinguishing token” at the first token cannot test tag identity. Typical tags are three-token spans (`</`, tag name, `>`). | Choose either HTML or XML and freeze a parser-supported subset. Generate a complete authored document, cut it immediately before a known closing-tag span, and score the whole equal-length tag span. A1 changes only the pending tag name; A2 changes only verified stack order. C must come from an authored self-closing/void example with a unique next span, not merely “no close.” |
| 13. Markdown/code-fence closing | **Block pending one exact Markdown grammar** | Code can continue after the prompt, so an open fence does not force immediate closure. Same-length fences inside a block are not ordinary nested fences, and the meaning of “escaped fence” depends on the Markdown dialect. GPT-2 encodes ``` as two tokens, four backticks as one, and `~~~` as two, making naive fence comparisons length-confounded. | Freeze a small CommonMark-compatible parser and generate complete authored documents before cutting at a known closer. Use an outer fence strictly longer than any literal inner run. Score the whole closing span and stratify by equal token length; do not compare fence types/lengths in one A arm. C needs a fixed authored next span. |
| 14. Subject–verb agreement | **Repair, then reuse the existing authority** | A2, as written, preserves which noun controls agreement and therefore need not change the answer, violating the common A2 contract. P may accidentally change attractor number. Collective nouns are dialect-dependent. | Reuse the established incongruent-attractor template. A1 flips head-noun number in a prepositional-phrase template; A2 independently flips head-noun number in a frozen relative-clause template. P changes attractor lexical identity while holding its number fixed. Use unequivocal coordinated subjects for C and exclude collective nouns. Keep ` is`/` are` single-token checks. |
| 15. Indefinite article selection | **Fatal as written; repair as an explicit cloze** | At the prediction point `She bought`, the following noun/adjective is in the future. A causal model cannot use its sound to choose `a` versus `an`. This is different from the documented natural-text article-*probability* feature, which predicts that an indefinite article is likely before seeing the noun. Acronym pronunciation and several `h`/`u` words are dialect- or reading-dependent. | Put the controlling word in context, for example `Choose the article before “hour”:` → ` an`, and keep a frozen pronunciation lexicon. A1 changes only the displayed head word's sound class. A2 uses a displayed adjective+noun phrase and changes only the adjective. P changes the later noun while retaining the adjective. Keep ` a`/` an` single-token checks and reserve unambiguous `hour`/`university`-type exceptions for C; exclude acronyms until pronunciations are explicitly registered. |
| 16. Two-number comparison | **Repair before capability gating** | Equal operands collapse answer and foil to the same token, so C has no comparison margin. Prior few-shot greater-of-two accuracy was heavily confounded by copying demonstration answers; accuracy alone was saturated. Multi-digit, negative, and punctuation-adjacent numbers have variable tokenization. | Use no demonstrations, or balance every numeral equally as demo answer and foil. Build balanced unequal-pair groups where every numeral appears equally by answer, foil, side, and relation. Score donor-directed answer-minus-other-operand margin. Replace equal C with a matched positional rule such as `Left operand of x and y`, which has distinct answer/foil and measures generic operand reading/copying. Restrict primary rows to one-token numeral spans; put other lengths under full-span scoring. |
| 17. Positional list retrieval | **Minor repair; best new pilot** | Arbitrarily permuting the full list changes multiple payload positions and recency relations. P may introduce a duplicate answer/foil. Repeated-value C is degenerate unless a distinct distractor remains. | For A2, swap only the queried value with one fixed distractor. For P, replace exactly one unqueried item with a value distinct from every answer and foil. For C, place the same target at two queried indices plus at least one distinct foil, so an index-changing transformation preserves the answer while remaining nontrivial. Keep length, punctuation, and value token lengths matched within a group. |
| 18. Named-field table retrieval | **Merge; do not count as a new behavior** | This is task 5/6 with semantic field-name surfaces: a key is repeated as a query and its associated value is copied. The displayed `Ada`, `Lima`, and `teal` values are multi-token when placed immediately after `=` in GPT-2 tokenization. Prior semantic key/value lookup was chance and literal lookup was weak, so capability cannot be assumed. | Add `named_field` as a surface/OOD family under the existing equality/key-value retrieval operation. Put a space after `=` and use a verified one-token value lexicon, or score full spans. Preserve field-order P, but stratify query-source distance. Report a native-capability negative rather than creating a replacement task if semantic fields remain at chance. |
| 19. List-marker/newline continuation | **Split, then merge its parts** | `-`, `*`, and numbered labels are not one causal variable: bullets copy a marker, while numbered labels also compute a successor. Indentation alone does not determine whether the next line remains nested or returns outward. A line item may be followed by another item or by prose, so neither the prompt nor C defines a unique next span. Newline+marker answers are multi-token and indentation changes token coordinates. | Remove numbered markers into task 3. Treat bullet-marker copying as a surface family of newline/list-state continuation. Generate complete authored lists, cut before a known next marker, and score the entire newline+indent+marker span with a semantic-coordinate map. A1 changes only bullet glyph; A2 changes only a parser-verified nesting transition. Give prose C an authored next span. |
| 20. Sentence-boundary capitalization | **Repair, then reuse the existing boundary authority** | A boundary does not determine which lexical word comes next; the task is a case-pair margin, not ordinary next-token correctness. A newline does not by itself imply a sentence boundary. Abbreviation and initial rules are style- and context-dependent. Most importantly, a prompt ending in whitespace is not tokenization-stable under GPT-2 BPE: encoding the prompt alone ends in a space token, while joint prompt+word encoding can merge that space into the word token. | Generate complete authored sentence pairs and cut immediately before the target word. Encode prompt+candidate jointly and require the prompt's chosen cut to be a common token prefix for lower/capital candidates; otherwise reject the row. A1 changes only boundary punctuation in a grammar-valid pair. A2 contrasts a true paragraph boundary with a verified within-sentence line wrap, not newline presence alone. Use a frozen set of unambiguous decimal/abbreviation controls. Score capitalized-minus-lowercase versions of the same lexical item and label this a case-routing margin. |

## Redundancy map

The bank should keep multiple surfaces without claiming that every surface is a
different behavior or circuit.

- **Ordered successor:** tasks 7–10 are one operation family and overlap task 4.
  Existing month/weekday/alphabet work already found the same dominant component
  set, with family-specific payload codes rather than one movable “successor”
  subspace. The battery should therefore emit one `operation_id=ordered_successor`
  and separate `surface_family` values.
- **Pending closer:** tasks 11–13 overlap task 2. Quote, tag, and fence rows can test
  reuse only after each grammar has a unique authored continuation; they should not
  enter the registry as three circuits merely because their delimiters differ.
- **Address-to-payload retrieval:** task 17 is a useful positional-address contrast.
  Task 18 is already the equality/key-value operation of tasks 5/6 with a semantic
  label surface and should be merged.
- **List continuation:** numbered task-19 rows duplicate task 3. Bullet rows test
  marker/newline state and can remain only as a separate surface after the split.
- **Already-documented behaviors:** tasks 11, 14, and 20 have substantial archived
  capability/circuit evidence. Their first implementation should adapt those row
  authorities to the generic schema, not rediscover sites from scratch.

## Capability-first order

This ordering minimizes new GPU work and maximizes the chance that the first generic
battery records exercise every protocol path.

1. **Task 14** after the small generator repair. Archived incongruent-attractor
   accuracy is 1.00 on 40 rows, so it is the cleanest known non-lexical capability.
2. **Tasks 7/8/9 as one ordered-successor batch.** Existing clean results were month
   100%, weekday 85% with all misses at wrap, and alphabet 65%. This is the best first
   test of shared operation versus family-specific payload, not three discoveries.
3. **Task 11 as a task-2 quote surface.** Quote-style accuracy was 1.00 on 40 archived
   prompts, but nested and apostrophe rows must wait for the formal grammar.
4. **Task 20** using the repaired joint-tokenized case-margin rows and the existing
   sentence-boundary authority.
5. **Task 17**, the cleanest genuinely new generator once its single-swap A2 and
   non-degenerate repeated-value C are installed.
6. **Task 10**, which cheaply tests whether the successor family transfers to ordinal
   vocabulary after token-length stratification.
7. **Task 16**, only after the balanced zero-shot/prior controls. Do not reuse the
   old few-shot headline as a capability gate.
8. **Task 15**, only in repaired explicit-cloze form; this is distinct from the known
   natural-context indefinite-article probability circuit.
9. **Tasks 12, 13, and bullet-only 19** after parser-backed authored-target generators
   exist. Their current prose is not executable.
10. **Task 18 receives no separate run.** It is a named-field surface in the task-5/6
    retrieval family, with capability checked before any site localization.

## Pre-implementation static gate

Before the generic executor can import a task, run a CPU-only exhaustive generator
test over every split seed and require:

- no duplicate prompt across FIT/SELECT/TEST/OOD and no shared `group_id` across
  splits;
- exactly the declared A1/A2/P/C members per group;
- A1/A2 answer changes, P answer equality, and C's declared relation, checked from
  structured fields rather than strings;
- distinct answer and foil spans in every scored row;
- exact prompt-prefix stability under joint tokenization of every candidate;
- equal sequence lengths and semantic coordinates within a physical-interchange
  group, unless an explicit position map is part of the task schema;
- balanced token, answer, foil, position, distance, and surface frequencies within
  each split for comparison/retrieval tasks;
- parser acceptance plus a unique authored next span for quote/tag/fence/list rows;
- a content hash of the full generated authority before the native-capability gate.

Failure is a generator failure, not a model capability result. Passing this gate
licenses only the native-capability assay; it does not license a distinct circuit
claim.

## Repository evidence used

- Behavior definitions and reuse intent: `CIRCUIT_BATTERY_BEHAVIOR_BANK_2026-09-04.md`,
  lines 90–214 and 216–234.
- Archived behavior screen: `basis_aligned/qk_mdl/RESULTS_l0_mdl.md`, §39–§42. It
  records quote-style 1.00, semantic key/value chance, literal key/value 0.567,
  subject–verb agreement 1.00, and the original greater-of-two confound.
- Greater-of-two demo-copy correction:
  `basis_aligned/qk_mdl/qk_gtwo_democtrl.json` and §40, especially the zero-shot
  0.444 static accuracy and demonstration-dependent output profile.
- Successor family dossier:
  `basis_aligned/qk_mdl/algo_tasks/successor/report.md`. It establishes shared
  component-level machinery but family-specific payloads, and documents the weekday
  wrap failures.
- Existing sentence-boundary routing result: `BILIN18_CONNECTION.md`, §644. This is
  evidence to reuse, not permission to accept the new task-20 generator unchanged.

No model, checkpoint tensor, GPU, queue, or outcome namespace was touched in this
audit.
