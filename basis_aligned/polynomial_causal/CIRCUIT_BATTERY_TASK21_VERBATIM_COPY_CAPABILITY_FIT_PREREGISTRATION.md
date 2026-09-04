# Task 21 verbatim-copy capability-only FIT preregistration

**Frozen prospectively:** 2026-09-04 06:17 UTC, after the task-number, canonical-ID, balance, and semantic-validator
repairs. **Execution status:** CPU authority and compiler construction only;
no model, checkpoint, GPU, queue, enqueue, outcome, or localization access is authorized by this document.

## Goal and why this task is next

The adoption track needs behaviors with clean counterfactual data before it searches for a circuit. Task 17's strict
positional-list capability screen validly returned `hard_abort`; its thresholds and result namespace remain unchanged.
Task 21 therefore asks only whether bilin18 can continue a trailing verbatim repetition, for example

```text
Repeat exactly: storm soft dawn dawn dawn -> dawn
```

Task choice used the old diagnostic battery only as a prioritization clue. In
`circuit_battery_v2_results.json` (SHA-256
`5924b2549d285175c80fbf7c8fc95a8a2fa06020acc1827bc472ddea69d9ec93`), `verbatim_repeat.copy` had descriptive native
accuracy `1.0` on each reported split and mean answer margin `6.6782`. That battery is invalid as phased circuit
evidence, so none of its rows, component choices, thresholds, or later-split outcomes is reused here. The new task-21
rows and bars were fixed without evaluating the model on them.

This is deliberately a strict **local previous-token repetition** screen: every registered answer is the final prompt
token. It is not an induction or remote-retrieval task. A2 leaves an older conflicting target visible, but both A1 and
A2 still put the donor answer in the last prompt position. A future circuit study must separately control the residual
stream/embedding route before attributing success to attention-mediated copying.

The immediate candidates below were compared before choosing; they do not exhaust the behavior bank:

1. **Verbatim copy** has the strongest old capability clue, a large open vocabulary, and two distinct answer-changing
   edits. Its C control can change repeat length while preserving copied identity, which is a real active comparison
   between *what token* is copied and *how much repetition evidence* is present. It does not pretend that identical
   occurrences have a uniquely identifiable source.
2. **Induction copy-successor** offers a richer selector-versus-payload decomposition, but existing R593 work still has
   a transaction-precision instrument problem and would mix this authority milestone with a harder intervention.
3. **Weekday/roman successor** had weaker diagnostic capability and a small fixed vocabulary that makes strict
   four-way lexical isolation awkward. It is a useful later grammar task, not the cleanest next capability screen.
4. **Repaired task-14 subject–verb agreement** is the strongest alternative in the preimplementation audit and is
   scientifically richer. Strict local copy is chosen first because its generator has no grammar-label ambiguity, its
   21-token roles can be exactly balanced, and it provides a low-risk validation anchor for the newly repaired phased
   pipeline. This ordering does not demote subject–verb agreement; task 14 should follow if the pipeline behaves as
   specified.

Thus verbatim copy is selected. The C control is meaningful enough to proceed: it changes a trailing run from three to
four copies on FIT/SELECT/TEST (four to five on OOD) with the answer held fixed. Future circuit work can ask whether a
candidate tracks token identity, repeat-evidence strength, or both.

## Exact linked panel

Every group begins from the same base sequence: distinct nonrepeat fillers followed by one repeated target. Its four
donors have equal token length and make these exact edits:

- **A1 — replace entire repeat:** keep every filler fixed and replace every token in the trailing repeat with another
  token. The answer changes to the donor token.
- **A2 — replace latest two-token run:** keep fillers and the earlier target occurrence fixed, but replace the last two
  tokens with a repeated alternative. The answer changes to that most recent repeated token. This differs structurally
  from A1 because the old repeated token remains visible.
- **P — irrelevant filler:** replace the first nonrepeat filler with a novel token while leaving the trailing repeat
  identical. The answer stays fixed.
- **C — repeat-strength control:** replace the filler immediately before the run with another copy of the same target.
  The target identity and answer stay fixed, but the trailing repeat is one occurrence longer. This is registered as
  an active control, not as a no-change input.

A1/A2 isolate two ways of changing copied content. P tests invariance to an unrelated earlier token. C separates copied
identity from repetition strength. All four rows literally share the same base prompt and group ID.

## Frozen authority and phase separation

SHA-256-derived phase permutations and a Latin-cycle assignment produce 21 groups per phase and exactly 84 rows per
phase. Each phase has 21 tokens, and every token appears exactly once as target, alternative, novel control, and each
filler-position role. This balances answer, alternative, control, surface-position, and registered-foil exposure so
token difficulty cannot masquerade as a transform effect. FIT, SELECT, TEST, and OOD use disjoint
interleaved token vocabularies, disjoint prompts, and separate immutable JSON artifacts. FIT/SELECT/TEST use two
fillers plus a three-token repeat under `Repeat exactly:`. OOD changes both surface and length: three fillers plus a
four-token repeat under `Echo the final run:`.

- Full 336-row semantic authority SHA-256:
  `191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b`.
- Task/semantic-validator source SHA-256:
  `bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2`.
- FIT record SHA-256 `c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8`;
  file SHA-256 `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94`.
- SELECT record SHA-256 `c437ebcf8fa4c00e43be26063ee985dacd767e76c41bbf0263ef9bde52638139`;
  file SHA-256 `151e50755c9570cf411e614111fe9c5857d5ea13aab7fb7e53d6ce493b8a1f67`.
- TEST record SHA-256 `d780a7e0993422ed0d52aafacb42c7eb3433503d1b01bf1197bffcdd8b8c6d45`;
  file SHA-256 `dc3340c18d7c2efaa460fecf1e0134bc07532f939d1b424016977ecab810c155`.
- OOD record SHA-256 `2ee14e4547291888608f484c43d4b656f65bc5e709625cafbc5cac4de9ab640b`;
  file SHA-256 `bf338c34ff0ffe17a56c6c8cb8f3e7c74fcf4c0549c4f9933065bbe8cca16c38`.

The FIT compiler may capture only the FIT artifact. It must neither generate nor include SELECT, TEST, or OOD bytes.
Those separate files are prospective future authorities and cannot open without the exact preceding phase receipts.

## Exact continuation and metric

For prompt string $p$ and answer string $a$, the validator jointly tokenizes $p+a$. It requires the tokens of $p$ to
be an exact prefix and $a$ to add exactly one GPT-2 token. Base and donor prompts have identical token counts, so a
future interchange position has the same absolute index on both sides. Every FIT prompt has exactly 8 tokens; the
scored continuation position is index 8, immediately after the final prompt token.

For every row and side $s$, let $y_s$ be its answer token and let $F_s$ be all distinct word-token candidates appearing
in either linked base or donor sequence, excluding $y_s$. Retain only

$$
z_{\mathrm{answer}}(i,s)=z_{i,-1,y_s},
\qquad
z_{\mathrm{foil}}(i,s)=\max_{f\in F_s}z_{i,-1,f}.
$$

Strict correctness is $z_{\mathrm{answer}}-z_{\mathrm{foil}}>0$. Full logits, activations, gradients, component names,
reader/writer candidates, and localization fields are neither requested nor retained.

## Opposing capability predictions

There are eight registered cells: `{base, donor} x {A1, A2, P, C}`, each with 21 rows. Thus the `0.85` cell bar means
at least 18 of 21 strict successes.

- **Capability prediction:** base-wide and donor-wide strict accuracies are each at least `0.90`; every cell accuracy is
  at least `0.85`; and both side-wide mean answer-minus-maximum-foil margins are strictly positive.
- **Capability failure:** either side-wide accuracy is below `0.90`, any cell accuracy is below `0.85`, or either
  side-wide mean margin is nonpositive.

These are exact logical complements. The unseen C rows are not exempted or given a weaker posthoc bar.

## Exact FIT calls and evidence price

The compiler must emit, in order, four base calls and four donor calls. Each call contains 21 explicit FIT rows of
length 8; duplicate base sequences across A1/A2/P/C remain separate row-side evaluations. Each call retains one
contiguous `float32[21]` answer-logit array and one contiguous `float32[21]` maximum-foil-logit array.

- 8 native forward calls;
- 168 row-side evaluations;
- 0 backward calls and 0 model updates; and
- $8\times2\times21\times4=1{,}344$ raw numeric evidence bytes.

Call JSON, array headers, result framing, and receipts are metadata rather than learned state. Any authority, source,
call, row order, token position, target/foil set, array, coverage, or price mismatch is an invalid instrument and
produces no scientific terminal.

## Stop and continuation rules

A capability failure publishes only a complete `hard_abort` package with every scientific projection field null. A
pass licenses only a new, separately frozen FIT-only localization preregistration. Neither result identifies a circuit
or opens SELECT, TEST, or OOD. This preregistration does not authorize a producer, model-facing adapter, checkpoint,
GPU, queue, enqueue, execution, or result namespace; each needs the same independent build/review sequence used after
task 17.
