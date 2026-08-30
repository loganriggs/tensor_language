# Current best understanding at the eight-hour deadline

**Updated:** 2026-08-29 13:03 UTC

## UPDATE: what changed since the earlier explanation

The earlier `CURRENT_PROJECT_EXPLANATION_2026-08-29_0334.md` is now materially stale.
It described the rank-512 stream map and Family F as pending. Both questions have since
been answered, along with the shared-map and predictive-state experiments.

The short update is:

1. A rank-512 map can imitate a site well when it receives the **native** stream, but
   it fails as a standalone recursively composed program. Refitting it on the stream
   produced by its own compressed prefix makes it much worse.
2. Family F found a better set of MLP3 multiplication gates, but the locally refitted
   Down decoder destroys much of that downstream advantage. Family F is a receipt-
   backed negative for a faithful port, not a pending experiment.
3. Output directions really are shared across the 36 sites at tight storage budgets,
   but one global basis, an attention/MLP split, and the tested large shared/private
   hierarchy are all incomplete descriptions.
4. The tested 64-dimensional causal state does not predict finite intervention
   transport well enough to serve as an interface.
5. The only open entry point is now E4: directly test a small named attention circuit
   on copy behavior, matched controls, and collateral loss. Its audited authority is
   frozen, but the actual model transaction has not started because another registered
   GPU job is still running.

This is real pruning. It means the project should stop spending its main budget on
locally low-rank fits whose only success measure is reconstruction.

## What fraction is explained?

There is still no honest single percentage. The settled ledgers are:

| Meaning | Settled value | What it does and does not say |
|---|---:|---|
| Structural implementation | 36/36 sites | We can execute every component; this is not semantic understanding. |
| Certified storage removed | 29,196,288 / 545,904,054 = **5.348245316%** | A real whole-program compression result. |
| Named causal CE | 0.57968 / 5.30682 = **10.923302467%** | The strict causal accounting; most behavior remains unnamed. |
| Unexplained CE in that ledger | **4.72714 nat = 89.076697533%** | The largest remaining quantitative gap. |
| Terminal extraction/removal actions | **0/68** | We do not yet have a receipt-backed practical circuit. |

The important distinction is that our compiler is structurally complete while the
causal and semantic explanation is still small.

## Exact deadline balance sheet

An experimental negative counts as evidence. A plan, test harness, cached row bank,
or unrun runner does not.

| Cell | Status | Evidence and interpretation |
|---|---|---|
| E1.1 recursive stream closure | **measured negative** | Rank-512 closed-stream deficits are `1.08978 / 1.27276 / 1.26133` nat. The native-stream oracle cannot be called a standalone replacement. |
| E1.2 drift localization | **scientifically pruned, not run** | E1.3 directly tested the more important rescue—fit on deployed inputs—and failed by a very large margin. Localization cannot change deployability. |
| E1.3 closed-input refit | **measured negative** | Three iterations give `5.49867 / 5.61939 / 5.59476` nat deficits, close to losing the whole uncovered-token prediction. |
| E2.1 one shared output dictionary | **receipt-backed negative** | Sharing helps at tight equal-storage ranks 64/128, but no rank passes both same-rank and equal-storage CE conditions. |
| E2.2 attention versus MLP dictionaries | **measured negative in the E2 receipt** | Typed rank 481 beats equal-price global rank 494 by only `0.00250 / 0.00237 / 0.00004` nat, below the registered 0.01 margin. |
| E2.3 stable sparse coordinates | **scientifically pruned, not run** | No global projector passed E2.1, so rotating it cannot restore the missing private directions. The separately tested rank-512 hierarchy also failed. |
| E3.1 response-panel rank | **receipt-backed negative** | The 32/64/96-column panels remain full rank, select no stable knee, and fail split-stability gates. |
| E3.2 unseen composition | **receipt-backed negative** | The rank-64 chain has output-KL error `0.4520`; direct transport is `0.4861`, and even the true-response rank-64 projection is `0.2709`, above the `0.25` gate. |
| E3.3 state-variable edit | **scientifically pruned for this state** | Editing a coordinate in a state that failed destination sufficiency would retrospectively pretend the failed locator is an API. |
| E4.1 terminal screen | **open; authority frozen, no outcome** | The audited copy-only attention screen is queued. Fit means and passing tests are prerequisites, not evidence. |
| E4.2 three behavior probes | **open** | The current prospective amendment covers copy only. Capitalization and number formatting have no outcome. |
| E4.3 extraction/removal | **open** | It is conditional on an E4.1 passer. No final/OOD role has been opened. |

Literal tally: **six measured negatives, three scientifically pruned cells, and three
open E4 cells**. Nothing in the deadline audit changes the strict storage, causal-CE,
or terminal-action ledgers.

## Family F: numerical receipt and preserved failure

Family F is complete as a fit experiment.

- V1's publication failure is preserved in
  `block3_consequence_family_f_v1_failure.json`.
- V2 reconstructed the exact V1 programs, reran the frozen 480-row report, and
  published `block3_consequence_family_f_v2_recovery_receipt.json` last.
- The deadline audit independently replayed both result and receipt semantics.
- Result SHA256:
  `18b03ccf3d6710813375bb7e09b1a3c313d5e7790e2ca3c9a9b683fbf91897c5`.
- Receipt-file SHA256:
  `e81673095c7b6202fdec293c6ad34924fb9acb15213d02ba4b203d5ff8c65a5a`.
- Runtime: `75.26` seconds; peak allocated CUDA memory: `4,719,026,176` bytes.
- It opened 480 fit rows, zero validation rows, zero final rows, and used zero ground-
  truth target tokens.

The registered faithful-port gate was summed-write NRMSE at most `0.20`. The local-
decoder programs obtain `0.78860` at 256 gates and `0.70275` at 512 gates. They fail
decisively and therefore do not authorize validation or global ledger credit.

What worked is support selection. At 512 gates, the consequence-selected/local-refit
program has teacher KL `0.08476`, versus `0.10077` for matched random support and
`0.08862` for activation-selected Family A. More strikingly, keeping the selected
gates' **native Down columns** gives KL `0.05772`, despite a worse local NRMSE of
`0.86957`.

This is one of the project's clearest results: a locally better decoder can be a
causally worse decoder. Family F found downstream-relevant support, but its local least-
squares Down refit broke the useful geometry.

## Result versus literal price

### Family F

The native MLP3 uses 4,608 products per token and about 63.71 MB of stored weights.

| Candidate | Products/token | Stored bytes | Compression | Scientific utility |
|---|---:|---:|---:|---|
| Family F K256 | 256 | 3,545,600 | about 94.4% fewer products/bytes | Fails faithful-port NRMSE; no validation. |
| Family F K512 | 512 | 7,086,592 | about 88.9% fewer products/bytes | Better support than controls, but local-refit port fails. |
| K512 with native Down | 512 | 7,086,592 | same literal program price | Best fit KL (`0.05772`), but registered as diagnostic only. |

The price is genuinely attractive. The missing quantity is transfer and editability on
fresh data, not another fit-set reconstruction number.

### Shared output maps

At rank 512, a global dictionary reduces map storage from 42,467,328 to 21,823,488
float32 values. Because covered-token tables dominate the full program, total program
storage falls only 7.73%. That saving costs `0.012--0.014` nat versus the best equal-
storage independent program and supplies no stable semantic coordinates or edit API.

At ranks 64 and 128, global sharing is more promising: it improves CE by
`0.022--0.036` nat over the strongest independent equal-storage allocations. But it
still loses to independent maps at the same rank by `0.038--0.070` nat. This justifies
one bounded tight-budget shared/private test, not another large global-basis program.

The large-budget shared/private hierarchy is not worth its price. A shared direction
costs as much as 18.5 private rank slots; a rank-128 trunk gives up 2,368 private slots
and loses to the all-private endpoint on all three roles.

### Predictive state

The E3 state is cheap in dimension—64 coordinates—but it does not preserve the causal
response needed downstream. A small interface that cannot predict a sealed finite
composition or destination output is not useful compression, regardless of its local
rank.

### Cost-flat table allocation

At 5,419 covered types and exactly 103.1086 million table values, attention/MLP ranks
`128/384` improve whole-program CE over uniform `256/256` by
`0.01874 / 0.01817 / 0.01695` nat. This is a free compiler improvement at fixed
storage, but it does not name a circuit or reduce storage by itself. The concurrently
running half-cost replication has no receipt yet and is not counted here.

## What simplicity metrics have actually earned trust?

- **Literal stored values and products** are useful when paired with held-out whole-
  program CE and causal tests. They produced the certified attention compression and
  make the Family-F native-Down candidate economically interesting.
- **Whole-program CE/KL at matched price** correctly rejects many locally attractive
  fits. It should remain the primary faithfulness currency.
- **Selective causal effect plus collateral loss** is the right currency for extracting
  or removing a behavior. E4 is the first direct test of this promise.
- **Local MSE/NRMSE** is a diagnostic, not a simplicity definition. Family F directly
  shows it can prefer the causally worse decoder.
- **Rank alone** is insufficient. It can compress storage, but its coordinates may be
  gauge-arbitrary, unstable, or compositionally insufficient.
- **Human-readable lexical classes** remain descriptive until they outperform a
  matched-price continuous program or enable a selective edit.

The general validation rule is:

> A simplicity measure is useful only if being simpler lets us predict, compose,
> extract, remove, transport, certify, or execute something better on untouched data.

## Weak branches pruned at the deadline

Do not spend another main experiment on:

- recursively refitting the same rank-512 stream map;
- a single rank-512 global dictionary or the tested rank-128 shared trunk;
- sparse rotation of a global projector that already failed downstream CE;
- the rank-64 pointwise gauge-transport state;
- direct-sum/HOSVD using the fixed projectors that lost to Haar controls;
- SAE/dictionary or tensor fits scored only by local reconstruction;
- another local least-squares Down refit after downstream support selection.

These are not claims that all nonlinear, hierarchical, SAE, or tensor decompositions
are impossible. They prune the tested interfaces and objective functions.

## The next two full experiments

### 1. Finish the E4 copy-circuit transaction and obey its receipt

Run all eight frozen attention candidates on all 192 selection documents plus the 32
reciprocal synthetic pairs. A candidate passes only if simultaneous document-bootstrap
lower bounds are positive for:

1. damage to genuine copy positions;
2. specificity relative to matched negative positions; and
3. the remaining margin under a `0.01`-nat collateral limit.

If none passes, preserve the scientific negative and do not open final/OOD. If one
passes, run the already registered final/OOD extraction-removal sequence for exactly
that candidate. This is the closest path to the first practical terminal circuit.

### 2. Prospectively test the Family-F K512 native-Down grammar

Freeze the 512 selected native product gates and their native Down columns before
opening fresh documents. Test finite positive and negative edits, matched random and
wrong-cross controls, downstream KL/CE, OOD transfer, and removal collateral. Compare
against Family A and an equally priced local-refit arm.

This experiment has high expected return because the fit result is already strong,
the artifact is roughly one ninth of native MLP3, and the hypothesis is sharply
falsifiable: the native decoder either preserves its advantage on untouched finite
edits or it does not.

Only after those two causal experiments should the project spend another full run on
the tight-budget shared/private hierarchy or the selective-risk/telescoping compiler.

## Current blocker

There is no missing data, checkpoint, cache, or software dependency. E4's selection
lifecycle passed 63/63 independent assurance tests, its canonical audit is committed,
and its one-shot authority is frozen. A prior shell invocation failed before importing
the lifecycle because the repository root was absent from Python's module path; no lock,
selection value, model, or output artifact was opened. The corrected command imports
cleanly and is queued behind one active registered GPU process.

Until that process exits, E4.1--E4.3 remain open and the strict ledgers do not move.
