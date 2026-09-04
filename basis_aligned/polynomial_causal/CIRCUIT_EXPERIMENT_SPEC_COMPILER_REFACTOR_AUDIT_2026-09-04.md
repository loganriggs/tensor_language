# Circuit experiment compiler: duplication audit and smallest useful refactor

Date: 2026-09-04 UTC  
Scope: CPU-only audit of the high-quality circuit pipeline from R545 through the approved R592 specifications  
Recommendation: build one small declarative contract compiler; do not build a scientific-gate language

## Short answer

We can save substantial time by generating the parts of an experiment that should be identical in meaning across
circuits: authority membership, split-local support, model-call schedules, evidence schemas, result/receipt envelopes,
terminal precedence, atomic publication, and immutable managed execution. The actual causal computation, intervention,
metrics, controls, and scientific gates should remain ordinary bespoke Python.

The smallest useful vertical slice is a pure-CPU `CircuitExperimentSpec` compiler plus a shared package publisher and
managed entry point. A circuit supplies frozen rows and three pinned pure functions:

1. construct the circuit-specific authority and semantic coordinates;
2. convert retained primitive evidence into scientific scores; and
3. map those scores to the preregistered scientific decision.

The compiler produces the exact row/cell/call/evidence/terminal manifests and the generic validators around those
functions. It does not decide what a selector, payload, cached value, MLP term, or meaningful control is.

This directly supports the project goal: circuits still have to predict held-out behavior, extract a computation,
survive selective removal, and compose or reuse. The refactor removes bookkeeping failures; it does not replace those
interpretability requirements with rank, compression, or generic causal measurements.

## What was audited

I inspected the source and model-free dry-run contracts for R545, R573, R576, R578, R586, R590, the R585 authority and
execution machinery that they informed, `result_contract.py`, the bilin18 model facade, immutable managed adapters,
`derive.py`, the bootstrap playbook, handoff versions 1–7, and all four approved R592 specification documents. I did
not inspect scientific outcome values or the live R592 implementation, and made no change to R590/R592 code, results,
queue state, or explanations.

Line counts below are physical source lines from `wc -l`. “Reusable responsibility” is a conservative review of
top-level function spans, not a claim that the text is byte-for-byte duplicated.

| pipeline unit | production/helper lines | concrete reusable responsibility already embedded in it |
|---|---:|---|
| R545 row builder | 234 | content IDs, split/group closure, uniqueness, finite JSON, row receipt, zero-model envelope; about 50–70 lines |
| R573 localization, semantic mapper, v2 wrapper | 735 | hashes, authority closure, batching, bootstrap mechanics, FIT-first result envelope; about 120–170 lines |
| R576 compiled removal | 480 | the same hashes, authority loading, length batches, bootstrap, phase accounting, result envelope; about 100–140 lines |
| R578 row authority | 736 | 143-line dataset validator plus IDs, receipt, split/group/sequence closure; about 180–220 lines |
| R586 clean replication | 530 | generic result-contract adapter, exact types, provenance, receipt, serialization, dry run; about 260–310 lines |
| R585 manifest + producer + managed adapter | 4,341 | authority/cell/bootstrap compiler, evidence membership and terminal closure, scheduling, serialization, atomic publication/recovery, immutable entry; at least 1,200–1,600 lines |
| R590 producer + managed adapter | 1,699 | provenance snapshot, support census, call manifest, price, dry run, result/receipt, atomic publication/recovery, immutable entry; about 1,000–1,150 lines |
| approved R592 specification lineage | 641 prose lines | 211 lines in the last two amendments specify generic partial-call evidence and nonfinite-mask packaging |

The primary R585 and R590 owner/adapter tests add another 2,219 lines. The full R585 adversarial suite is several
thousand more lines because each repair had to preserve the old attack as a new fixture. Those tests are valuable, but
many should run once against shared machinery rather than be copied into every future circuit.

Across this sample, roughly 3,000–3,600 production/specification lines implement responsibilities that recur across
experiments. Not all can disappear: shared code itself needs about 900–1,200 lines and strong tests. The realistic
reduction after two migrations is about 700–1,100 production lines and 300–500 owner-test lines per complex circuit,
while leaving its scientific kernel intact.

## The repeated work and the bugs it produced

### 1. Authority and panel construction

R545 and R578 each independently implement content IDs, row uniqueness, split-disjoint grouping, family censuses, and
receipts. R585 then independently converts R578 into rows, endpoints, directions, target cells, control cells, coverage
keys, structural identities, control-to-target scale lookups, and bootstrap cells. R590 independently builds another
split/family/condition support census and checks for borrowing, replacement, duplication, and shrinkage.

This repetition caused the class of bugs now recorded as handoff lessons 9, 11, 16, and 23: labels standing in for
token semantics, self-consistent invented IDs, global support standing in for FIT support, and silent changes in cell
membership.

### 2. Call schedules and literal price

R573 and R576 calculate batches and forward counts in their `main` functions. R585 has endpoint and direction
schedulers. R590 spends about 160 lines constructing and validating a 510-call manifest, then uses source-code AST
counts to argue that the manifest reaches the model. Approved R592 has to restate endpoint chunks, five calls per
directed chunk, the unpadded SELECT tail of 16, and all partial-call prefixes.

This repetition produced hidden-call, fixed-shape, padding, and partial-prefix failures. The same compiler can derive:

- R585: 459 FIT + 231 SELECT = 690 maximum;
- R590: 510 possible calls, including 379 unconditional FIT calls and the guarded remainder; and
- R592: 639 FIT + 322 SELECT = 961 maximum, with the final SELECT directed batch exactly 16.

### 3. Evidence, decisions, and invalid terminals

R586 wraps `result_contract.py` with roughly 250–300 lines of exact field, provenance, receipt, and dry-run checks.
R585 grew a much richer evidence validator that joins endpoint tokens, semantic coordinates, direction endpoints,
factor arrays, hook changes, primitive logits, score reports, and bootstrap inputs. R590 again derives its result from
primitive evidence and validates a new result/receipt schema.

The important lesson is not “use JSON Schema.” The recurring requirement is executable reconstruction:

```text
frozen authority + retained primitive evidence
    -> exact instrument failures
    -> exact scientific scores
    -> exact terminal and next step
```

Recent failures came from trusting one of the arrows: saved failure text, a self-consistent row hash, summary metrics,
or a result decision. The compiler should force a single projection function to be rerun during production, final
validation, and independent audit.

### 4. Publication and managed execution

R590 contains about 225 producer lines for same-filesystem staging, receipt-last publication, stale-stage recognition,
conservative recovery, and complete-package validation, plus a 306-line immutable managed adapter. R585 has another
implementation of both. These are infrastructure, not circuit science.

This duplicated area caused repeated failures involving partial packages, unreachable recovery, complete outcomes
mistaken for stale files, hash-then-reopen races, import-time side effects, unpinned transitive dependencies, and dry
runs that opened outcome-bearing authorities. A shared implementation should make those attacks library tests.

## Reusable pieces that should survive

### `ops/result_contract.py` — keep and extend

Its 439 lines already provide strict finite JSON, declared field types, split closure, exact row/group membership,
forward/backward/update envelopes, and provenance hashes. R586 demonstrates that it works. It should become the scalar
result layer of the compiler, not be replaced.

Missing capabilities are array descriptors, foreign-key joins, phase-specific evidence closure, terminal precedence,
and result/receipt/package reconstruction.

### `bilin18_observed_model_facade.py` — keep as the only model boundary

Its 323 lines pin the checkpoint and topology and expose explicit attention/MLP dispatch. This is the correct place to
enforce model structure and forward shapes. Add a named shape policy to the call request; do not let each experiment
toggle a vague `require_production=False` flag.

The compiler must never implement attention or MLP science. It passes one validated `ForwardRequest` to the facade and
receives one `ForwardObservation`.

### R585 manifest — extract the pattern, not the circuit names

`induction_selector_payload_frozen_factor_rung585_manifest.py` is the best existing example of a pure authority
compiler: exact rows/endpoints/directions, cell manifests, deterministic control-scale lookup, SHA-defined bootstrap
draws, and phase accounting. The selector/payload family logic stays in an R585-specific authority builder; canonical
IDs, cell compilation, support checks, and draw grammar move to shared code.

### R590 publisher and immutable adapter — extract nearly unchanged

The same-filesystem stage, fsync, receipt-last moves, marker-bound recovery, pre-import `O_NOFOLLOW` capture, in-memory
compilation of verified bytes, and transitive dependency closure are the strongest versions in the repository. They
should become one tested runtime. Per-experiment adapters should declare dependencies and call that runtime rather than
copy 200–300 lines.

### `ops/derive.py` — keep separate

This 67-line source-splicing tool saves time for the rapid below-block probe series, where a child deliberately differs
from its parent at one body region. It should not generate high-quality circuit experiments: literal string rewriting
does not provide authority joins, evidence reconstruction, split closure, or immutable execution. It solves a
different, useful problem.

## The smallest declarative vertical slice

### Public data types

The initial module should expose these frozen dataclasses. Every instance must serialize to canonical finite JSON.

```python
@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    sha256: str
    kind: Literal["source", "prereg", "authority", "outcome"]
    executable: bool = False
    dryrun_access: bool = False

@dataclass(frozen=True)
class AuthoritySpec:
    builder: CodeRef                 # pinned pure function
    row_id: str
    split: str
    group_keys: tuple[str, ...]
    semantic_fields: tuple[str, ...]
    cell_keys: tuple[str, ...]
    expected_split_rows: Mapping[str, int]
    expected_cell_support: Mapping[str, CellSupport]

@dataclass(frozen=True)
class CallFamilySpec:
    name: str
    phase: str
    source_table: Literal["rows", "endpoints", "directions"]
    arms: tuple[str, ...]
    batch_size: int
    group_by: tuple[str, ...]
    physical_width: int | None
    final_batch: Literal["literal", "forbidden"]
    guard: str
    forward_interface: str
    shape_policy: str

@dataclass(frozen=True)
class ArraySpec:
    name: str
    call_kinds: tuple[str, ...]
    dtype: str
    shape: tuple[DimensionExpr, ...]
    retained: bool
    finite_policy: Literal["always", "final_nonfinite_diagnostic"]
    authority_axis: str | None

@dataclass(frozen=True)
class PredicateSpec:
    predicate_id: str
    phase: str
    priority: int
    evaluator: CodeRef               # pinned pure function over retained evidence
    required_arrays: tuple[str, ...]
    disposition: Literal["diagnostic", "hard_abort"]

@dataclass(frozen=True)
class ScienceProjectionSpec:
    projector: CodeRef               # evidence -> scores/failures
    decision: CodeRef                # scores/failures -> terminal/next_step
    output_types: Mapping[str, str]
    allowed_terminals: tuple[str, ...]

@dataclass(frozen=True)
class CircuitExperimentSpec:
    experiment_id: str
    rung: int
    artifacts: tuple[ArtifactRef, ...]
    phases: tuple[PhaseSpec, ...]
    authority: AuthoritySpec
    calls: tuple[CallFamilySpec, ...]
    arrays: tuple[ArraySpec, ...]
    predicates: tuple[PredicateSpec, ...]
    science: ScienceProjectionSpec
    publication: PublicationSpec
```

`CodeRef` is only a module role plus qualified function name. The referenced module is an executable `ArtifactRef`, so
its bytes are captured and verified before import. A spec may not contain an arbitrary filesystem import or lambda.

### Compiler and runtime interfaces

```python
def compile_experiment(
    spec: CircuitExperimentSpec,
    *,
    authority_inputs: Mapping[str, bytes],
) -> CompiledExperiment:
    """Pure CPU: build and validate authority, support, calls, schemas and prices."""

def validate_call_evidence(
    compiled: CompiledExperiment,
    prefix: Sequence[CallEvidence],
) -> EvidenceAudit:
    """Check exact prefix/order, joins, shapes, hashes, finiteness and predicates."""

def project_result(
    compiled: CompiledExperiment,
    evidence: EvidencePackage,
) -> ResultPackage:
    """Rerun the pinned scientific projection and deterministic terminal choice."""

def stage_and_publish(
    compiled: CompiledExperiment,
    package: ResultPackage,
) -> None:
    """Validate, fsync, atomically publish evidence/result, then receipt last."""

def managed_main(spec_path: Path, environment: Mapping[str, str]) -> NoReturn:
    """Capture full executable closure, run model-free preflight or exact science."""
```

The model-facing loop should have one narrow hook:

```python
class CircuitKernel(Protocol):
    def prepare_call(self, call: CompiledCall, cache: FrozenCache) -> ForwardRequest: ...
    def observe_call(self, call: CompiledCall, output: ForwardObservation) -> PrimitiveEvidence: ...
    def project_science(self, evidence: EvidencePackage) -> Mapping[str, JSONValue]: ...
```

The generic runtime—not the kernel—calls the facade exactly once per `CompiledCall`. `ForwardRequest` contains tokens,
query positions, attention/MLP dispatchers, the named shape policy, and requested retained tensors. This makes literal
call counts real rather than inferred from source AST. The kernel may compute arbitrary circuit-specific tensors, but
it cannot publish files, choose splits, add calls, or construct a terminal.

### Compile-time invariants

The compiler rejects a spec unless all of these hold:

1. every machine name is unique and canonical; display labels do not define identity;
2. every row belongs to exactly one split and group, and every requested split-local cell has its registered number of
   distinct rows without replacement or borrowing;
3. call IDs are a deterministic total order and batching accounts for every source row exactly once per arm;
4. the sum of compiled calls equals the declared maximum price and each phase guard gives an exact possible prefix;
5. every facade call has an explicit compatible shape policy;
6. every diagnostic predicate uses only retained arrays; a predicate depending on an unretained tensor is forced to
   `hard_abort`;
7. predicate priorities are unique and terminals form a total deterministic precedence;
8. all evidence IDs have explicit foreign keys into authority rows, endpoints, directions, calls, arms, and sites;
9. every scientific summary and failure list is regenerated by the pinned projector from primitive evidence;
10. dry-run executable closure contains no `kind="outcome"` artifact, while the real closure is captured before its
    first import; and
11. result, evidence, and receipt paths are distinct, same-filesystem stageable, and unused at preflight.

The R592 nonfinite rule belongs in the generic array validator: for the final failing call only, generate the canonical
one-to-one `{raw_stem}.npy -> nonfinite_masks/{raw_stem}.mask.npy` index and require exact set equality. This prevents
both recent R592 specification bugs from recurring.

## Generated artifacts

Generate data, not large Python files. One thin entry module imports the shared runtime and a circuit kernel.

1. `<experiment>_compiled_contract.json`
   - exact authority/cell/support hashes;
   - complete ordered call manifest and all legal guarded prefixes;
   - array descriptors and authority joins;
   - predicate order/dispositions;
   - terminal closure table;
   - dependency graph and output namespaces.
2. `<experiment>_dryrun.json`
   - model-free regeneration hash;
   - split/cell/row counts;
   - call-kind, arm, shape, and guard counts;
   - minimum/maximum phase prices and zero backward/update claims.
3. `<experiment>_contract_cases.json`
   - canonical positive fixture census and required generic planted-attack IDs. The fixture data remain in tests.
4. Runtime result package
   - primitive evidence plus the compiled-contract hash;
   - regenerated scores, exact failure list, terminal, and next step;
   - mutually bound result/evidence hashes and receipt-last package ID.
5. Managed entry
   - a 10–25 line experiment file declaring the spec and kernel roles; shared code performs capture, dry-run, recovery,
     and immutable dispatch.

Do not generate a bespoke score function, intervention, semantic coordinate mapper, or counterfactual dataset. Those
remain reviewed source artifacts and are hash-pinned by the compiled contract.

## Generic tests versus circuit tests

### Run once against the shared compiler/runtime

- nonfinite JSON and exact scalar/list type attacks;
- missing/extra/duplicate/cross-split/replacement panel rows;
- call deletion, insertion, reorder, wrong arm, wrong batch, unpadded-tail violations, and price mismatch;
- hidden/unregistered forward request and incompatible facade shape policy;
- retained diagnostic predicate with a missing input and unretained predicate not marked hard-abort;
- invented/omitted/duplicated/reordered failure clauses and terminal-precedence mutations;
- wrong evidence foreign keys, primitive-to-summary mutation, and result decision mutation;
- two- and three-array nonfinite masks, wrong set/path/shape/bytes/hash/count/first coordinate, and traversal;
- crashes after each staged write and rename, stale recovery, arbitrary occupied bytes, and complete-outcome refusal;
- dependency tamper before import, import-time side effect, transitive mutable reopen, and hash-then-exec race; and
- a dry-run leaf that attempts to read an outcome artifact.

### Keep per circuit

- token-level meaning of every counterfactual and both physical directions;
- semantic source/query/answer positions;
- exact tensor algebra and independent reconstruction;
- native activity and active controls;
- every scientific metric, sign, denominator, scale, threshold, bootstrap statistic, and opposing prediction;
- the causal evidence ladder and claim boundary; and
- planted scientific nulls where the circuit-specific projector must choose a different terminal.

This split is the guard against weakening science. Generic tests prove that the declared experiment ran and can be
audited. Circuit tests prove that the declared experiment answers the intended interpretability question.

## Migration order

Do not rewrite any completed result or currently frozen R590/R592 byte authority.

1. **Extract without adoption.** Copy `result_contract` primitives, R590 package publication/recovery, and R590
   immutable capture into shared modules with their existing adversarial tests. Require byte-independent behavioral
   parity; no experiment imports them yet.
2. **Compile R578/R585 authority in shadow mode.** Express the R578 rows and R585 authority/cell/bootstrap manifests as
   an `AuthoritySpec`. Require exact equality of all existing ordered IDs, 1,872/936 rows, 1,728/864 endpoints,
   3,744/1,872 directions, 20/32/24/88/124 cell counts, bootstrap IDs/draw hashes, and 459/231 accounting.
3. **Compile R590 dry run in shadow mode.** Without reading its outcome, reproduce the 510-call manifest hash, 379
   unconditional FIT guards, all call-kind/shape counts, phase support hash, and registered prefixes. Do not change the
   approved R590 runner or adapter.
4. **Use one new, not-yet-frozen circuit as the first end-to-end pilot.** Keep its intervention and score projector
   bespoke. Compare agent time, review iterations, per-rung LOC, and attack coverage with R590.
5. **Use the next R592 successor—not current R592—as the array/prefix pilot.** Reproduce 639/322 prices, literal batch
   16, all five call-stop positions, and multi-array nonfinite diagnostics. Do not invalidate current approved hashes.
6. **Adopt only after two different circuits pass independent review.** Then make the shared runtime the default for
   future circuits. Historical scripts remain immutable evidence.

## Expected savings

These are targets to measure, not guarantees:

| unit | current observed scale | target after compiler | expected saving |
|---|---:|---:|---:|
| complex producer + adapter | R590: 1,699 lines | 500–850 bespoke lines + 100–200 spec lines | 650–1,000 per circuit |
| owner + adapter tests | R590: 620 lines | 180–300 circuit tests | 300–440 per circuit |
| very rich producer | R585: 3,426 lines | roughly 1,600–2,200 bespoke lines | 1,200–1,800, if the evidence joins generalize |
| contract review iterations | R585/R590/R592 had repeated infrastructure blocks | one circuit review plus shared-runtime version review | 50–70% fewer infrastructure-only iterations |
| agent build/review time | often several repair cycles over one to two days | same-day CPU candidate for an established experiment shape | about 4–10 agent-hours per complex circuit |

The compiler itself will likely cost 900–1,200 production lines and 600–900 test lines. It should break even after two
complex circuits, not after one. It will not save the time required to invent good counterfactuals, derive a correct
intervention, or interpret a scientific null.

## Kill criteria: stop before this becomes framework work

Stop or shrink the refactor if any of these occur:

1. the shared production implementation exceeds 1,200 lines before it can reproduce R578/R585 authority and the R590
   dry run;
2. a circuit needs more than two escape hatches or more than 250 infrastructure-only spec lines;
3. expressing a scientific gate in the compiler is harder to review than its existing Python function;
4. the compiler cannot reproduce the existing ordered IDs, support hashes, call IDs, shapes, and prices exactly;
5. immutable execution becomes weaker—for example, a verified module is reopened by path or a dry run reaches an
   outcome-bearing artifact;
6. after two pilots, per-circuit production plus owner-test lines fall by less than 30%, or time to independent dry-run
   approval falls by less than 40%;
7. a generic abstraction hides the raw primitive evidence needed to recompute a circuit's metric or invalid reason;
8. one shared-runtime defect can silently misclassify two experiments without both package validators detecting it; or
9. migration pressure suggests editing historical outcomes or approved experiment bytes.

If the call-execution hook proves too heterogeneous, keep the compiler CPU-only and still share authority, schedules,
evidence contracts, packaging, and adapters. That smaller boundary captures most of the savings without pretending all
circuit interventions are alike.

## First implementation ticket

Build only these pieces first:

```text
ops/circuit_experiment_spec.py       dataclasses, canonical compiler, call/support/evidence/terminal manifests
ops/circuit_artifact_package.py      strict result projection, staging, receipt-last publication, recovery
ops/circuit_managed_entry.py         pre-import closure capture and immutable dispatch
ops/test_circuit_experiment_spec.py  shared attacks listed above
```

Acceptance is exact shadow parity on the R578/R585 authority manifests and R590 dry run, plus a new synthetic fixture
covering R592's five-arm partial prefix and three simultaneous nonfinite arrays. No model call is needed. Only after
that narrow ticket passes independent review should a new scientific circuit use it.
