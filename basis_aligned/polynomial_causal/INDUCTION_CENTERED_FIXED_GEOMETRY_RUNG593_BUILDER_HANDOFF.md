# R593 builder handoff for independent exact-byte review

**Builder status:** prospective candidate only; not self-approved and not authorized for execution

**Frozen amendment commit:** `53ff70aea`

R593 carries forward R592's rows, calls, centered interventions, scientific scores, target/control gates, bootstrap,
FIT-first split policy, and claim boundary. It changes only the two invalid instrument checks established by the R592
post-execution audit.

## Candidate inputs frozen before adapter finalization

| artifact | SHA-256 |
|---|---|
| producer | `193013a0c0cf1bec19be4843dee751c355d56f69fbf2d761df57baaa86c6024a` |
| model runtime | `768c0ed002f107c7549070a0c162552a0e1825ed3de411ff85987a79a8165777` |
| owner test | `7c573951d8631e1870e6b7d565294223d15739e96d2c44dd24b3a52c840b9a43` |
| fake-runtime/state-machine test | `c8b7422d4cf6a3735cb0298489b648b00c2c64a32aae7b2ecf59706a32973860` |
| deterministic dry run | `a763b8f48541d152c302cd6d31127aa108f1a90abf54e07cc77ff77c224c36a1` |
| prospective R593 amendment | `df0ceebf57818534a9b4ac5de4cd82ca64f2c1228cdfd476e350e62e5707729c` |

The final adapter pins this handoff but the handoff deliberately does not name the adapter hash, avoiding a circular
hash dependency. The final candidate receipt must list the adapter and its test separately.

## Repair implementation

Every endpoint call now carries the raw expected-support mask hash and exact true/false counts. The mask is derived from
the frozen token and semantic A/C coordinates. Per-call evaluation compares every observed bit to the cached authority
mask. Complete phase evidence reconstructs the entire mask independently and checks the frozen record hashes and
censuses. There is no `support.all()` path.

The runtime casts the observed native float32 attention pattern, value vectors, and head projection slice separately to
float64. It then performs three independent masked/full reductions and projections for equality, complement, and full
selected-head output. Remainder is not defined by subtraction. Those three raw and canonical arrays are float64;
factorized terms, logits, interventions, and full-nine-head reconstruction remain float32. The unchanged absolute
decomposition threshold is `1e-5`.

The invalid diagnostic and receipt now bind producer, adapter, runtime, checkpoint, and transitive source provenance
when the managed model boundary was crossed. Normal results also record the adapter hash.

The first different-agent review blocked candidate `6392ebaef` because an invalid call before the first canonical slice
could publish all preallocated canonical files. The repaired store deletes every zero-bound canonical file. For a nonzero
ledger-proven prefix it rewrites the NumPy header in place, truncates the file at the exact axis-0 bound, flushes and fsyncs
the file, then recomputes every retained slice hash against the fsynced ledger before publication. It never allocates a
second data copy, so invalid-prefix preparation adds zero data bytes and cannot exceed the frozen streaming peak. The
invalid diagnostic and receipt explicitly bind the phase, endpoint/directed written bounds, retained filename bounds, and
ledger-record count. The first-call fixture leaves only the raw failing call and a zero-byte ledger; a mid-arm fixture
retains exactly the completed endpoint prefix plus the current raw directed calls.

## Frozen tripwires

- support records: FIT 13,824 = 5,760 true + 8,064 false; SELECT 6,912 = 2,880 true + 4,032 false;
- endpoint support histograms: FIT `{0:288, 1:1440}`, SELECT `{0:144, 1:720}`;
- evidence bytes: FIT 5,501,463,552; SELECT 2,750,731,776; largest live chunk 43,440,640; peak 8,295,635,968;
- capacity: 9,455,639,040 before model and 3,954,175,488 before SELECT, equality accepted and one byte below refused;
- model price: 639 FIT + conditional 322 SELECT = 961 maximum, zero backward/update;
- geometry/output: physical width 30 and complete 50,304-logit vectors;
- split closure: dry run opens no SELECT, FINAL, or OOD data.

Model-free owner and fake-runtime suites pass 24 tests, and the managed-adapter suite adds 7 tests, for 31 total. The
three inherited R592 streaming/storage functional suites add 25 passing tests. They include real authority rows with zero and one supported
roles, true-bit and false-bit flips, all-true rejection, full-phase one-bit corruption, actual-scale float32 inputs to
the independent float64 primitive, a planted `2e-5` error, source inspection forbidding subtraction-defined remainder,
all inherited invalid/hard-abort/receipt-last state transitions, exact evidence pricing, and capacity boundaries.
The numerical fixture now has native-head RMS `27.9266`, matching the amendment's approximate observed scale; the absolute
falsifier remains mathematically independent of RMS and is unchanged at `1e-5`.

The superseded R592 repair-review suite was also invoked: its functional attacks pass, but its deliberate worktree-SHA
assertion fails because it pins the older `521e4c38c` R592 candidate rather than the later executed R592 bytes. No R592
source or frozen review test was changed to mask that expected lineage mismatch.

## Required different-agent review

The reviewer should independently rederive the support hashes from R578/R585 rather than trust this handoff; inspect
the actual runtime contraction graph; verify all six float64 raw/canonical array positions and byte prices; attack
mask/hash/count inconsistencies; plant high-scale and cancellation-heavy numerical examples; ensure invalid provenance
is receipt-bound; rerun inherited R592 streaming, nonfinite, atomic-publication, FIT-first, full-logit, and call-price
attacks; and verify the adapter's immutable closure.

Current R592 invalid evidence remains read-only and materially reduces free disk. Even an approved R593 candidate must
not be queued unless the managed adapter observes at least 9,455,639,040 free bytes. No model, Torch, checkpoint tensor,
CUDA/GPU, queue, or R593 outcome was opened while building this candidate.
