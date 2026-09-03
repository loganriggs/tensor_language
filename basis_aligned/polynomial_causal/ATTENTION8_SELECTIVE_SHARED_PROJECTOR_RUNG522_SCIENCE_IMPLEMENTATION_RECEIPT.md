# Rung 522 scientific-runner implementation receipt

Timestamp: 2026-09-03 07:39 UTC  
Pre-outcome commit: `2836dac0ae20817dc268f120f8e28be3fedc38a0`  
Scientific model outcome available when written: **no**

## What is frozen

The complete rung-522 entrypoint is now executable. It fits and archives all 103 registered rank-4 frames before
TEST, computes every provisional VALIDATION gate, and either stops with a create-only negative result or writes the
canonical pre-TEST manifest and opens TEST exactly once. The TEST sweep evaluates the frozen real, oracle,
label-permutation, recovery-only, Haar, and geometry-selected all-three objects. Mean-centered removal runs only if
the combined VALIDATION+TEST A/B gates and Prediction C pass.

Rank four is only the matched capacity of each intervention. The claimed object is a downstream-use-defined part of
attention8 that predicts omitted circuit effects, beats task-free and label-randomized controls, transfers to the
reserved fourth circuit, and supports selective removal.

The runner additionally writes a create-only JSON file containing every scalar VALIDATION measurement and response
hash. The pre-TEST manifest binds that file's byte hash and canonical-content hash and checks that its embedded
decision and call ledger exactly equal the separately supplied decision and ledger.

## Frozen bytes

- scientific runner: `b9ff888e808cca1459c469ea15c111a421ebbb0a2d56999c10378099c5e305d0`
- archive/manifest validator: `02680d4912d48d4199b6aaa607d1c77120822217e8e56b40a61d80bddb33dec9`
- runner CPU tests: `de3b9e635f6f5176ce15e3dd7f082ae97c3b875ccb9a32dd39c0f4035112770e`
- archive CPU tests: `a8ef7380dfb1481067183c7d1b255e05e9b8e6b88c7e2c1fbca0ef3ada53130c`
- preflight addendum: `4f75c97dcdce1e652030cb933301c10540aa750d9a78cf5049c15aae48546ca6`

## Checks passed before queueing

- 69 focused runner/archive/validation/sparse-null CPU tests passed.
- The synthetic 200-update fit produced an archive record whose frame geometry, optimizer history, health decision,
  and exact FIT/VALIDATION scheduler payloads were independently re-derived by the archive validator.
- The repository fast checks passed with zero failures.
- Python compilation passed.
- The experiment gate reported `GATE: PASS` with no findings.
- The import-free hash dry run reported the exact registered ceilings: 103 frames, 20,600 forward and 20,600
  backward optimization events, 9,422 inference forwards, and a one-way TEST seal.
- Both managed queue files were empty at 07:39 UTC.

The next authorized action is to enqueue this exact committed runner through `ops/enqueue.sh`. Any byte change to a
registered dependency, any missing arm, any unexpected call bucket, any unhealthy required fit/control, or any
failed provisional gate stops the run before TEST or before the conditional removal stage, as appropriate.
