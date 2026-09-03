# Rung 523 optimizer-repair implementation receipt

**Frozen:** 2026-09-03 08:30 UTC  
**Pre-outcome implementation commit:** `c6f4587ea`  
**Rung-523 model outcome available:** no

**Pre-outcome test extension:** 2026-09-03 08:35 UTC. A complete synthetic 200-update execution test was added after
the first receipt commit; the scientific runner bytes did not change. The test and check counts below include it.

## Implemented computation

The runner implements the three prospective cells frozen in the rung-523 preregistration:

1. row-specific normalization with Adam learning rate `.003`;
2. fixed FIT target/map normalization with learning rate `.03`; and
3. fixed FIT target/map normalization with learning rate `.003`.

Each cell fits the same 15 real leave-one-target-out frames for 200 updates. The old row-specific/`.03` cell is read
from the immutable rung-522 archive and receives only a common VALIDATION rescore. The fixed scale for target `t` and
donor map `m` is the mean squared complete-attention8 CE response across every eligible FIT member position for that
target/map. All candidate initial and final frames are compared with that same FIT-derived scale on the unchanged
VALIDATION health batch.

The executable split guard accepts only `FIT` and `VALIDATION` and raises on `TEST`. No omitted target is evaluated.
The result explicitly records `test_opened=false`, `omitted_targets_evaluated=false`, and
`scientific_a_through_d_scored=false`.

## Exact model-call ledger

- prospective optimization: `3 arms * 15 fits * 200 updates = 9,000` forwards and `9,000` backwards;
- native FIT/VALIDATION capture: 131 inference forwards;
- independent native replay: 131;
- self-donor no-op checks: 2;
- complete-attention8 FIT response cache: 95;
- complete-attention8 VALIDATION response: 36;
- common health evaluations: 120 = 30 for the archived baseline plus 90 for the three prospective cells;
- total inference-only forwards: exactly 515;
- removal forwards: 0.

The runner refuses a changed bucket or total and uses create-only JSON and tensor artifacts.

## Frozen bytes

- runner: `06ac12a85e0feafad9c4eead98a06d279461f393c330310f64b3a31c71bf7426`
- runner tests: `f5421735cc9ce6a2ed233fc2fc067e1bab8e1e21e70955b752296ac1a9d6a69b`
- pure repair math: `0d16b27cdf107efcf40f425bdc1e81350b07d3367db83eeded61a49d676e39e1`
- pure math tests: `097e33c52e80e8854035d9b89b5c27c2e1d50894c6d6bb4c4755be86aa0eec68`
- preregistration: `930a751ff6b7f6c69ae6765b569aa31172b5b5aea334ed1f639d17111861e035`
- rung-522 frame archive: `2b8d3709714903890c4ae935a07da7284ac3253b7b2242d055023b33adeca2bb`

The runner validates the preregistration, pure math, rung-522 runner/archive/scheduler/state modules, and baseline
archive hashes before importing the model stack.

## Checks passed before queueing

- eight focused CPU tests pass, including a complete synthetic 200-update fit through the prospective candidate
  path with exact forward/backward/health-call accounting;
- Python compilation passes;
- the import-free dry run reports the exact 9,000/9,000 optimization calls, 515 inference calls, and inaccessible
  TEST split;
- the repository experiment gate passes with three explicit registered predictions;
- the repository fast checks pass with zero failures; and
- source diff checks pass.

Rung 522 was still live in the managed GPU lane when this receipt was first written. Rung 523 was therefore initially
left off the queue.

**Pre-outcome scheduling correction, 2026-09-03 08:57 UTC:** the managed runner is the serialization boundary, so a
queue entry cannot execute concurrently with rung 522. Rung 523 may be appended while rung 522 remains live, avoiding
an idle-GPU gap after it exits. This changes no code, data, optimizer setting, threshold, or experimental order:
rung 523 still begins only after the rung-522 process is terminal, and its executable guard still permits only FIT
and VALIDATION. The independent terminal auditor will verify rung 522's JSON as soon as it is created; R523 does not
consume that JSON and cannot open TEST.
