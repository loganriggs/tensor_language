# Handoff — the depth ladder and compressibility across the grid

Written 2026-08-08 15:40 UTC for a reader with no memory of the work. Both
tasks are DONE at seed 0 and written up (RESULTS.md FINDING 14 and FINDING 15,
GRID.md, MAILBOX.md). What remains is seed replication, which two detached
chains are producing on their own.

> **There was a concurrent Claude session (`claude -r`, pid 444766) writing to
> this directory at the same time.** It pushed MAILBOX entries at 15:10 and
> 15:30 quoting the same measurements from earlier snapshots of the same JSONs
> (9 cells, before the depth-4 cells landed). The 15:36 entry supersedes both
> and says so. If two sessions are still running, check `pgrep -af claude`
> before starting anything and prefer re-running the two report scripts over
> re-deriving numbers.

## Finished

| thing | where |
|---|---|
| registered predictions P1–P7, written before the first depth-3 training step | `tf_depth_ladder_predictions.json` |
| 18-cell depth ladder, **seed 0 of all six cells trained and interpreted** | `tf_vanilla_d{3,4}_w{64,128,256}_b8192_s0{,_interp3,_routeuse}.json` |
| depth-1/2 cells re-run through `tf_interp3.py` so the ladder is ONE code path | `tf_vanilla_d{1,2}_w*_b8192_s*_interp3.json` |
| depth aggregation + verdicts + figure | `tf_depth_report.py` → `tf_depth_ladder.json`, `tf_depth_ladder_table.md`, `fig_tf_depth_ladder.png` |
| route-USE test (does the newly-opened route carry the algorithm?) | `tf_depth_addendum.py` → `*_routeuse.json` |
| compressibility, 13 of 14 cells at seed 0 | `tf_compress_grid.py` → `tf_vanilla_*_cgrid.json` |
| compressibility aggregation + trend + figure | `tf_cgrid_report.py` → `tf_cgrid_summary.json`, `tf_cgrid_table.md`, `fig_tf_compressibility_vs_size.png` |

## In flight (detached, no babysitting needed)

- `tf_depth_ladder_chain.sh` (log `tf_depth_ladder_chain.log`) — seeds 1 and 2
  of all six depth-3/4 cells, training then `tf_interp3.py` immediately after
  each. Roughly 50 minutes per seed pass. Idempotent: it skips any cell whose
  `.pt` and `_interp3.json` already exist.
- `tf_cgrid_chain.sh` (log `tf_cgrid_chain.log`) — depth-4 width-256 seed 0,
  then the seed-1/2 replicates at the extreme widths. Also idempotent (skips
  any stem with a `_cgrid.json`).

## Exact commands to finish the job

```bash
cd /workspace/tensor_language/basis_aligned/tiny_full_interp
source /venv/main/bin/activate

# 1. are the chains still alive?
pgrep -f -- 'tf_[d]epth_ladder_chain\.sh'; pgrep -f -- 'tf_[c]grid_chain\.sh'
tail -5 tf_depth_ladder_chain.log tf_cgrid_chain.log

# 2. if either died, just re-run it -- both are idempotent
setsid nohup ./tf_depth_ladder_chain.sh > tf_depth_ladder_chain.stdout 2>&1 < /dev/null &
setsid nohup ./tf_cgrid_chain.sh        > tf_cgrid_chain.stdout        2>&1 < /dev/null &

# 3. the route-USE test for any new depth-3/4 checkpoint (seconds per cell)
python tf_depth_addendum.py            # --all-deep is the default when no --stem

# 4. REGENERATE EVERYTHING.  Every table, figure and verdict in FINDING 14 and
#    FINDING 15 is produced by these two scripts from the JSONs -- nothing in
#    RESULTS.md was transcribed by hand except the prose around them.
python tf_depth_report.py              # -> tf_depth_ladder{,_table}.{json,md}, fig_tf_depth_ladder.png
python tf_cgrid_report.py              # -> tf_cgrid_{summary.json,table.md}, fig_tf_compressibility_vs_size.png
```

## What to change in RESULTS.md when the seeds land

Both findings carry an explicit "SEED STATUS" block saying depths 3 and 4 are
seed 0 only and that the claims are provisional. When
`tf_depth_report.py` reports `n_seeds: 3` for the depth-3/4 cells:

1. Replace the two tables in FINDING 14 with the regenerated
   `tf_depth_ladder_table.md` (it prints mean ± sd over whatever seeds exist).
2. Delete the SEED STATUS block and the "provisional on one seed" clauses.
3. The two verdicts to re-check are the ones a seed could move:
   - **depth 4 width 64** clears its induction power floor by only 1.3× at seed
     0. If it drops below floor at two of three seeds, the octave-per-layer
     claim becomes "256 → 128 → 128" and the P1 headline must be rewritten.
   - the **route fraction** (layer-1 attention into layer 2's read as a share of
     the dominant MLP term) is 0.17–0.39 at seed 0; it is five orders of
     magnitude above the depth-2 value so it is unlikely to vanish, but the
     *size* of the effect is a one-seed number.
4. FINDING 15's trend has no seed replicates yet at all. `tf_cgrid_report.py`
   already computes `seed_spread` for every ratio; quote it beside the slope
   once the replicates exist, and if the mean per-cell seed standard deviation
   is larger than about 0.02 the slope (−0.042 per e-fold) needs restating with
   a clustered error.

## The two things most worth doing next

- **The route-use test at three seeds is the strongest single claim here** —
  "at depth 3 the induction circuit becomes an attention→attention circuit" is
  a cleaner statement than anything the six-architecture slice produced, and it
  is currently one seed. It is 3 seconds per cell.
- **Depth 5 at width 256.** The route fraction grows with width at depth 3
  (0.17 → 0.26 → 0.39) and the induction score triples from depth 2 to depth 4.
  One more layer would say whether the attention→attention route keeps growing
  or saturates, and it is ~25 minutes of GPU.
