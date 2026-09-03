# Smoke-test hygiene (Claude, 2026-09-03 15:29 UTC)
- Never run a 16-thread CPU smoke while the runner is executing a CPU probe: OpenMP oversubscription on 16 cores slowed both by
  >10x (15:21-15:29). Cap threads: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python ...` and set `torch.set_num_threads` accordingly,
  or wait for `runlogs/runner.log` to show `exit=` for the current job.
- Smoke on RANDOM tokens but with the REGISTERED tensor shapes (rows are 257 columns: inputs `[:, :256]`, targets `[:, 1:257]`).
  The 15:02 rank-map failure was a shape bug the smoke did not exercise.

## Never `git stash` while the runner is alive
`queue.txt`, `runlogs/_completed.txt` and `runlogs/runner.log` are tracked files the runner reads/writes live. `git stash`
(and `pull --rebase --autostash`) reverts them to HEAD for the duration of the stash; the runner then sees a stale queue and
completed-list — on 2026-09-03 16:02 it re-ran a finished rung and dropped a just-enqueued job. Instead: `git add` the live
runner files into your commit, then `git pull --rebase` (clean tree), then push. If a rebase conflicts on those files, take the
working-tree (live) version.

## Lane 2 — CPU-only runner (prepared 2026-09-03 17:10Z; approved by Codex 16:56Z; NOT YET INSTALLED)

Files: `ops/bqrunner2.sh` (pops `queue2.txt`; runs ONLY scripts carrying the literal header `# BQLANE: cpu`, else
drops them fail-closed; forces `CUDA_VISIBLE_DEVICES=""`; 4 threads; `nice 10`; own state `runlogs/runner2.log`,
`runlogs/_completed2.txt`, `runlogs/<name>.2.log`; never touches lane-1 files), `ops/bqrunner2.conf` (supervisor
program), `ops/lane2_isolation_canary.py` (first job: proves no CUDA, thread cap, priority, lane-1 queue untouched),
and `LANE=2 bash ops/enqueue.sh <abs path>` (same parse/fast-test/gate/dry-run/dedup as lane 1 + header check).

Install (root shell; Claude's session is not permitted to install supervisor services or write queue2.txt):
```
cp ops/bqrunner2.conf /etc/supervisor/conf.d/bqrunner2.conf && supervisorctl reread && supervisorctl update
LANE=2 bash ops/enqueue.sh /workspace/tensor_language/basis_aligned/bilinear_quotient/ops/lane2_isolation_canary.py
# wait for runlogs/_completed2.txt "lane2_isolation_canary exit=0"; read lane2_isolation_canary_results.json
```
No science goes on lane 2 until the canary's four preds are TRUE. Lane 2 is for CPU-by-design probes (R.load_model
on CPU); never for model code that could silently fall back from CUDA.
