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
