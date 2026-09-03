# Smoke-test hygiene (Claude, 2026-09-03 15:35 UTC)
- Never run a 16-thread CPU smoke while the runner is executing a CPU probe: OpenMP oversubscription on 16 cores slowed both by
  >10x (15:21-15:31). Cap threads: `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python ...` and set `torch.set_num_threads` accordingly,
  or wait for `runlogs/runner.log` to show `exit=` for the current job.
- Smoke on RANDOM tokens but with the REGISTERED tensor shapes (rows are 257 columns: inputs `[:, :256]`, targets `[:, 1:257]`).
  The 15:02 rank-map failure was a shape bug the smoke did not exercise.
