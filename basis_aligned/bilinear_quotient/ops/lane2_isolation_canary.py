#!/usr/bin/env python
"""lane2_isolation_canary -- proves the CPU-only lane 2 (ops/bqrunner2.sh) is isolated from lane 1 before it carries any
science. OPS canary, not evidence. Runs on lane 2 only.
# BQLANE: cpu
# BQGATE: EXPERIMENT  pred_a_no_cuda pred_b_thread_cap pred_c_lane1_state_untouched pred_d_reduced_priority
"""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "lane2_isolation_canary_results.json"
LANE1 = [ROOT / "queue.txt", ROOT / "runlogs" / "_completed.txt", ROOT / "runlogs" / "runner.log"]
THREAD_CAP = 4
NICE_MIN = 10


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "absent"


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dry_run_passed", "rung": "lane2_isolation_canary"})); return
    before = {str(p): sha(p) for p in LANE1}
    import torch
    x = torch.randn(512, 512)
    for _ in range(20):
        x = (x @ x.T) / 512.0            # a few seconds of real CPU work while lane 1 may be running
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    cuda_avail = bool(torch.cuda.is_available())
    threads = int(torch.get_num_threads())
    nice = int(os.nice(0))
    time.sleep(3)
    after = {str(p): sha(p) for p in LANE1}
    # lane 1 files may legitimately change if lane 1 is running its own job; the isolation claim is that
    # THIS process never writes them -- checked by comparing our own write set: we only write OUT.
    preds = {
        'pred_a_no_cuda': bool(cuda_env == "" and not cuda_avail),
        'pred_b_thread_cap': bool(threads <= THREAD_CAP),
        # fail-closed: lane 1 rewrites queue.txt only at its own job boundaries, so an unchanged sha across this
        # canary's lifetime is evidence lane 2 did not pop or rewrite it (a coincident lane-1 pop can only FAIL this)
        'pred_c_lane1_state_untouched': bool(before[str(LANE1[0])] == after[str(LANE1[0])]),
        'pred_d_reduced_priority': bool(nice >= NICE_MIN),
    }
    out = {"rung": "lane2_isolation_canary", "status": "complete", "preds": preds, "cuda_visible_devices": cuda_env,
           "cuda_available": cuda_avail, "torch_threads": threads, "nice": nice, "lane1_sha_before": before, "lane1_sha_after": after,
           "lane1_changed_during_canary": [k for k in before if before[k] != after[k]], "pid": os.getpid(), "cwd": os.getcwd(),
           "argv0": sys.argv[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sys.path.insert(0, str(ROOT / "ops"))
    from receipt import dump
    dump(out, OUT)
    print(json.dumps(out, indent=1))
    if not all(preds.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
