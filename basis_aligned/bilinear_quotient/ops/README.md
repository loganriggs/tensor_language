# `ops/` — how work actually runs in `bilinear_quotient`

There are ~2,000 files here. Almost all are **one-off rung scripts** (193 of them with a companion
`test_<name>.py`). This file indexes the small set that is **infrastructure**, so neither agent has to
rediscover it by reading the ledger.

Everything below was read from the scripts themselves. Each one carries its own `WHY` comment naming the
measured failure that caused it to exist — those comments are the authoritative detail; this is the map.

---

## The pipeline, end to end

```
write prereg (polynomial_causal/*.md)   →  freeze its sha into the script
        ↓
python ops/gate.py <script>             →  static checks; must print GATE: PASS
BQLIB_DRYRUN=1 python <script>          →  dependencies exist ON DISK, no GPU touched
        ↓
bash ops/enqueue.sh /abs/path/script.py →  parse + fast-test + gate + dry-run + dedup, then queue.txt
        ↓
bqrunner (supervisor service)           →  pops queue.txt, runs, appends runlogs/runner.log
        ↓
<script>_results.json                   →  the receipt: preds, nulls, bars, summary, price, hashes
        ↓
ledger section in BILIN18_CONNECTION.md →  canonical `Price:` / `Results:` lines
        ↓
python ops/audit_ledger_prices.py       →  every cited receipt must resolve and match
```

**Never run GPU work directly while the runner is busy.** Enqueue it. `ops/gpu_free.sh` exits 0 only when no
compute process is resident, so it is safe to chain.

---

## Infrastructure

### Queueing and running
| file | what it does |
|---|---|
| `bqrunner.sh` / `.conf` | the supervisor service that pops `queue.txt` (lane 1, GPU). **Runner-owned — do not edit.** |
| `bqrunner2.sh` / `.conf` | lane 2, the **CPU-only** lane. A script must carry `# BQLANE: cpu` and lands in `queue2.txt`. |
| `enqueue.sh` | the only supported way in. Runs parse, fast tests, `gate.py`, `BQLIB_DRYRUN`, and a **dedup guard** (added after two symmetric double-enqueue races). `LANE=2` for the CPU lane; `FORCE=1` overrides dedup. |
| `lane_depth.sh` | is a lane about to go idle? Exit 1 = actionable. Enforces **≥ 2 queued experiments**, because measured over 48 h, long idles were the single largest loss bucket (36% of the span). |
| `arm_waiter.sh` | canary-filtered landing waiter; reads the current last non-canary `_completed.txt` line as its own sentinel, so arming needs no copy-paste. **Invoke by absolute path** — relative invocation from a drifted cwd gave two silent 127s. |
| `gpu_free.sh` | prints real GPU occupancy and exits nonzero if anything is resident. Replaced a habit that printed "(empty=free)" unconditionally. |

### Writing results up
| file | what it does |
|---|---|
| `next_section.sh` | prints the next free ledger `§` number. Board prose claims still override. |
| `stamp.sh` | prints the prereg stamp line **from the box clock**. Hand-written stamps drifted 1–3 min ahead five times in one hour, and the prereg hash covers the stamp — a late fix means a re-hash means a re-derived script. |
| `audit_quick.sh` | one-command receipt audit: verdict pass + science-leaf grep. |
| `receipt_runtimes.sh` | recent receipts with `runtime_s`, newest first. Uses absolute timestamps because this box's `find` is `bfs`, which rejects the relative `-72 minutes` form — and a `2>/dev/null` once swallowed that error for two hours. |
| `audit_ledger_prices.py` | maps each ledger `§` to its receipt via the `Results:` line and checks the `Price:` line matches. Also reports **UNAUDITABLE** sections and refuses two sections citing one receipt. |
| `push_both.sh` | pushes and **confirms by re-reading the remote**. Replaces a helper that ended in `… \| tail -1 && echo OK` — a pipeline's status is the last command's, so it tested `tail` and printed OK on a failed push. |

### Repo hygiene
| file | what it does |
|---|---|
| `repo_health.py` | **read-only, one command for "is this tree fit".** Six checks, each present because something went wrong unnoticed: ledger receipts resolve; the orphan scan is a fixed point; no half-committed renames (bare `A `/`D ` lines — `git mv` + a one-sided pathspec commit leaves the other half staged); no untracked path over 100 MB **on disk**; ops-lane tests green; lane 1 queue depth ≥ 2. Exit 1 if any fail. |
| `repo_orphans.py` | **read-only.** Finds root artefacts nothing references — no ledger §, board entry, backlog note, prereg, script, wrapper or queue file, with self-mentions excluded. Splits `dead` from `ran-but-uncited` using `runlogs/runner.log`. Includes `archive/**` in its corpus so archiving cannot cascade. |
| `archive_orphans.py` | moves what `repo_orphans` found into `archive/<date>/<tier>/`, re-scanning at move time. `git mv` for tracked files, `MANIFEST.json` per batch. **Deliberately a separate command** — deciding what is dead and acting on it should never be one step. |
| `restore.sh` | rebuilds this box's runtime after a Vast **recycle/destroy**. `${WORKSPACE}` is *not* volume-backed here (`vast-capabilities \| jq '.instance.workspace_is_volume'` → false), so a recycle wipes everything outside git. |

### Shared libraries (by how many scripts import them)
| module | imports | role |
|---|---|---|
| `bilin18_joint_removal` | 545 | the model facade: `fwd`, `orth`, `m`, `FW`, `DEV`. The core everything builds on. |
| `circuit_dictionary` | 291 | `classify`, `COMPS`, `CLS` — the token-class dictionary. |
| `census_lib` | 264 | census/aggregation helpers. |
| `receipt` | 173 | receipt construction. |
| `mlp_in_situ_usage_rank_map_probe` | 171 | in-situ MLP usage/rank mapping. |
| `bilin18_observed_model_facade` | 152 | observed-model facade. |
| `bqlib` | 108 | shared runner-side helpers (`BQLIB_DRYRUN` lives here). |
| `scoring` / `target_token_classes` / `result_contract` / `interchange` | 12 / 10 / 3 / 3 | scoring, class definitions, the result contract, interchange interventions. |

Frontier-lane helpers: `frontier_evalarms.py` (fit-once/eval-many; `factorial_arms`, `subset_arms`,
`check_eval_only`), `frontier_fitcache.py` (`stack_key`/`save_stack`/`load_stack(key, device=)`/`verify_stack`),
`armsweep.py`.

Other docs here: `README_SMOKE_TESTS.md`, `RESULT_CONTRACT_USAGE.md`, `NOISE_FLOOR_SCHEDULE.md`,
`EFFICIENCY_LOG.md`, `wake_prompt.md`.

---

## Conventions that are enforced, not optional

- **`# BQGATE:` header** listing every `pred_*` key. `gate.py` checks it, and flags hazards such as two scripts
  writing one receipt.
- **Preregistration is hash-frozen.** The script stores the prereg's sha256 and refuses to run if it changed.
  Write the stamp with `stamp.sh`, or substitute `$(date -u …)` into a **quoted** heredoc — an unquoted `<<EOF`
  command-substitutes backticks and silently deletes text.
- **Canonical price lines**, or the auditor cannot read them:
  `Price: N GPU forwards, X GPU-seconds` and `Results: <file>.json`. One receipt per section.
- **Two controls for every new transform path**, not one:
  1. **identity** — apply at the setting that must change nothing; require `0.0000`.
  2. **route agreement** — apply at a setting already on record, through the new path *and* a trusted path;
     require them to match.
  An identity control alone is *not* sufficient: at its identity setting, a faithful knob and a knob that does
  nothing are the same measurement. A front-split knob was silently wiped by a later `_apply_*` that resets its
  target from a snapshot, every arm read exactly `0.0000`, and only route agreement caught it.
- **Ordering:** a later `_apply_*` that resets from a snapshot when its argument is `None` will wipe an earlier
  one. Put a new knob last, or compose explicitly.
- **Sign convention:** frontier L2 is **CE added above the real model — lower is better**. State it inline in
  every directional claim.
- **Measure disk usage with `st_blocks`, not `st_size`.** `rung592_invalid_evidence/` holds sparse `.npy`
  files: 4.9 GB apparent, **11 MB on disk**. A health check that cries wolf is worse than none.

## Before you push

```
python ops/repo_health.py
```

## Lane boundaries

`ops/circuit_*.py`, `ops/target_token_classes.py`, the `rung5xx` scripts and their tests, `circuits/*.json`,
`ops/determinism_fingerprint_history.jsonl`, `ops/bqrunner*.sh`, `queue.txt` and `runlogs/runner.log` are owned
by the other lane or by the runner. Read them; do not edit them. Raise changes on `AGENT_BOARD.md` instead.

## Naming: hypotheses, not arms

The four transform families of a fast screen — **A1, A2, P, C** — are **hypotheses**, not "arms". Each is a
distinct claim the screen tests at once:

| | the hypothesis it tests |
|---|---|
| **A1** | the site carries the causal variable |
| **A2** | it carries the same variable in a *different construction*, so it is state and not surface |
| **P** | it leaves an answer-preserving edit alone |
| **C** | it leaves an unrelated behaviour alone |

A site is selective only if all four hold. Use "the C hypothesis", "the P hypothesis" in receipts, board
notes and commit messages.

**Known lesson (2026-09-05):** the *choice of stimulus family* filling a hypothesis can flip a verdict on its
own. Three verdicts from this lane were overturned that way — two nulls on C, one asymmetry on P — with the
other three hypotheses byte-identical. Before believing a verdict that hinges on one hypothesis, re-run with
a different family for it. It costs ~11 s. Do not turn this into a standing four-way sweep on every screen:
vary the hypothesis the conclusion actually rests on.

Compare those focused reruns with `circuit_fast_screen_profile.py`. It selects candidate attention/MLP/head sites
using A1/A2 target transfer only, reports residual boundaries separately as ceilings, and retains each named P/C
response instead of allowing a related control to erase evidence for a shared target carrier. It also reports raw P
margin movement because the historical normalized P score can change when an A2 family changes its target scale.

```bash
python ops/circuit_fast_screen_profile.py \
  --member original=circuits/fast_screens/<original>_result.json \
  --member alternate_P=circuits/fast_screens/<alternate>_result.json
```
