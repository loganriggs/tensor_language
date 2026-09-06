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

## What each gating metric is computed from

Read out of `circuit_fast_screen_kernel.py`, not inferred. Every gate is scored on **one hypothesis' own
records** — no gate mixes hypotheses:

| metric | gate | computed from |
|---|---|---|
| `a1` / `a2` `mean_absolute_effect`, `direction_fraction` | `minimum_target_family_recovery` 0.5, `minimum_target_direction_fraction` 0.8 | the A1 / A2 records respectively |
| `target_recovery` | reported, not gated | `(a1.mean_effect + a2.mean_effect) / 2` — the only metric combining hypotheses |
| `p_invariance_effect` | `maximum_p_invariance_effect` 0.2 | **P records alone** |
| `c_absolute_recovery` | `maximum_c_absolute_recovery` 0.35 | **C records alone** |
| capability accuracy | `minimum_*_capability_accuracy` | each family's own cells |

**Why this table exists.** On 2026-09-05 this lane spent a screen testing whether P-invariance was inflated by
the P hypothesis sharing an answer vocabulary with A1. It cannot be: `p_invariance_effect = p.mean_absolute_effect`
is P's records only, so the hypothesis was impossible by construction. Five lines of the kernel would have
shown it before anything was authored.

**The rule that follows:** *run* an instrument to find defects (every tool bug this lane found surfaced that
way); *read* the definition to find what a number depends on. They are different questions and only the first
is answered by experiment.

### `screen_preflight.py` — is this run already in the ledger?

`circuit_fast_screen_ledger.append_entry` refuses an execution whose key already appears:

    (candidate_id, prior_art_sha256, spec_sha256, authority_sha256,
     max_forward_calls, max_example_evaluations, max_evidence_bytes)

**None of those seven fields comes from the screen's output.** They are fixed by the runner's
PROTOCOL literals and by compiling the frozen rows — all CPU. But the check runs at *publish*
time, after the science, so a duplicate costs a whole screen plus the round trip to work out why.

    python ops/screen_preflight.py --runner run_circuit_fast_screen_<name>   # exit 1 if it would be refused
    python ops/screen_preflight.py --all

Reports the full 7-field key when the candidate exposes `compile_plan` (task14 engine), and a
3-field partial key otherwise — labelled as partial, because a partial match can only *suspect*
a duplicate while a partial mismatch *proves* there is none. `dev_*` probes are skipped: they
never publish to this ledger.

**Case it came from.** 2026-09-05T05:02Z: `task14_select_cross_noun` ran to completion and was
then refused against ledger entry 25, identical in all seven fields. The fix varied one literal
(`prior_art_sha256`). Validated against that exact history — the pre-fix runner on the ledger as
of 05:02 reports DUPLICATE against entry 25; the post-fix runner reports CLEAR, and it published.
Across all 22 runners it matches ground truth: every published runner predicts DUPLICATE, the one
not yet in the ledger predicts CLEAR.

**Fixing a refusal** means varying a key field — normally registering a distinguishing prior-art
receipt. Do not vary one to dodge the check: two entries differing only in a digest are two claims
of distinct evidence, and the ledger is what makes that claim.

## Two design invariants that keep costing rework

Both learned the hard way, each more than once, and each caught only by running `build_rows()`:

**P must not vary the final input token.** The invariance edit has to change something the model
reads but not the token being patched, because `matched_final_input_token` compares base and donor
of the same row. In a long design the subject is safely mid-prompt and swapping it is fine; in a
SHORT design the subject often IS the final token, and P must vary an earlier slot instead — the
season, the place, the adverb. Hit on `narrative_tense.short_cue` and again on
`aspectual_anchor`.

**A2 needs its own matched suffix whenever it ends on a different slot from A1.** `matched_suffix`
is per row, not per candidate. If A1 ends on an adverb and A2 ends on a participle, passing A1's
suffix to A2 fails `matched_long_final_suffix`. Hit on `polarity_state` (participle) and again on
`preposition_selection` (second adverb table).

Neither is caught by review; both are caught in one second by calling `build_rows()` before
writing a runner. Do that first.

## The one-sided capability failure: a repairable stimulus defect, not a model limit

When a screen returns `native_behavior_incapable` and the failure is **strictly one-sided** — one
answer side at or near 1.00 across both constructions, the other well below the bar — that is the
signature of a broken collocation in the weak side's frame, not a limit of the model or of the
behaviour class. Twice now, swapping only the weak side's verbs has converted the null into a
`selective_causal_site` with the strong side untouched:

    finiteness_selection   v1 finite 1.00 / nonfinite 0.50   ->  decided->refused, offered->declined  -> selective
    dative_alternation     v1 recipient 1.00 / benefactive 0.12 -> bought->reserved, ordered->prepared -> selective

Contrast a genuine limit, which is symmetric: `pronoun_antecedent.gender_reference` sat at chance
on BOTH sides with margin 0.00, and no redesign changes that.

**How to use it without fooling yourself.** A one-sided failure licenses ONE repair of the weak
side, with a stop rule registered in the prior-art receipt before running: if a second verb pair
chosen for collocational strength still fails, record the behaviour as unscreenable under that
design and stop. What makes this legitimate rather than tuning-until-it-passes is that the defect
is diagnosed from the asymmetry first, and the repair leaves the passing side untouched.
