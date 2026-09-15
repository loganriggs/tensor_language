# Resume after the September13 instance expiry

The research checkpoint is14:34UTC. Read the [final report](../basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-13_final_compression.md), [main startup guide](../CODEX_RESEARCH_SESSION_STARTUP.md), and [pasteable prompt](../NEXT_CODEX_PROMPT.md). Main research changes were already committed/pushed through `8330ebfd0` before this handoff.

## Restore the checkout and Python

Clone the existing `loganriggs/tensor_language` repository to **`/workspace/tensor_language`**, using the new environment’s normal Git authentication. Do not copy credentials from the expired machine. Fetch the latest branch state rather than checking out only the old research cutoff. Existing runners, bindings and helpers have absolute paths.

Use the instance’s `/venv/main` environment. The recorded working runtime was Python with Torch `2.11.0+cu128`, NumPy `2.5.2`, SciPy `1.18.1`, Transformers `5.16.1`, datasets `5.0.1`, huggingface-hub `1.28.0`, safetensors `0.8.0`, einops `0.8.2` and tiktoken `0.14.0`. The [venv inventory](python_packages_venv_inventory.json) is a version inventory, not a command to blindly replace a new instance’s GPU stack. Check the new image’s agent guide and GPU architecture. The previous GPU was an RTX5090; its working Torch build bundled CUDA12.8. Do not install a host driver in the container.

The companion `/workspace/theseus-bench` repository is also part of the durable project. Restore it from its existing remote if the next task requires it. Its research head at handoff was `e233031`; its uncommitted `registry/priorities.md` is included in the recovery archive.

## Restore the exact public checkpoint

The main model is not in Git. Download the pinned Hugging Face snapshot into the same cache layout:

```bash
/venv/main/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd',
    revision='ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240',
    cache_dir='/workspace/.hf_home/hub',
)
PY
```

Required model path:
`/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin`.

Expected SHA256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`. Run the startup script with `--verify-checkpoint` to verify it. The snapshot config is also required. Tokenizer-loading code may download its own public assets on first use; inspect a chosen runner’s dry run and dependencies before execution.

## Restore the skill and managed runners

```bash
cd /workspace/tensor_language
/venv/main/bin/python research_session_startup.py --verify-checkpoint
/venv/main/bin/python research_session_startup.py --install-skill --install-runners --start-runners
```

Default invocation is read-only. The flags copy the saved skill if absent, install missing Supervisor configs, and start stopped services after dependency checks. Existing different configs/skills are preserved for inspection. Existing running services are not restarted. Supervisor must already be running; this script does not install the operating-system service manager. Both runner wrappers remain the repository’s existing implementations, including their logs and queue rules.

`bqrunner` is the only GPU lane. `bqrunner2` forces CPU-only execution. Before enqueueing: inspect live processes and queues, claim on the board, freeze predictions/error bars/dependency hashes, run the CPU dry run, then use the existing `ops/enqueue.sh` with `EXPECTED_SHA256`. Do not edit bound source or payload after enqueue. Do not replay completed jobs or overwrite old results; successor runs need new identifiers and an explicit reason.

At14:44both services were running and both queues were empty. The last research GPU job, `run_sparse_pair_package_native_v1`, completed14:25:37; later canary execution is housekeeping. There is **no unfinished setting2 fit to recover**.

## Restore the review cadence

The startup script reads the newest `HOURLY_STRATEGIC_REVIEW_*` and `THREE_HOURLY_MATHEMATICAL_REVIEW_*` filenames and prints deadlines. Last reviews:14:28and14:29UTCSeptember13; then due15:28and17:29. On migration, perform overdue reviews at the first safe boundary and set fresh deadlines. Do not manufacture reports for offline hours.

On the restored local workstation, user-level systemd timers launch serialized,
bounded Codex reviews; see `LOCAL_START_2026-09-14.md`. On another machine, restore
that scheduler or have the active Codex session maintain the same clocks. The saved
skill alternates `CIRCUIT` and `WEIGHT_FOLDING` every hour and combines the
three-hour mathematical review with an organization and efficiency audit. Reviews
must inspect real receipts and take a bounded consequence; a timestamp-only template
does not count. Continue checking elapsed time during work. The main goal remains
active; the instance-expiry handoff is a user-directed pause, not scientific completion.

## What is backed up, and what is not

The final package, sparse candidate, primary result receipts, code, reports and the token sources for the96-prefix confirmation are in Git. Additional head17 attention artifacts needed for the next setting were explicitly preserved with this handoff.

`uncommitted_workspace_2026-09-13.tar.gz` is an emergency snapshot of tracked dirty files and selected small untracked project files at handoff, **not** an alternate set of adopted results. It includes peer-owned changes without modifying or committing them over the shared working tree. Consult `workspace_backup_manifest.json` for the complete inclusion/exclusion inventory and hashes. The archive uses `tensor_language/` and `theseus-bench/` prefixes. Inspect it before restoring; extract to a temporary directory and compare, rather than blindly overwriting a newer checkout.

Large historical untracked tensors and raw arrays are not all backed up. The manifest identifies them as omitted; some are failed-fit checkpoints or invalid-evidence caches. Their absence is not evidence of an unrun experiment: preserve the committed verdicts and use original source scripts/provenance if regeneration becomes necessary. Ignored files, model caches, environment secrets and the full operating-system filesystem are not included. The public model checkpoint must be redownloaded. Do not claim the old instance was fully cloned.

The selected archive and Git push are the practical recovery path; the old `/workspace` was not a persistent volume. The final answer records whether the remote push actually succeeded.

A final [setting2 recovery audit](SETTING2_RESTART_AUDIT.json) rebuilt the existing12×1152×128mixed tensor onCPU and verified all13importedproject/source/artifactdependencies against their committedGitblobs. The externalcheckpoint remains the separate download requirement. This is a recovery check, not a new fit orbehavioral result.
