# Local restart, 14 September 2026 UTC

## Live restoration update, 02:45 UTC

Supersedes the initial missing-environment observations below. The pinned model
is downloaded and SHA256 verified. Both Supervisor lanes are RUNNING under
`systemctl --user status bilin18-runners.service`. The first managed GPU canary
completed successfully with `ALL: GREEN`; its numerical fingerprint differs from
the old machine, so old/new bit-level numerical equality must not be assumed.

The hourly and three-hour active Codex reviews are enabled as user systemd timers.
Their first bounded reviews have been started and serialize on one file lock.
The initial workspace-write sandbox attempts could not launch commands (host
uid-map restriction); the corrected jobs use this session's full-access execution
mode. User lingering is enabled, so services remain available after logout while
the machine is on. The broken historical reminder cron was removed, with a backup
at `/home/loganriggs/.local/share/bilin18/crontab.before-20260914`.

Use `bash session_recovery/local_namespace.sh /venv/main/bin/python ...` for
historical absolute-path research scripts. Background services set
`BILIN18_PATH_BACKEND=proot` because systemd's host context rejects user namespaces.
The dedicated environment reuses the installed tensor-similarity site-packages
through a `.pth` file and adds Supervisor/tiktoken locally. Control runners with
`/home/loganriggs/.local/share/bilin18/bin/supervisorctl status`.
Service definitions live under `/home/loganriggs/.config/systemd/user/bilin18-*`;
logs under `/home/loganriggs/.local/share/bilin18/logs/`.

Check the live schedules with `systemctl --user list-timers 'bilin18-*'`.
Stop only the scheduled reviews with
`systemctl --user stop bilin18-hourly-review.timer bilin18-mathematical-review.timer`;
an already active review service must be stopped separately if cancellation is
intended. Stop the queue runners with `systemctl --user stop bilin18-runners.service`.
Services and timers are enabled at user-manager startup and user lingering is on.
The machine must remain powered on; these are local services, not hosted jobs.

### First managed research job, 02:48 UTC

Lane2 completed `run_native_retained_error_v1` with exit 0 in 0.59 seconds of
CPU calculation. All three registered instrument criteria passed across four
32-row panels using the actual weights and synthetic Gaussian raw-state corners.
Full nine-term error-Gram/direct replay differed by at most 1.97e-16 relative;
folded/native-factor replay by at most 1.14e-15. Diagonal-only energy exceeded
complete joint energy by 1.16–12.66%. The frozen sparse operator's conditional
relative mixed error was 8.47–9.08%. These are finite synthetic-measure diagnostics,
not native-text results, optimized fits or evidence of improvement. The complete
raw-state measure remains to be justified before fitting. Primary receipt:
`../basis_aligned/polynomial_causal/NATIVE_RETAINED_ERROR_V1_RESULT.json`.

Initial enqueue failed its static gate because predicate keys were expressed as
`dict` keywords and only two predicates were declared. Before any execution, the
existing producer-sum replay was registered as the third measured criterion and
literal predicate keys were added. The second enqueue passed; no failed research
run was hidden or force-requeued.

## Initial observations, before restoration

Updated both local checkouts to fetched `origin/main`: tensor_language
`3f30132a6`, theseus-bench `e233031`. The tensor_language checkout was switched
from `codex-local-simplicity-audit` to `main`; that old branch remains available.
Previously ignored local research artifacts remain on disk.

The current instructions are `CODEX_RESEARCH_SESSION_STARTUP.md` and
`NEXT_CODEX_PROMPT.md`. Next research: setting2, head17.2 → MLP17 → selected
outputs, using a weights-only objective composed with the retained three
attention contractions. Preserve the sparse baseline's narrow conditional result;
do not repeat the completed independent Gaussian marginal correction or claim
that compression establishes OOD prediction, selective removal or reuse.

The default startup check ran with system Python. Portable sparse-even package
hashes pass; both queue files are empty. `/workspace/tensor_language`,
`/venv/main`, Supervisor and the pinned checkpoint are absent here. System Python
lacks most research dependencies. An existing local environment imports Torch
2.10.0+cu128, NumPy 2.3.5 and SciPy 1.17.0 successfully:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/loganriggs/Coding/tensor-similarity/.venv/bin/python <script>
```

No packages or services were installed. The old
`ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_ARTIFACT.pt` cache is absent. Review deadlines
reported by startup are overdue; full strategic/mathematical continuation reviews
remain pending before selecting a native-weight fit.

## First CPU control executed

`../basis_aligned/polynomial_causal/retained_contraction_error_control_v1.py`
uses the existing normalized projected-port code with synthetic shared raw-state
corners, both QK factors, inherited first-layer values and all three contractions.
Let their head writes be `a_k`, the shared background `z`, the downstream mixed
operator error `E`, and the supplied denominator `d`. Form
`e_k = E(z, a_k) / d`. The complete conditional squared-error objective is
`sum_kl mean(<e_k, e_l>)`, not just its three diagonal terms.

Producer replay is exact; the full 3-by-3 Gram reproduces direct composed error
to relative error 2.50e-16. In this synthetic control, joint energy is 0.222067,
diagonal-only energy 0.246265. A separate exact-cancellation witness passes.
This validates an objective identity and exposes the consequence of dropping
cross terms. It is not a native-weight result or evidence that a particular
compression will improve. Final RMS is supplied and softcaps are outside scope.

Primary receipt:
`../basis_aligned/polynomial_causal/RETAINED_CONTRACTION_ERROR_CONTROL_V1_RESULT.json`.
The script refuses to overwrite it. To recheck without writing, import `run()`.

Next implementation dependency is the pinned checkpoint from the existing recovery
guide. With it available, replace synthetic matrices with native producers and
define the weights-only raw-state/corner measure explicitly, preserving shared
backgrounds, all nine cross moments and normalization. Validate any stochastic
objective's integration error before using it to select a fit. Native cached text
must remain validation data, not a weights-only discovery objective.
