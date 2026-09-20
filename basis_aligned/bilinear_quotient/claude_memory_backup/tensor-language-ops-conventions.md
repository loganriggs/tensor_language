---
name: tensor-language-ops-conventions
description: "How GPU work runs in tensor_language/basis_aligned/bilinear_quotient — managed runner, enqueue.sh, BQGATE headers, dryrun, board protocol, git identity, and the fresh-instance restore steps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3a5c1d2d-901e-4c52-8edc-21792b8663e5
  modified: 2026-09-17T21:06:56.194Z
---

- GPU work goes only through `bash ops/enqueue.sh /abs/path/ops/<script>.py` (runs parse, fast
  tests, gate.py, `BQLIB_DRYRUN=1` dry run, dedup) → `bqrunner` supervisor service pops
  `queue.txt`; logs `runlogs/<script>.log`. Script needs `# BQGATE: EXPERIMENT pred_...` header,
  a dryrun branch on `BQLIB_DRYRUN`/`BQLIB_NO_MODEL`, and preregistered predictions/bars/price in
  the docstring. Model facade: `ops/circuit_fast_screen_producer.Bilin18TorchBackend` (exact
  manual forward; pre-`c_proj` head slices are 128-d).
- Board `AGENT_BOARD.md` is append-only, `### <UTC> — <agent>: ...`; use `date -u` for stamps
  (I once hand-typed wrong ones). Commit + push everything (workspace is not a volume); git
  identity via `-c user.name=loganriggs -c user.email=logan.smith.5@gmail.com`.
- Fresh instance restore: `uv pip install torch --index-url .../cu128` separately from the other
  packages, `snapshot_download` of `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd` rev
  ed914654 into `/workspace/.hf_home/hub`, then `research_session_startup.py --verify-checkpoint`
  and `--install-skill --install-runners --start-runners`.
- Lesson (2026-09-17): whole-write zeroing conflates norm with direction — always run the
  equal-norm random-direction null; and random coordinate pieces of a cue-defined delta are
  individually "selective", so per-piece selectivity does not certify a module partition.

**How to apply:** reuse `ops/aspectual_dod_lib.py` (fresh rows, capture, midpoint/random/
subtract removal modes) rather than re-deriving. See [[claude-circuit-lane-2026-09-17]].

- The queue gate (`ops/gate.py`) reads prediction keys from literal dict keys / `dict(pred_a=...)` in the runner file, not from the header or docstring. Runners that delegate scoring to `dod_battery.run` must declare a literal `PREDICTIONS = {...}` dict (v71 pattern), or the enqueue is refused with "expected at least 3 pred_* keys".

- Battery runners that set `L.READERS` / `L.UNRELATED` at module level (v55, v97, v104…) change the readers for EVERY runner that imports them, even just for word lists (v119/v120 got will−would instead of was−were). Set readers explicitly in each runner and import word pools from `dod_lexicon` (AGENT_POOL / OBJECT_POOL), not from other runners.

- Bilinear MLP units: a writer fold of each FACTOR (L·x, R·x) names the writers but not the sign of their effect on the product (Lx)(Rx) — v177 found MLP-6 unit 3230 *damps* the male detector 3152 although it carried 82% of each factor's contrast (v173). Fold the product into writer-pair terms (v16-style, per unit) before reading a sign, and confirm with an edit.

- **Mass ≠ carriage in product folds (v187–v190, 2026-09-18).** Pair-term shares of a bilinear unit's contrast attribute product mass; a writer whose write is constant across the contrast can hold a large share. Always add the carrier split (Δ own factor × mean partner total; exact, sums to 1) — runner pattern `run_pronoun_number_dod_unit829_carrier_split_v188.py`. Row-dependent readers (product gradients) confound value vs reader change: split three ways (v187).

- **Carrier share = linear term; quote the in-place leave-out (v225, 2026-09-18).** For a bilinear unit, a writer's carrier share (Δ own factor × mean partner) overshoots the finite edit 2–4×; the exact in-place leave-out D = u(x̂) − u(x̂ − c) from the native trace predicts the unit edit to <1 point ONLY when the edited units and the read unit are in adjacent blocks (v225, MLP 5→6); with MLPs in between it overshoots 2–4× (v226, MLP 6→8: intervening blocks respond). Report D next to every carrier share as the no-response ceiling; the edit still decides.

- **Bilinear detectors: never pool Δh × a mean gradient (v238, 2026-09-18).** The detector's gradient along a writer can have opposite signs on the two row classes; a common-mode shift of a relay then widens the contrast on both sides. Split responses by class (G_P·dh_P and G_S·dh_S) before interpreting a 'compensation'. Also: rms_norm makes detectors scale-invariant — radial writes do nothing (v236); cast tensors to float before JSON (v237).

- **Verify the write before the board claims it (18 Sep 18:16).** A document rewrite failed on a `%` in a `%d`-formatted string and the board entry describing the rewrite was committed anyway; a correction entry followed. Rule: in one command, assert the change landed (grep/count) BEFORE `dod_record.sh`; never `%`-format prose that contains percentages — use `.replace`.
- **No backticks inside double-quoted shell arguments (18 Sep 18:39).** A scorecard row passed to `dod_scorecard_row.py` in double quotes had `last` … `the` in backticks; bash executed them as a command and the words vanished from the row. Write rows with single quotes or via a Python heredoc; grep the row after writing.
- **Count natural-row files before pricing (18 Sep 19:09).** Pile and FineWeb files can differ in row count (correlative both/neither: 48 vs 64); v264's receipt was withheld (80 > 64 forwards). Compute batches from the actual row counts in the dry-run plan and set FORWARDS_MAX from that.
- **Gate the wait loop on the enqueue (18 Sep 19:31).** Twice a failed derive/dry-run left the wait loop polling for a receipt that could not exist (v223 9 min, v271 10 min). Pattern: `derive && check && enqueue || { echo FAILED; exit 1; }` in one command, wait loop in the NEXT command only after QUEUED is seen.

- **Term-wise censuses nominate; the counterfactual must match the claim (19 Sep, v295/v305/v306).** In a bilinear MLP the cross and context² terms are individually large and cancel at layer level (v295) — census the NET per-unit change δ_j = (D_j·T̂)(h_ctx − h_alone) (sums exactly to (α−1)‖T‖). And to test "unit j cancels the lookup in context", REPLACE h_ctx with h_alone; ZEROING removes the unit's whole in-context write (already ≈ 0 along the entry) and can only lower α — v305 mis-read that as a falsification until v306.
- **Block patch-back does not localise readers in bilin18 (19 Sep, v315).** Restoring any single early block's writes to native recovers ~88% of an MLP-1 edit's loss cost (shares over blocks sum to 7.5): the λ-recurrence re-injects state each block, so harm propagates as a cascade. Use path-restricted injection (the edited write fed to ONE block's input, all else native) to attribute readers.
- **Null-relative bars need an absolute floor (19 Sep, v350/v351).** v350 scored 5/5 against a zero random-unit null while the absolute effect was KL 0.0008 nats; the population edit (v351) was 0.094. Register "> 3× null AND ≥ absolute floor" for effect predictions.
- **Never wait on a result file with a bare until-loop (19 Sep, v386).** If the gate refuses the enqueue the file never appears and the loop idles to the tool timeout (10 min lost). Use `ops/dod_run_wait.sh <runner> <max_s>` (check → enqueue → wait → show, fail-fast), and before enqueuing a derived runner grep it for inherited module constants (`PHRASE`, `K`, `BATCH`, `UNITS`) — the gate refuses duplicates.

**Natural aligned pairs (19 Sep, v405):** swap the cue noun's number in place with the tokenizer (`t+"s"`, `t[:-1]`, …, keep only single-token partners) — 122/128 of the v272/v273 rows pair up. Use for any both-ends fold on text instead of unpaired label means. Exact degree expansion of a squared-attention head's source term: factors s1, s2, w; 7 factor-change terms + query-change term (`run_both_ends_degree_v401.py`).
**Closure bars (19 Sep, v414):** on float32 sums over ≥ 10⁴ terms (e.g. a 244² cross matrix) register relative 1e-3, not 1e-6 — 1e-6 failed as float noise (3e-5).
**Serial chains (19 Sep, v425):** per-writer fold shares (v406-style) double count when later writers read earlier ones (MLPs 5–7 → MLP 8): MLP 8 alone edited = 0.93 of MLPs 5–8 edited. Use replace-edits of whole writes at the position (`run_layer_swap_edit_v425.py` pattern) to arbitrate; folds only nominate.
**Nested swap edits cancel (19 Sep, v435):** a pair-swap is an involution. Swapping an upstream source (copier's value at the noun) makes downstream states partner-like; also swapping downstream values to the partner's ORIGINALS undoes it (0.68 → 0.645 → 0.57 as more was swapped). Joint replace-edits compose only for sources at ONE position (e.g. all exits at the noun). For nested paths use the upstream edit alone, or patch downstream from the already-edited run.
**Price bars on derived runners (19 Sep, v499/v500):** the forwards check runs AFTER the GPU work; a stale `FORWARDS_MAX` inherited from a parent with fewer batches aborts the run with no result (30 forwards lost). When re-deriving with a different row set, recompute batches × passes and update both the PRICE line and `FORWARDS_MAX`.
**Template d = 0 guard (19 Sep, v529):** when the mean-difference d is taken at a position other than the swapped token, block 0 (embedding only) has d = 0 → 0/0 templates → NaN scores that sort to the top. Guard `T.norm() < 1e-6` → score 0.

**Regex-derived runners: grep before enqueue (19 Sep, after v539's fourth instance).** `grep -n "if d >=\|dist_items\|verb_items\[:\|OUT = \|_MAX\b\|_MIN\b\|schema\|MODES = {\""` (v543: a literal MODES dict kept v461's names → KeyError) — re-target the row filter (adjacent `d == 1` vs distant `d >= 2`), the replay block's rows, every bar name in the plan dict, the OUT filename and the schema string. Then re-price: FORWARDS_MAX = batches × passes + 1.

**Aligned pairs by in-place swap (19 Sep).** Works when both cue forms fill the same slot with the same syntax (noun number: guests/guest; noun gender: king/queen; person: I/you). Fails for since/by (fineweb 'by' is mostly passive; gap 0.48 logits, sign constancy 0.53, v596). Always register and check the native gap's sign constancy (≥ 0.75) before spending edits on a new line.

**Disk budget (20 Sep, after a full-disk outage).** The instance's root disk is only 32GB; re-saving a large composed matrix (e.g. W_U@Down_17, ~1GB) on every capture run filled it to 0 bytes free and broke ALL Bash/Write tools for ~4 hours (even `rm` couldn't run — the tool's own output-capture needs a few free bytes). Fix: `ops/disk_guard.py` (`guard_write`/`guard_torch_save`) checks `shutil.disk_usage` before any large write and refuses below a 3GB margin, warns below 6GB. Rule going forward: NEVER serialize a composed/product matrix (anything derived by multiplying two weight matrices together, e.g. W_U@Down_L) to disk between two runners that run seconds apart -- save the small FACTOR matrices only (or nothing; recompute from the model in the next script, it's 0 forwards and seconds) and call disk_guard before any torch.save of anything that could be large. Check `df -h /` opportunistically before big writes.
