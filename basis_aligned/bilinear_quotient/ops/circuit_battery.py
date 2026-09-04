#!/usr/bin/env python
"""circuit_battery -- ONE reusable circuit protocol run over the whole task bank.

User directive 2026-09-04T03:43Z: "Why do you need fresh data for every unique circuit? We should do the 20/80 here... Figure out
what's essential, build tools that can be built once and reused, and then scale."  This is that tool.  For EVERY behaviour in
ops/circuit_battery_tasks.py the same four stages run with the same frozen bars, so a new circuit costs a task-bank entry and a
couple of GPU-minutes instead of a bespoke dataset, prereg and rung:

  CAPABILITY   native argmax over the task's answer vocabulary on A1 base prompts (a behaviour the model cannot do gets no circuit).
  LOCALISE     donor-to-base interchange patch of each of the 36 components (attn_l, mlp_l), scored as normalized logit-difference
               recovery REC = (ld_patch - ld_base) / (ld_donor - ld_base); FIT chooses the writer, SELECT scores it.
  SPLIT        the chosen writer's final-position write W is carried as a parallel residual (scaled by each block skip lambda0) and
               subtracted from the INPUT of a chosen reader set only: DIRECT (final norm), READS (all downstream components),
               each component singly, and FULL (= every edge, which must equal ablating W itself -- the instrument check).
  SELECTIVITY  the same FULL deletion on the answer-preserving family P and the copy control C, relative to its damage on A1.

# BQGATE: EXPERIMENT  pred_a_instrument_full_equals_writer_ablation pred_b_bank_capability pred_c_writer_localisation
#                     pred_d_writer_selectivity pred_e_readers_are_redundant_not_concentrated
#                     pred_f_screen_writer_replicates pred_g_screen_no_writer_is_selective
#                     pred_h_screen_redundancy_holds_on_ood

SIGN CONVENTION: margin m = logit(answer) - max logit(other candidate); damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS the
behaviour.  Recovery REC is 0 for no effect and 1 for a full swap.  Nothing here installs into the SS312 frontier (SS2135 applies only
to frontier L2 numbers, which this rung does not touch).
Preregistration: polynomial_causal/CIRCUIT_BATTERY_PROTOCOL_PREREGISTRATION.md
"""
import hashlib
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery_tasks as BANK
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("circuit_battery is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_PROTOCOL_V2_PREREGISTRATION.md"
SMOKE_OUT = ROOT / "circuit_battery_v2_smoke_results.json"   # a smoke never clobbers the real receipt
HASHES = {PREREG: "e24f69d5e1a5cafb06766a32c351f94815900dfe67245a8e35aac25ce27c2505", R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
RUNG = "circuit_battery_v2"
PROTOCOL = "circuit_battery_v2"    # v1 (SS2809) is preserved as a diagnostic screen, not evidence
D, NH, HD, NL = R.D, R.NH, R.HD, R.NL
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = SMOKE_OUT if SMOKE else ROOT / "circuit_battery_v2_results.json"
PER_CELL = 6 if SMOKE else 24
TASKS = sorted(BANK.TASKS)
COMPONENTS = [(kd, l) for l in range(NL) for kd in ("attn", "mlp")]

SCREEN = {"writer": "attn8", "writer_tasks": 7, "selective_tasks": 0, "ood_top3": 0.60}
BARS = {"exact_tol": 1e-4, "capability_acc": 0.80, "capability_tasks": 8, "localise_rec": 0.50,
        "localise_tasks_frac": 0.50, "select_ratio": 0.25, "select_tasks_frac": 0.50,
        "reader_top3_share": 0.80, "margin_floor": 0.5}
NULLS = {"capability_tasks_le": 2, "localise_rec_le": 0.20, "select_ratio_ge": 0.75,
         "reader_top3_share_ge": 0.80}


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run(m, tokens, finals, *, cache=None, patch=None, writer=None, removed=(), ablate=False):
    """One forward through the observed model with optional component caching,
    component-output patching, and exact residual path-removal of the writer's
    final-position write.  `cache` (dict) is filled with per-component outputs."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x
    v1 = None
    Wj = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if Wj is not None:
            Wj = blk.lambdas[0] * Wj
        xr = x - Wj if (Wj is not None and ("attn", site) in removed) else x
        write, v1 = blk.attn(F.rms_norm(xr, (D,)), v1)
        if patch is not None and ("attn", site) in patch:
            write = patch[("attn", site)]
        if cache is not None:
            cache[("attn", site)] = write
        if writer == ("attn", site):
            Wj = torch.zeros_like(x); Wj[ar, finals] = write[ar, finals]
            if ablate:                       # reference for the instrument check: never write W at all
                write = write - Wj
        x = x + write
        xm = x - Wj if (Wj is not None and ("mlp", site) in removed) else x
        out = blk.mlp(F.rms_norm(xm, (D,)))
        if patch is not None and ("mlp", site) in patch:
            out = patch[("mlp", site)]
        if cache is not None:
            cache[("mlp", site)] = out
        if writer == ("mlp", site):
            Wj = torch.zeros_like(x); Wj[ar, finals] = out[ar, finals]
            if ablate:
                out = out - Wj
        x = x + out
    xf = x - Wj if (Wj is not None and "direct" in removed) else x
    logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(xf, (D,))) / 30.0)
    return logits[ar, finals].float()


def margins(logit_rows, ans_ids, cand_ids):
    """m = logit(answer) - max logit(other candidate in the task vocabulary)."""
    sub = logit_rows[:, cand_ids]
    pos = (cand_ids.unsqueeze(0) == ans_ids.unsqueeze(1)).float()
    own = (sub * pos).sum(1)
    other = (sub - 1e4 * pos).max(1).values
    return own - other


def batches(rows, size=32):
    by_len = defaultdict(list)
    for r in rows:
        by_len[len(r["base_ids"])].append(r)
    for _, group in sorted(by_len.items()):
        for i in range(0, len(group), size):
            yield group[i:i + size]


def pack(rows, key):
    ids = torch.tensor([r[f"{key}_ids"] for r in rows], device=DEV)
    finals = torch.full((len(rows),), ids.size(1) - 1, device=DEV, dtype=torch.long)
    ans = torch.tensor([r[f"{key}_answer_id"] for r in rows], device=DEV)
    return ids, finals, ans


# --------------------------------------------------------------------------- #
# per-task protocol
# --------------------------------------------------------------------------- #
def run_task(m, task_id, fwd):
    rows = BANK.build_rows(task_id, per_cell=PER_CELL)
    cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(task_id)}), device=DEV)
    cells = defaultdict(list)
    dropped = 0
    for r in rows:
        if r["family"] in ("A1", "A2") and len(r["base_ids"]) != len(r["donor_ids"]):
            dropped += 1                      # interchange needs aligned lengths
            continue
        cells[(r["family"], r["split"])].append(r)

    out = {"task_id": task_id, "dropped_unaligned": dropped, "rows": len(rows),
           "description": BANK.TASKS[task_id].description,
           "causal_variable": BANK.TASKS[task_id].causal_variable}

    manifest = {sp: {fam: hashlib.sha256(
        ",".join(sorted(r["row_id"] for r in cells[(fam, sp)])).encode()).hexdigest()
        for fam in BANK.TASKS[task_id].families} for sp in BANK.SPLITS}
    out["row_manifest_sha256"] = manifest
    out["group_ids"] = {sp: sorted({r["group_id"] for r in cells[("A1", sp)]})[:3] for sp in BANK.SPLITS}
    out["split_policy"] = BANK.split_policy(task_id, per_cell=PER_CELL)
    out["phases"] = {"FIT": "writer selection only", "SELECT": "scoring",
                     "TEST": "held out, opened after selection", "OOD": "held out, opened after selection"}

    # ---- CAPABILITY: native argmax over the task vocabulary, A1 base prompts ----
    acc, nat_m = [], []
    acc_by = {}
    for split in BANK.SPLITS:
        for b in batches(cells[("A1", split)]):
            ids, fin, ans = pack(b, "base")
            lg = run(m, ids, fin); fwd[0] += 1
            hit = (lg[:, cand].argmax(1) == (cand.unsqueeze(0) == ans.unsqueeze(1)).float().argmax(1)).float().cpu().numpy()
            acc.append(hit); acc_by.setdefault(split, []).append(hit)
            nat_m.append(margins(lg, ans, cand).cpu().numpy())
    out["capability_acc"] = float(np.concatenate(acc).mean())
    out["capability_by_split"] = {k: float(np.concatenate(v).mean()) for k, v in acc_by.items()}
    out["native_margin"] = float(np.concatenate(nat_m).mean())
    out["capable"] = out["capability_acc"] >= BARS["capability_acc"]

    # ---- LOCALISE: donor->base interchange patch of every component ----
    rec = {}
    for split in ("FIT", "SELECT"):
        per = defaultdict(list)
        for b in batches(cells[("A1", split)]):
            bid, bfin, bans = pack(b, "base")
            did, dfin, dans = pack(b, "donor")
            dcache = {}
            ld_don = run(m, did, dfin, cache=dcache); fwd[0] += 1
            lb = run(m, bid, bfin); fwd[0] += 1
            base_ld = (lb.gather(1, dans[:, None]) - lb.gather(1, bans[:, None])).squeeze(1)
            don_ld = (ld_don.gather(1, dans[:, None]) - ld_don.gather(1, bans[:, None])).squeeze(1)
            denom = (don_ld - base_ld).clamp_min(1e-3)
            for comp in COMPONENTS:
                lp = run(m, bid, bfin, patch={comp: dcache[comp]}); fwd[0] += 1
                ld = (lp.gather(1, dans[:, None]) - lp.gather(1, bans[:, None])).squeeze(1)
                per[comp].append(((ld - base_ld) / denom).cpu().numpy())
        rec[split] = {f"{k}{l}": float(np.concatenate(v).mean()) for (k, l), v in per.items()}
    out["recovery"] = rec
    fit_rank = sorted(rec["FIT"], key=lambda k: -rec["FIT"][k])
    out["fit_rank"] = fit_rank[:6]
    writer_name = fit_rank[0]
    kind = "attn" if writer_name.startswith("attn") else "mlp"
    writer = (kind, int(writer_name[len(kind):]))
    out["writer"] = writer_name
    out["writer_recovery_select"] = rec["SELECT"][writer_name]

    # ---- SPLIT: who reads the writer's final-position write? ----
    wl = writer[1]
    readers = ([("mlp", wl)] if kind == "attn" else []) + \
              [(k2, l2) for l2 in range(wl + 1, NL) for k2 in ("attn", "mlp")]
    arms = {"FULL": tuple(readers) + ("direct",), "DIRECT": ("direct",), "READS": tuple(readers)}
    for c in readers:
        arms[f"COMP_{c[0]}{c[1]}"] = (c,)
    split_dm, exact = defaultdict(list), []
    for b in batches(cells[("A1", "SELECT")]):
        ids, fin, ans = pack(b, "base")
        lg = run(m, ids, fin); fwd[0] += 1
        mn = margins(lg, ans, cand)
        abl = run(m, ids, fin, writer=writer, ablate=True); fwd[0] += 1   # writer's final-position write never happens
        for name, rem in arms.items():
            lg2 = run(m, ids, fin, writer=writer, removed=rem); fwd[0] += 1
            split_dm[name].append((mn - margins(lg2, ans, cand)).cpu().numpy())
            if name == "FULL":               # every reader edge + direct removed == the write never happening
                exact.append((abl - lg2).abs().max().item())
    out["split_d_m"] = {k: float(np.concatenate(v).mean()) for k, v in split_dm.items()}
    out["instrument_max_dev"] = float(max(exact))
    reads = max(out["split_d_m"]["READS"], BARS["margin_floor"])
    comps = sorted(((k, v) for k, v in out["split_d_m"].items() if k.startswith("COMP_")),
                   key=lambda kv: -kv[1])
    out["reader_ladder"] = [[k, round(v, 4)] for k, v in comps[:8]]
    out["reader_top3_share"] = float(sum(v for _, v in comps[:3]) / reads)
    out["reads_share"] = float(out["split_d_m"]["READS"] / max(out["split_d_m"]["FULL"], BARS["margin_floor"]))
    out["direct_share"] = float(out["split_d_m"]["DIRECT"] / max(out["split_d_m"]["FULL"], BARS["margin_floor"]))

    # ---- SELECTIVITY: same FULL deletion on the P and C control families ----
    ctrl = {}
    for fam in ("P", "C"):
        dm = []
        for b in batches(cells[(fam, "SELECT")]):
            ids, fin, ans = pack(b, "base")
            lg = run(m, ids, fin); fwd[0] += 1
            lg2 = run(m, ids, fin, writer=writer, removed=arms["FULL"]); fwd[0] += 1
            dm.append((margins(lg, ans, cand) - margins(lg2, ans, cand)).cpu().numpy())
        ctrl[fam] = float(np.concatenate(dm).mean()) if dm else float("nan")
    a1 = max(out["split_d_m"]["FULL"], BARS["margin_floor"])
    out["control_d_m"] = ctrl
    out["selectivity_ratio"] = float(max(ctrl["P"], ctrl["C"]) / a1)

    # ---- HELD-OUT: the writer's necessity and the reader ladder on TEST and OOD ----
    for split in ("TEST", "OOD"):
        dm, ladder = [], defaultdict(list)
        for b in batches(cells[("A1", split)]):
            ids, fin, ans = pack(b, "base")
            lg = run(m, ids, fin); fwd[0] += 1
            mn = margins(lg, ans, cand)
            lg2 = run(m, ids, fin, writer=writer, removed=arms["FULL"]); fwd[0] += 1
            dm.append((mn - margins(lg2, ans, cand)).cpu().numpy())
            for name in ("READS",) + tuple(f"COMP_{c[0]}{c[1]}" for c in readers[:6]):
                lg3 = run(m, ids, fin, writer=writer, removed=arms[name]); fwd[0] += 1
                ladder[name].append((mn - margins(lg3, ans, cand)).cpu().numpy())
        out[f"{split.lower()}_full_d_m"] = float(np.concatenate(dm).mean())
        lad = {k: float(np.concatenate(v).mean()) for k, v in ladder.items()}
        out[f"{split.lower()}_ladder"] = lad
        comps_h = sorted(((k, v) for k, v in lad.items() if k.startswith("COMP_")), key=lambda kv: -kv[1])
        out[f"{split.lower()}_top3_share"] = float(
            sum(v for _, v in comps_h[:3]) / max(lad.get("READS", 0.0), BARS["margin_floor"]))
    out["heldout_gap"] = abs(out["test_full_d_m"] - out["split_d_m"]["FULL"])
    return out


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


def main():
    t0 = time.time()
    check_hashes()
    fwd = [0]
    m = R.load_model().to(DEV).eval()
    results = {}
    for tid in TASKS:
        r0 = time.time()
        results[tid] = run_task(m, tid, fwd)
        results[tid]["seconds"] = round(time.time() - r0, 2)
        print(f"[battery] {tid:34s} cap={results[tid]['capability_acc']:.2f} "
              f"writer={results[tid]['writer']} rec={results[tid]['writer_recovery_select']:.2f} "
              f"reads={results[tid]['reads_share']:.2f} sel={results[tid]['selectivity_ratio']:.2f} "
              f"({results[tid]['seconds']}s)", flush=True)

    capable = [t for t in TASKS if results[t]["capable"]]
    loc = [t for t in capable if results[t]["writer_recovery_select"] >= BARS["localise_rec"]]
    sel = [t for t in capable if results[t]["selectivity_ratio"] <= BARS["select_ratio"]]
    top3 = [results[t]["reader_top3_share"] for t in capable]
    dev = max(results[t]["instrument_max_dev"] for t in TASKS)
    attn8 = [t for t in capable if results[t]["writer"] == SCREEN["writer"]]
    ood3 = [results[t]["ood_top3_share"] for t in capable]
    preds = {
        'pred_a_instrument_full_equals_writer_ablation': bool(dev <= BARS["exact_tol"]),
        'pred_b_bank_capability': bool(len(capable) >= BARS["capability_tasks"]),
        'pred_c_writer_localisation': bool(capable and len(loc) >= BARS["localise_tasks_frac"] * len(capable)),
        'pred_d_writer_selectivity': bool(capable and len(sel) >= BARS["select_tasks_frac"] * len(capable)),
        'pred_e_readers_are_redundant_not_concentrated':
            bool(top3 and float(np.median(top3)) <= BARS["reader_top3_share"]),
        # prospective replication of the SS2809 SCREEN's three headline claims, on the repaired bank
        'pred_f_screen_writer_replicates': bool(len(attn8) >= SCREEN["writer_tasks"]),
        'pred_g_screen_no_writer_is_selective': bool(len(sel) == SCREEN["selective_tasks"]),
        'pred_h_screen_redundancy_holds_on_ood':
            bool(ood3 and float(np.median(ood3)) <= SCREEN["ood_top3"]),
    }
    nulls = {
        "b_null_capability_le_2": bool(len(capable) <= NULLS["capability_tasks_le"]),
        "c_null_localise_rec_le_.2": bool(capable and float(np.median(
            [results[t]["writer_recovery_select"] for t in capable])) <= NULLS["localise_rec_le"]),
        "d_null_selectivity_ge_.75": bool(capable and float(np.median(
            [results[t]["selectivity_ratio"] for t in capable])) >= NULLS["select_ratio_ge"]),
        "e_null_top3_ge_.8": bool(top3 and float(np.median(top3)) >= NULLS["reader_top3_share_ge"]),
        "f_null_attn8_le_2": bool(len(attn8) <= 2),
        "g_null_majority_selective": bool(capable and len(sel) >= 0.5 * len(capable)),
        "h_null_ood_top3_ge_.8": bool(ood3 and float(np.median(ood3)) >= 0.8),
    }
    summary = {"capable": capable, "localised": loc, "selective": sel,
               "attn8_writers": attn8, "screen": SCREEN,
               "median_ood_top3_share": float(np.median(ood3)) if ood3 else None,
               "median_top3_share": float(np.median(top3)) if top3 else None,
               "median_recovery": float(np.median([results[t]["writer_recovery_select"] for t in capable])) if capable else None,
               "max_instrument_dev": dev,
               "writers": {t: results[t]["writer"] for t in TASKS}}
    result = {"rung": RUNG, "protocol": PROTOCOL, "bank_source_sha256": BANK.bank_digest()["source_sha256"],
              "splits": list(BANK.SPLITS), "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "summary": summary, "tasks": results, "bank": BANK.bank_digest(),
              "per_cell": PER_CELL, "smoke": SMOKE, "device": "cuda",
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": {k: summary[k] for k in
          ("capable", "localised", "selective", "median_top3_share", "median_recovery")}}, indent=1))


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: {len(TASKS)} tasks x (36 localise + ~40 split/control) forwards; "
              f"per_cell={PER_CELL}; no model loaded")
        sys.exit(0)
    main()
