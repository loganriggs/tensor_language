#!/usr/bin/env python
"""numbered_list_cached_value_read_split_probe -- WHO READS the final-label cached-value term T of the numbered-list successor circuit?
R573/R576: T = sum_{h in {3,7}} p8_{h,q,k} W_O,h^(8) (lambda8 W_V,h^(0) z_k^(0)) written by attention 8 at the final query from the final
visible label is necessary for the +1 answer (FIT margin damage 2.2-2.4) but not selective (on repeated-label lists its deletion LOWERS CE
by .31). This rung path-patches T EDGE BY EDGE: T is carried alongside the native residual (scaled by the same block-skip lambda0 products)
and subtracted from the INPUT of a chosen reader set only -- the final norm (DIRECT), all 19 downstream component reads (READS), each
component singly, each block, the FIT-chosen top-2 jointly, and every edge (FULL = R576's whole-term deletion, the instrument check).
Plus a zero-forward direct lens: does T's own unembedding favour COPY of the label or its successor? CUDA lane-1. Codex allocation
2026-09-04T03:22Z item 1 (task.numbered_list.index_successor downstream-read split). Codex's rung573/rung576 modules are IMPORTED, not edited.

# BQGATE: EXPERIMENT  pred_a_instrument_full_reproduces_r576 pred_b_downstream_readers_carry_the_successor_effect
#                     pred_c_top2_readers_concentrate_the_effect pred_d_same_readers_carry_the_repeated_list_collateral
#                     pred_e_t_direct_lens_reads_copy_not_successor

SIGN CONVENTION: margin damage d_m = m_NATIVE - m_arm, POSITIVE = the arm HURTS the correct answer; CE change d_CE = CE_arm - CE_NATIVE,
NEGATIVE = the arm HELPS. Nothing installs into the §312 frontier.
Preregistration: polynomial_causal/NUMBERED_LIST_CACHED_VALUE_READ_SPLIT_PROBE_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SELF = Path(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0].endswith(".py") else Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_read_split_probe.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")   # jacclust.tt_model for the facade
import mlp_in_situ_usage_rank_map_probe as R
import numbered_list_cached_value_weight_removal_rung576 as r576   # Codex's frozen R576 module: compiled T, candidate pool, margin/ce
import numbered_list_factor_localization_rung573 as r573          # Codex's frozen R573 module: exact attention replay
import bilin18_observed_model_facade as facade

if not torch.cuda.is_available():
    raise RuntimeError("numbered_list_cached_value_read_split_probe is a lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "NUMBERED_LIST_CACHED_VALUE_READ_SPLIT_PROBE_PREREGISTRATION.md"
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
POSITIONS = ROOT / "numeric_factor_removal_positions_rung575.json"
R576_RESULTS = ROOT / "numbered_list_cached_value_weight_removal_rung576_results.json"
OUT = ROOT / "numbered_list_cached_value_read_split_probe_results.json"
HASHES = {PREREG: "5698b36a626067881b384a14921b6d87845e9caf9a5944e9b06a380f2d343aca",
          ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
          POSITIONS: "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
          R576_RESULTS: "a6041c28cefc4f695f6e649210884774ed576bae80c14c031473d6b8c8ff2f73",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
RUNG = "numbered_list_cached_value_read_split_probe"
D, NH, HD, NL = R.D, R.NH, R.HD, R.NL
LAYER = r576.LAYER            # 8
LIST_TARGETS = r576.LIST_TARGETS
CONTROL = "list_repeated_index_control"
FAMILIES = LIST_TARGETS + (CONTROL,)
ENDPOINTS = r576.ENDPOINTS    # ("base", "donor")
SPLITS = ("FIT", "SELECT")
BATCH = 64
BOOTSTRAPS = 2000
SEED = 2808
COMPONENTS = [("mlp", LAYER)] + [(kd, l) for l in range(LAYER + 1, NL) for kd in ("attn", "mlp")]   # 19 reads of T
BLOCKS = {j: [c for c in COMPONENTS if c[1] == j] for j in range(LAYER, NL)}
BARS = {"exact_tol": 1e-8, "a_tol": 0.02, "b_reads_share": 0.6, "b_direct_share": 0.4, "c_top2_share": 0.5, "c_reads_floor": 0.5,
        "d_help_floor": -0.05, "d_frac": 0.5, "e_copy_frac": 0.75, "floor": 0.002}
NULLS = {"b_direct_share": 0.6, "c_top2_share": 0.3, "d_top2_ce": -0.05, "e_copy_frac": 0.25}


def check_hashes():
    for p, h in HASHES.items():
        if h is None or len(h) != 64:
            continue
        if not p.is_file() or R.sha256(p) != h:
            raise RuntimeError(f"frozen hash mismatch: {p}")


def cname(c):
    return "direct" if c == "direct" else f"{c[0]}{c[1]}"


def lower(values, seed):
    a = np.asarray(values, dtype=np.float64)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(a), size=(BOOTSTRAPS, len(a)))
    return float(np.quantile(a[idx].mean(1), .025))


def load_rows():
    rows = [r for r in json.loads(ROWS.read_text())["rows"] if r["split"] in SPLITS and r["family_id"] in FAMILIES]
    pdoc = json.loads(POSITIONS.read_text())
    assert pdoc["model_loaded"] is False and pdoc["outcomes_opened"] == []
    positions = {it["row_id"]: it for it in pdoc["records"]}
    assert len(rows) == 288 and all(r["row_id"] in positions for r in rows)
    return rows, positions


def chunk_items(items):
    """Equal-length groups of at most BATCH (row, endpoint) items; deterministic order."""
    ordered = sorted(items, key=lambda it: (len(it[0][f"{it[1]}_ids"]), it[0]["row_id"], it[1]))
    out, cur = [], 0
    while cur < len(ordered):
        L = len(ordered[cur][0][f"{ordered[cur][1]}_ids"]); ch = []
        while cur < len(ordered) and len(ordered[cur][0][f"{ordered[cur][1]}_ids"]) == L and len(ch) < BATCH:
            ch.append(ordered[cur]); cur += 1
        out.append(ch)
    return out


@torch.no_grad()
def run(m, tokens, finals, sources, removed):
    """One forward with T subtracted from the INPUT of every reader in `removed` (subset of COMPONENTS + {"direct"}).
    T is carried as a parallel residual tensor scaled by the same block-skip lambda0 as the residual itself, so removing it
    from every edge is exactly R576's whole-term deletion, and removing it from one reader changes only what that reader sees."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,)); x0 = x; v1 = None; Tj = None; T = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        if Tj is not None:
            Tj = blk.lambdas[0] * Tj
        xr = x - Tj if (Tj is not None and ("attn", site) in removed) else x
        st = F.rms_norm(xr, (D,))
        if site == LAYER:
            _w, tensors, _e = r573.replay_attention(st, v1, blk.attn, finals)
            cached = r576.compiled_cached(m, tokens)
            T = r576.projected_terms(tensors, cached, finals, sources, blk.attn.c_proj.weight)
            Tj = torch.zeros_like(x); Tj[ar, finals] = T.to(x.dtype)
        write, v1 = blk.attn(st, v1)
        x = x + write
        xm = x - Tj if (Tj is not None and ("mlp", site) in removed) else x
        x = x + blk.mlp(F.rms_norm(xm, (D,)))
    xf = x - Tj if ("direct" in removed) else x
    logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(xf, (D,))) / 30.0)
    rms = xf[ar, finals].float().pow(2).mean(-1).add(torch.finfo(torch.float32).eps).sqrt()
    return logits[ar, finals].float(), T.float(), rms


def arms_dict():
    arms = {"NATIVE": set(), "FULL": set(COMPONENTS) | {"direct"}, "DIRECT": {"direct"}, "READS": set(COMPONENTS)}
    for c in COMPONENTS:
        arms["COMP_" + cname(c)] = {c}
    for j, cs in BLOCKS.items():
        arms[f"BLOCK_{j}"] = set(cs)
    return arms


def evaluate(m, items, positions, arms, WU, log):
    """raw[arm][family][endpoint] -> list of per-row dicts (d_m, d_ce); plus the direct-lens rows and the replay check."""
    raw = {a: defaultdict(lambda: defaultdict(list)) for a in arms}
    lens_rows = []
    fwd = 0; replay_err = None
    for gi, group in enumerate(chunk_items(items)):
        tokens, finals, sources = r576.batch_endpoint(group, positions, DEV)
        nat, T, rms = run(m, tokens, finals, sources, set()); fwd += 1
        if replay_err is None:
            ref = facade.forward_with_dispatch(m, tokens, lambda e: e.block.attn(e.state, e.first_value), lambda e: e.block.mlp(e.state), require_production=False)
            ref = ref[torch.arange(tokens.size(0), device=DEV), finals].float(); fwd += 1
            replay_err = float((ref - nat).square().sum() / ref.square().sum().clamp_min(1e-30))
        lens = (T @ WU.T) / rms[:, None]                      # (B, V) direct-path pre-softcap logit contribution of T
        nat_m = []; nat_ce = []
        for i, (row, ep) in enumerate(group):
            a, aid = row[f"{ep}_answer"], row[f"{ep}_answer_id"]
            nat_m.append(r576.margin(nat[i], aid, a)); nat_ce.append(r576.ce(nat[i], aid))
            if row["family_id"] in LIST_TARGETS:
                lab = int(tokens[i, sources[i]])
                lens_rows.append({"row_id": row["row_id"], "endpoint": ep, "split": row["split"], "family": row["family_id"],
                                  "lens_label": float(lens[i, lab]), "lens_answer": float(lens[i, aid]),
                                  "copy_over_successor": bool(lens[i, lab] > lens[i, aid]), "t_norm": float(T[i].norm())})
        for an, rem in arms.items():
            if an == "NATIVE":
                continue
            out, _, _ = run(m, tokens, finals, sources, rem); fwd += 1
            for i, (row, ep) in enumerate(group):
                a, aid = row[f"{ep}_answer"], row[f"{ep}_answer_id"]
                raw[an][row["family_id"]][ep].append({"row_id": row["row_id"], "split": row["split"],
                                                       "d_m": nat_m[i] - r576.margin(out[i], aid, a),
                                                       "d_ce": r576.ce(out[i], aid) - nat_ce[i]})
        if gi % 3 == 0:
            log(chunk=gi, forwards=fwd)
    return raw, lens_rows, replay_err, fwd


def cell_table(raw_arm, split):
    cells = {}
    for fam in FAMILIES:
        for ep in ENDPOINTS:
            rs = [r for r in raw_arm[fam][ep] if r["split"] == split]
            if not rs:
                continue
            dm = [r["d_m"] for r in rs]; dce = [r["d_ce"] for r in rs]
            cells[f"{fam}/{ep}"] = {"n": len(rs), "mean_d_m": float(np.mean(dm)), "lower_d_m": lower(dm, SEED), "pos_frac": float(np.mean(np.array(dm) > 0)),
                                    "mean_d_ce": float(np.mean(dce)), "lower_d_ce": lower(dce, SEED)}
    return cells


def pooled(raw_arm, split, fams=LIST_TARGETS):
    dm = [r["d_m"] for fam in fams for ep in ENDPOINTS for r in raw_arm[fam][ep] if r["split"] == split]
    return float(np.mean(dm)), lower(dm, SEED + 1)


def control_ce(raw_arm, split):
    d = [r["d_ce"] for ep in ENDPOINTS for r in raw_arm[CONTROL][ep] if r["split"] == split]
    return float(np.mean(d)), lower(d, SEED + 2)


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        check_hashes()
        arms = arms_dict(); assert len(arms) == 33 and len(COMPONENTS) == 19
        print(json.dumps({"status": "dry_run_passed", "rung": RUNG, "arms": len(arms) + 1, "out": str(OUT)})); return
    smoke = os.environ.get("SURROGATE_SMOKE") == "1"
    log = lambda **kw: print(json.dumps({**kw, "t": round(time.time() - started, 1)}), flush=True)
    m = R.load_model().to(DEV)
    WU = m.lm_head.weight.float()
    check_hashes()
    rows, positions = load_rows()
    r576_res = json.loads(R576_RESULTS.read_text())
    if smoke:
        keep = {}
        for r in rows:
            keep.setdefault((r["family_id"], r["split"]), []).append(r)
        rows = [r for v in keep.values() for r in v[:2]]
    items = [(r, ep) for r in rows for ep in ENDPOINTS]
    arms = arms_dict()
    # pass 1: all fixed arms on FIT + SELECT rows together (each row scored at its own split).
    raw, lens_rows, replay_err, fwd = evaluate(m, items, positions, arms, WU, log)
    log(stage="pass1_done", forwards=fwd, replay_err=replay_err)
    # FIT selects the top-2 component readers by pooled signed mean margin damage over the ten list cells.
    comp_fit = {cname(c): pooled(raw["COMP_" + cname(c)], "FIT") for c in COMPONENTS}
    top2 = sorted(comp_fit, key=lambda k: -comp_fit[k][0])[:2]
    top2_set = {c for c in COMPONENTS if cname(c) in top2}
    raw2, _, _, fwd2 = evaluate(m, items, positions, {"NATIVE": set(), "TOP2_JOINT": top2_set}, WU, lambda **kw: None)
    raw["TOP2_JOINT"] = raw2["TOP2_JOINT"]; fwd += fwd2
    # ---- tables ----
    tables = {sp: {an: cell_table(raw[an], sp) for an in raw if an != "NATIVE"} for sp in SPLITS}
    pool = {sp: {an: pooled(raw[an], sp) for an in raw if an != "NATIVE"} for sp in SPLITS}
    ctrl = {sp: {an: control_ce(raw[an], sp) for an in raw if an != "NATIVE"} for sp in SPLITS}
    # ---- pred_a: instrument ----
    ref_fit = r576_res["fit_report"]
    a_dev = {}
    for fam in LIST_TARGETS:
        for ep in ENDPOINTS:
            a_dev[f"{fam}/{ep}"] = abs(tables["FIT"]["FULL"][f"{fam}/{ep}"]["mean_d_m"] - ref_fit["list_necessity"][fam][ep]["mean_margin_damage"])
    for ep in ENDPOINTS:
        a_dev[f"{CONTROL}/{ep}/ce"] = abs(tables["FIT"]["FULL"][f"{CONTROL}/{ep}"]["mean_d_ce"] - ref_fit["active_copy_controls"][CONTROL][ep]["mean_ce_increase"])
    a_ok = (replay_err <= BARS["exact_tol"]) and (max(a_dev.values()) <= BARS["a_tol"] if not smoke else True)
    # ---- pred_b: shares on SELECT ----
    full_s = max(pool["SELECT"]["FULL"][0], BARS["c_reads_floor"])
    share = {an: pool["SELECT"][an][0] / full_s for an in ("READS", "DIRECT")}
    b_ok = share["READS"] >= BARS["b_reads_share"] and share["DIRECT"] <= BARS["b_direct_share"]
    # ---- pred_c: top-2 on SELECT ----
    reads_s = max(pool["SELECT"]["READS"][0], BARS["c_reads_floor"])
    top2_sum = sum(pool["SELECT"]["COMP_" + k][0] for k in top2)
    top2_lowers = {k: pool["SELECT"]["COMP_" + k][1] for k in top2}
    c_ok = top2_sum >= BARS["c_top2_share"] * reads_s and all(v > 0 for v in top2_lowers.values())
    # ---- pred_d: repeated-index collateral on SELECT ----
    full_ce = ctrl["SELECT"]["FULL"][0]; top2_ce = ctrl["SELECT"]["TOP2_JOINT"][0]
    d_applicable = full_ce <= BARS["d_help_floor"]
    d_ok = d_applicable and top2_ce <= BARS["d_frac"] * full_ce
    # ---- pred_e: direct lens ----
    copy_frac = float(np.mean([r["copy_over_successor"] for r in lens_rows]))
    e_ok = copy_frac >= BARS["e_copy_frac"]
    preds = {
        'pred_a_instrument_full_reproduces_r576': bool(a_ok),
        'pred_b_downstream_readers_carry_the_successor_effect': bool(b_ok),
        'pred_c_top2_readers_concentrate_the_effect': bool(c_ok),
        'pred_d_same_readers_carry_the_repeated_list_collateral': bool(d_ok),
        'pred_e_t_direct_lens_reads_copy_not_successor': bool(e_ok),
    }
    nulls = {"b_null_direct_share_ge_.6": bool(share["DIRECT"] >= NULLS["b_direct_share"]),
             "c_null_top2_share_le_.3": bool(top2_sum <= NULLS["c_top2_share"] * reads_s),
             "d_null_top2_ce_ge_-.05": bool(top2_ce >= NULLS["d_top2_ce"]),
             "e_null_copy_frac_le_.25": bool(copy_frac <= NULLS["e_copy_frac"])}
    summ = {"replay_rel_sq_err": replay_err, "a_max_dev": max(a_dev.values()), "a_dev": a_dev,
            "pooled_list_d_m": {sp: {an: v[0] for an, v in pool[sp].items()} for sp in SPLITS},
            "pooled_list_lower": {sp: {an: v[1] for an, v in pool[sp].items()} for sp in SPLITS},
            "control_mean_d_ce": {sp: {an: v[0] for an, v in ctrl[sp].items()} for sp in SPLITS},
            "shares_select": share, "share_sum_select": share["READS"] + share["DIRECT"],
            "top2_fit": top2, "comp_fit_pooled": {k: v[0] for k, v in comp_fit.items()}, "top2_sum_select": top2_sum, "top2_lowers_select": top2_lowers,
            "reads_select": pool["SELECT"]["READS"][0], "full_select": pool["SELECT"]["FULL"][0],
            "full_ce_control_select": full_ce, "top2_ce_control_select": top2_ce, "d_applicable": d_applicable,
            "copy_over_successor_frac": copy_frac, "lens_median_label_minus_answer": float(np.median([r["lens_label"] - r["lens_answer"] for r in lens_rows])),
            "t_norm_median": float(np.median([r["t_norm"] for r in lens_rows]))}
    out = {"rung": RUNG, "status": "complete", "smoke": smoke,
           "sign_convention": "d_m = m_NATIVE - m_arm (POSITIVE = arm hurts the correct answer); d_ce = CE_arm - CE_NATIVE (NEGATIVE = helps)",
           "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS, "device": "cuda",
           "program": "exact residual path-patching of the R576 final-label cached-value term T by reader edge: DIRECT, READS, 19 single component reads, 10 blocks, FIT-chosen TOP2_JOINT, FULL; plus T's direct unembedding lens (copy vs successor)",
           "arms": {an: sorted(cname(c) for c in rem) for an, rem in {**arms, "TOP2_JOINT": top2_set}.items()},
           "summary": summ, "tables": tables, "lens_rows": lens_rows, "raw": raw,
           "splits_opened": list(SPLITS), "splits_closed": ["FINAL_TEST", "OOD"],
           "price": {"gpu_forwards": fwd, "backwards": 0, "fitted_parameters": 0, "gpu_seconds": time.time() - started},
           "hashes": {str(k): (v if v and len(v) == 64 else R.sha256(k)) for k, v in HASHES.items()}, "script_sha256": R.sha256(SELF)}
    if smoke:
        print(json.dumps({k: out[k] for k in ("preds", "nulls", "summary", "price")}, indent=1)); return
    from receipt import dump
    dump(out, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": {k: v for k, v in summ.items() if k not in ("a_dev", "comp_fit_pooled")}}, indent=1))


if __name__ == "__main__":
    main()
