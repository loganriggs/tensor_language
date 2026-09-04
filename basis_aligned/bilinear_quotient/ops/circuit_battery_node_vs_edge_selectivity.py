"""circuit_battery_node_vs_edge_selectivity -- is "no component is selective" a structure, or a SATURATION artifact?

SS2849 swept all 36 components on the numbered-list successor with the answer-preserving family P and the copy control C, and
every one of the seven ADMISSIBLE components came out at selectivity ratio EXACTLY 1.00 -- attn1, attn5, attn6, attn8, mlp0,
mlp1, mlp4. Meanwhile SS2819 measured the EDGES of attention 8's write at .59 (mlp11), .90 (mlp10), 1.06 (mlp9), 1.12 (mlp8).
Read naively that says task specificity in this model lives in edges, not nodes.

But a ratio of exactly 1.00 across seven very different components is also what SATURATION looks like: if removing a component
destroys the answer on every family, each family's damage equals its own native margin and the ratio is 1 by construction,
carrying no information about selectivity at all. Nothing in SS2849 distinguished those, and the whole edge-versus-node reading
depends on which it is.

This rung measures the saturation directly -- damage as a fraction of each family's NATIVE margin -- for nodes and for edges on
the same rows, and only then compares their selectivity.

# BQGATE: EXPERIMENT  pred_a_node_arms_are_saturated pred_b_controls_are_saturated_too
#                     pred_c_edges_are_not_saturated pred_d_edges_are_more_selective_than_nodes
#                     pred_e_ratios_replicate_the_earlier_rungs

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm in margin units, POSITIVE = the arm HURTS that family's own answer; saturation
fraction = d_m / max(m_NATIVE, .5), so 1.0 means the arm removed the entire native margin; selectivity ratio =
max(|d_P|, |d_C|) / max(d_A1, .5), LOWER IS MORE SELECTIVE. No CE and no SS312 L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_NODE_VS_EDGE_SELECTIVITY_PREREGISTRATION.md
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_node_vs_edge_selectivity.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_NODE_VS_EDGE_SELECTIVITY_PREREGISTRATION.md"
SWEEP = ROOT / "circuit_battery_successor_full_sweep_results.json"
RSEL = ROOT / "circuit_battery_reader_selectivity_results.json"
RUNG = "circuit_battery_node_vs_edge_selectivity"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "d63035b30c9a832b0238a9b7f19f12a7eecf8da1f529767a2c485fac940315e6",
          SWEEP: "6e20b244de4a96f1158d5fbbcdc704926048f6b87b1c524480c8b6f3c2a7b3be",
          RSEL: "8170669c13428850aa07c4539de28dd7d8164f25eb100f11996b63b02121735f",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D, NLAY = R.D, R.NL
TASK = "numbered_list.index_successor"
SPLIT = "OOD"
PER_CELL = 4 if SMOKE else 24
WRITER = ("attn", 8)
NODES = (("attn", 1), ("attn", 5), ("attn", 6), ("attn", 8), ("mlp", 0), ("mlp", 1), ("mlp", 4))
EDGES = (("mlp", 8), ("mlp", 9), ("mlp", 10), ("mlp", 11))
BARS = {"node_sat": 0.90, "ctrl_sat": 0.90, "edge_sat": 0.60, "gap": 0.30, "repro": 0.20, "floor": 0.5}
NULLS = {"node_sat_le": 0.60, "ctrl_sat_le": 0.60, "edge_sat_ge": 0.90, "gap_le": 0.0}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_node(m, tokens, finals, remove=()):
    """Whole-component ablation at every position -- SS2849's arm."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        if ("attn", site) in remove:
            write = torch.zeros_like(write)
        x = x + write
        out = blk.mlp(F.rms_norm(x, (D,)))
        if ("mlp", site) in remove:
            out = torch.zeros_like(out)
        x = x + out
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()


def main():
    t0 = time.time()
    check_hashes()
    m = fastload.load_model_fast().to(DEV).eval()
    rows = BANK.build_rows(TASK, per_cell=PER_CELL)
    fams = set(BANK.TASKS[TASK].families)
    cells = {f: [r for r in rows if r["family"] == f and r["split"] == SPLIT]
             for f in ("A1", "P", "C") if f in fams}
    cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(TASK)}), device=DEV)
    fwd = [0]

    native, node_d, edge_d = {}, defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for fam, rws in cells.items():
        nat = []
        for b in CB.batches(rws):
            ids, fin, ans = CB.pack(b, "base")
            lg = run_node(m, ids, fin); fwd[0] += 1
            mn = CB.margins(lg, ans, cand)
            nat.append(mn.cpu().numpy())
            for comp in NODES:
                lg2 = run_node(m, ids, fin, remove=(comp,)); fwd[0] += 1
                node_d[fam][f"{comp[0]}{comp[1]}"].append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
            for comp in EDGES:
                lg3 = CB.run(m, ids, fin, writer=WRITER, removed=(comp,)); fwd[0] += 1
                edge_d[fam][f"{comp[0]}{comp[1]}"].append((mn - CB.margins(lg3, ans, cand)).cpu().numpy())
        native[fam] = float(np.concatenate(nat).mean())

    nd = {f: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for f, d in node_d.items()}
    ed = {f: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for f, d in edge_d.items()}

    def sat(dmg, fam):
        return {k: v / max(native[fam], BARS["floor"]) for k, v in dmg[fam].items()}

    def ratio(dmg, k):
        ctrl = max(abs(dmg.get("P", {}).get(k, 0.0)), abs(dmg.get("C", {}).get(k, 0.0)))
        return ctrl / max(dmg["A1"][k], BARS["floor"])

    node_sat = sat(nd, "A1"); edge_sat = sat(ed, "A1")
    ctrl_sat = {k: max(sat(nd, "P").get(k, 0.0), sat(nd, "C").get(k, 0.0)) for k in nd["A1"]}
    node_ratio = {k: ratio(nd, k) for k in nd["A1"]}
    edge_ratio = {k: ratio(ed, k) for k in ed["A1"]}
    med = lambda d: float(np.median(list(d.values())))
    gap = min(node_ratio.values()) - min(edge_ratio.values())

    sw = json.load(open(SWEEP))["tasks_detail"][TASK]["components"]
    rs = json.load(open(RSEL))["tasks"][TASK]["reader_ratios"]
    repro_node = max(abs(node_ratio[k] - sw[k]["ratio"]) for k in node_ratio)
    repro_edge = max(abs(edge_ratio[k] - rs[k]) for k in edge_ratio if k in rs)

    preds = {
        'pred_a_node_arms_are_saturated': bool(med(node_sat) >= BARS["node_sat"]),
        'pred_b_controls_are_saturated_too': bool(med(ctrl_sat) >= BARS["ctrl_sat"]),
        'pred_c_edges_are_not_saturated': bool(med(edge_sat) <= BARS["edge_sat"]),
        'pred_d_edges_are_more_selective_than_nodes': bool(gap >= BARS["gap"]),
        'pred_e_ratios_replicate_the_earlier_rungs': bool(max(repro_node, repro_edge) <= BARS["repro"]),
    }
    nulls = {
        "a_null_nodes_not_saturated": bool(med(node_sat) <= NULLS["node_sat_le"]),
        "b_null_controls_not_saturated": bool(med(ctrl_sat) <= NULLS["ctrl_sat_le"]),
        "c_null_edges_saturated_too": bool(med(edge_sat) >= NULLS["edge_sat_ge"]),
        "d_null_no_gap": bool(gap <= NULLS["gap_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "task": TASK, "split": SPLIT, "nodes": [f"{k}{l}" for k, l in NODES],
              "edges": [f"{k}{l}" for k, l in EDGES],
              "summary": {"native_margin": native,
                          "node_saturation": node_sat, "control_saturation": ctrl_sat,
                          "edge_saturation": edge_sat,
                          "node_ratio": node_ratio, "edge_ratio": edge_ratio,
                          "min_node_ratio": min(node_ratio.values()),
                          "min_edge_ratio": min(edge_ratio.values()), "gap": gap,
                          "medians": {"node_sat": med(node_sat), "ctrl_sat": med(ctrl_sat),
                                      "edge_sat": med(edge_sat)},
                          "repro_node": repro_node, "repro_edge": repro_edge},
              "node_damage": nd, "edge_damage": ed, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"]}, indent=1)[:1400])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: saturation and selectivity for {len(NODES)} node ablations and {len(EDGES)} "
              f"edge removals on {TASK} {SPLIT}; no model loaded")
        sys.exit(0)
    main()
