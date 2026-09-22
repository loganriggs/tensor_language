#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_program_replays pred_b_sum_grows pred_c_half_in_few_heads pred_d_cost_tracks_value pred_e_relative_recovery_falls
"""Where does the training-length descent come from — a few heads or all of them? (v776). Pythia-160m step 1000 (v767: rank-32 recovery 1.009)
vs final (v760: 0.932). The saved rank-32 programs (kappa fitted, factored q/k maps) are applied to ONE head at a time (all other heads native;
kappa_r recomputed on 64 fit rows as in the fit) and the CE ADDED on the 192 held-out rows is the head's ISOLATED program cost; compared with
its mean-ablation value (from the result JSONs) as relative recovery 1 - cost_h / value_h. The full program is replayed first as a check.
Row-centred protocol throughout (kernels and kappa_r on row-centred logits; native row mean added back; column 0 and the diagonal native).
PREDICTIONS (scored as written; failures preserved)
    pred_a_full_program_replays   full rank-32 program CE cost within 0.02 of the saved snapshot cost, both checkpoints (kappa_r recomputed). Prior: likely
    pred_b_sum_grows              sum over heads of isolated cost at final >= 1.5 x the sum at step 1000. Prior: likely
    pred_c_half_in_few_heads      at final, the fewest heads whose isolated costs sum to half the total number <= 15 of 144. Prior: unsure
    pred_d_cost_tracks_value      at final, Spearman(isolated cost, mean-ablation value) over all 144 heads >= 0.5. Prior: likely
    pred_e_relative_recovery_falls median relative recovery over heads with value >= 0.02 is lower at final than at step 1000. Prior: likely
PRICE (registered maximum): per checkpoint: the 193-row caches take 7 forwards per CE: native 7 + kappa_r 4 + full program 7 + 144 x 7 = 1026; two checkpoints 2052 forwards (first run tripped an 1800 bar priced at 6 per CE); 0 backwards; 0 fits. Bar <= 2100.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import torch
import dod_battery
import pythia_backend as PB

ROOT = dod_battery.ROOT
TAG = "pythia160m_perhead_v776"
OUT = ROOT / f"circuits/followups/{TAG}_result.json"
CHECKPOINTS = [("step1000", "step1000", "pythia160m_step1000_v767"), ("final", None, "pythia160m_v760")]
REPO = "EleutherAI/pythia-160m"
FIT64 = ROOT / ".rowcache/pythia_fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/pythia_fineweb_n192_skip7000.pt"
CANDIDATE_ID = f"pythia.perhead_{TAG}"
FORWARDS_MAX, EBATCH, KBATCH = 2100, 32, 16
REPLAY_TOL, GROW, FEW, RHO_MIN, VALUE_FLOOR = 0.02, 1.5, 15, 0.5, 0.02
PREDICTIONS = {"pred_a_full_program_replays": "+-0.02 both", "pred_b_sum_grows": ">= 1.5x", "pred_c_half_in_few_heads": "<= 15 heads", "pred_d_cost_tracks_value": "spearman >= 0.5", "pred_e_relative_recovery_falls": "final < step1000"}


def spearman(x, y):
    rx = torch.tensor(x).argsort().argsort().double(); ry = torch.tensor(y).argsort().argsort().double(); rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "model": REPO, "checkpoints": [c[0] for c in CHECKPOINTS], "bars": {"replay_tol": REPLAY_TOL, "grow": GROW, "few": FEW, "rho_min": RHO_MIN, "value_floor": VALUE_FLOOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter(); dev = "cuda"; forwards = 0; report = {}
    ev = torch.load(EVAL_ROWS, map_location="cpu").long(); fit64 = torch.load(FIT64, map_location="cpu").long()[:64]
    for name, rev, tag in CHECKPOINTS:
        res = json.load(open(ROOT / f"circuits/followups/{tag}_result.json"))["report"]; value = res["mean_ablation_cost"]; snap = res["arms"]["32"]["snapshot_cost"]; joint = res["joint_value"]
        prog = torch.load(ROOT / f"circuits/followups/{tag}_programs.pt", map_location="cpu")["programs"]["32"]
        model = PB.load(REPO, dev, revision=rev); L, H, D, hd, rot = PB.geometry(model); ALL = [(l, h) for l in range(L) for h in range(H)]
        scaling = model.gpt_neox.layers[0].attention.scaling
        P = {k: {"kappa": prog[f"{k[0]}.{k[1]}"]["kappa"].to(dev), "q": tuple(t.to(dev) for t in prog[f"{k[0]}.{k[1]}"]["q"]), "k": tuple(t.to(dev) for t in prog[f"{k[0]}.{k[1]}"]["k"])} for k in ALL}
        bias = {}
        for (l, h) in ALL:
            _, bq, _, bk = PB.head_qk(model, l, h); bias[(l, h, "q")] = bq.to(dev); bias[(l, h, "k")] = bk.to(dev)
        state = PB.instrument(model)
        native, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw

        def row_centre(X):
            T = X.shape[-1]; pos = torch.arange(T, device=X.device); rep = ((pos[:, None] > pos[None, :]) & (pos[None, :] > 0)).float()
            m = (X * rep[None]).sum(-1) / rep.sum(-1).clamp_min(1)[None]
            return X - m[:, :, None], m

        def lowrank_logits(l, h, n):
            Uq, Vq = P[(l, h)]["q"]; Uk, Vk = P[(l, h)]["k"]
            q = (n @ Vq.T) @ Uq.T + bias[(l, h, "q")]; k = (n @ Vk.T) @ Uk.T + bias[(l, h, "k")]
            q, k = PB.rotary_qk(q, k, state["cos"], state["sin"], rot)
            return torch.einsum("bqd,bkd->bqk", q, k) * scaling

        def offset_mean(Pm, T):
            Pm, _ = row_centre(Pm)
            pos = torch.arange(T, device=dev); dmat = pos[:, None] - pos[None, :]; qm = (pos >= 8)[:, None] & (dmat > 0) & (pos[None, :] > 0); dflat = dmat[qm]
            X = Pm[:, qm]; counts = torch.bincount(dflat, minlength=513).double() * X.shape[0]
            sums_ = torch.zeros(513, dtype=torch.float64, device=dev).index_add_(0, dflat, X.double().sum(0))
            return torch.where(counts > 0, sums_ / counts.clamp_min(1), torch.zeros_like(sums_)).float()

        kr = {}; acc = {k: 0 for k in ALL}; nb = 0; state["logit_edit"] = None
        with torch.no_grad():
            for s in range(0, 64, KBATCH):
                idx = fit64[s:s + KBATCH, :-1].contiguous().to(dev); model(input_ids=idx); forwards += 1; T = idx.shape[1]
                for (l, h) in ALL:
                    acc[(l, h)] = acc[(l, h)] + offset_mean(lowrank_logits(l, h, state["n"][l]), T)
                nb += 1
        for k in ALL:
            kr[k] = acc[k] / nb
        sel = {"heads": None}     # None = all heads programmed; else a set of (l, h)

        def logit_program(l, logits):
            B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0) & (pos[None, :] > 0)
            cols = []
            for h in range(Hn):
                if sel["heads"] is not None and (l, h) not in sel["heads"]:
                    cols.append(logits[:, h]); continue
                kap = P[(l, h)]["kappa"]; pr, _ = row_centre(lowrank_logits(l, h, state["n"][l])); _, m = row_centre(logits[:, h])
                cols.append(torch.where(off[None], m[:, :, None] + kap[dmat][None] + (pr - kr[(l, h)][dmat][None]), logits[:, h]))
            return torch.stack(cols, 1)

        state["logit_edit"] = logit_program
        full, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw
        print(f"[{TAG}] {name}: native {native:.5f} | full rank-32 program cost {full - native:+.4f} (saved snapshot {snap:+.4f}) | joint value {joint:.3f}")
        cost = {}
        for (l, h) in ALL:
            sel["heads"] = {(l, h)}; c_, fw = PB.ce(model, ev, dev, EBATCH); forwards += fw; cost[f"{l}.{h}"] = c_ - native
        sel["heads"] = None; state["logit_edit"] = None; state["_restore"]()
        keys = sorted(cost, key=cost.get, reverse=True); total = sum(cost.values()); run = 0.0; n_half = 0
        for k in keys:
            run += cost[k]; n_half += 1
            if run >= 0.5 * total:
                break
        rel = {k: 1 - cost[k] / value[k] for k in cost if value[k] >= VALUE_FLOOR}; rels = sorted(rel.values()); med_rel = rels[len(rels) // 2] if rels else float("nan")
        rho = spearman([cost[k] for k in cost], [value[k] for k in cost])
        report[name] = {"native": native, "full_program_cost": full - native, "saved_snapshot_cost": snap, "joint_value": joint, "isolated_cost": cost, "value": value, "sum_isolated": total,
                        "n_heads_half": n_half, "spearman_cost_value": rho, "relative_recovery": rel, "median_relative_recovery": med_rel, "n_valued": len(rel)}
        print(f"[{TAG}] {name}: sum of isolated costs {total:+.4f} | half the total in {n_half} heads | top: " + " ".join(f"{k}:{cost[k]:+.3f}(v {value[k]:.3f})" for k in keys[:8]) + f" | Spearman(cost, value) {rho:.3f} | median relative recovery over {len(rel)} heads >= {VALUE_FLOOR}: {med_rel:.3f}")
        del model, P, bias, state; torch.cuda.empty_cache()
    a = all(abs(report[n]["full_program_cost"] - report[n]["saved_snapshot_cost"]) <= REPLAY_TOL for n in report)
    predictions = {"pred_a_full_program_replays": a, "pred_b_sum_grows": report["final"]["sum_isolated"] >= GROW * report["step1000"]["sum_isolated"],
                   "pred_c_half_in_few_heads": report["final"]["n_heads_half"] <= FEW, "pred_d_cost_tracks_value": report["final"]["spearman_cost_value"] >= RHO_MIN,
                   "pred_e_relative_recovery_falls": report["final"]["median_relative_recovery"] < report["step1000"]["median_relative_recovery"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": f"pythia_perhead_result_{TAG}", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards, "backwards": 0,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
