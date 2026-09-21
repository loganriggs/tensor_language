#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays pred_b_joint_value_comparable pred_c_kernels_only_cost pred_d_kernel_energy_top4 pred_e_four_shapes_cheap
"""Softmax replication, v751: the same rung as v750 on GPT-2 SMALL (openai-community/gpt2: 12 layers x 12 heads x 768, softmax attention,
learned absolute positions, same BPE tokenizer as bilin18 so the same row caches apply; rows fed as 513 tokens = 512 predictions).

Native CE (stock vs explicit instrument); mean-ablation value of every head (means over the 480 skip80 rows) and the joint value (all 144);
closed-form kernels kappa_h(d) = mean pre-softmax logit at offset d (queries >= 8, 64 rows) installed for every head as the off-diagonal
logits (diagonal native); the SVD k-ladder (1, 2, 4, 8, 16 shapes) inside the kernels-only program; the named 4-basis; and — new for a
model with absolute positions and a first-token sink — the kernels-only program with the POSITION-0 COLUMN kept native. Held-out
skip7000. CE ADDED. bilin18 references: joint value 3.996; kernels-only closed-form 1.45; top-4 energy 0.964; 4 shapes +0.004.
PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays    explicit-softmax native CE within 0.002 of the stock native CE (instrument)
    pred_b_joint_value_comparable joint value (all 144 mean-ablated) within [2, 8]. Prior: unsure
    pred_c_kernels_only_cost     kernels-only program costs <= 1.5. Prior: unsure (the first-token sink may make it far worse)
    pred_d_kernel_energy_top4    top-4 SVD shapes carry >= 0.90 of the kernels' energy. Prior: unsure
    pred_e_four_shapes_cheap     4 shapes inside the kernels-only program cost <= kernels-only + 0.05. Prior: unsure
PRICE (registered maximum): stock native 6; instrumented native 6; means 15; kernels 2; 144 x 6 = 864; all-ablated 6; kernels-only 6; column-0-native 6;
k-ladder 30; named-4 6; total 947 forwards; 0 backwards; 0 fits. Bar <= 970.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import torch
import dod_battery
import disk_guard
import gpt2_backend as GB

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/gpt2_values_kernels_v751_result.json"
OUT_PT = ROOT / "circuits/followups/gpt2_values_kernels_v751_tensors.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
CANDIDATE_ID = "gpt2.values_kernels_v751"
FORWARDS_MAX = 970
EBATCH = 32
LAYERS = tuple(range(12)); H = 12
KS = (1, 2, 4, 8, 16)
REPLAY_TOL, JOINT_RANGE, KO_MAX, E4_MIN, K4_TOL = 0.002, (2.0, 8.0), 1.5, 0.90, 0.05
PREDICTIONS = {"pred_a_instrument_replays": "+-0.002", "pred_b_joint_value_comparable": "in [2, 8]", "pred_c_kernels_only_cost": "<= 1.5", "pred_d_kernel_energy_top4": ">= 0.90", "pred_e_four_shapes_cheap": "<= kernels-only + 0.05"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "model": GB.REPO, "ks": list(KS),
            "bars": {"replay_tol": REPLAY_TOL, "joint_range": JOINT_RANGE, "ko_max": KO_MAX, "e4_min": E4_MIN, "k4_tol": K4_TOL}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    model = GB.load("cuda"); dev = "cuda"; D = model.config.n_embd; hd = D // H; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long(); fit = torch.load(FIT_ROWS, map_location="cpu").long(); fit64 = fit[:64]
    ALL = [(l, h) for l in LAYERS for h in range(H)]

    def ce_rows(rows):
        return GB.ce(model, rows, dev, EBATCH)

    def run_rows(rows):
        fw = 0
        with torch.no_grad():
            for s in range(0, rows.shape[0], EBATCH):
                ids = rows[s:s + EBATCH].contiguous().to(dev); model(input_ids=ids); fw += 1
        return fw

    native_stock, fw = ce_rows(ev); forwards += fw
    state = GB.instrument(model)
    native, fw = ce_rows(ev); forwards += fw
    print(f"native CE on skip7000: stock sdpa {native_stock:.5f} | explicit softmax {native:.5f}")
    # ---- capture: head means (480 rows) and per-offset mean logits (64 rows, queries >= 8) --------------------------------------------
    sums = {k: torch.zeros(hd, dtype=torch.float64, device=dev) for k in ALL}; n_tok = [0]
    ksum = {k: torch.zeros(513, dtype=torch.float64, device=dev) for k in ALL}; kcnt = torch.zeros(513, dtype=torch.float64, device=dev); cap = {"mode": None}

    def z_capture(l, z):
        if cap["mode"] == "means":
            for h in range(H):
                sums[(l, h)] += z[:, h].double().sum(dim=(0, 1))
            if l == 0:
                n_tok[0] += z.shape[0] * z.shape[2]
        return z

    def logit_capture(l, logits):
        if cap["mode"] == "kernels":
            B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]); qm = (pos >= 8)[:, None] & (dmat > 0)
            dflat = dmat[qm]
            for h in range(Hn):
                X = logits[:, h][:, qm].double()                                   # [B, n]
                ksum[(l, h)].index_add_(0, dflat, X.sum(0))
            if l == 0:
                kcnt.index_add_(0, dflat, torch.full_like(dflat, B, dtype=torch.float64))
        return logits

    state["z_edit"] = z_capture; state["logit_edit"] = logit_capture
    cap["mode"] = "means"; forwards += run_rows(fit[:, :-1])
    cap["mode"] = "kernels"; forwards += run_rows(fit64[:, :-1])
    cap["mode"] = None
    means = {k: (sums[k] / n_tok[0]).float() for k in ALL}
    kappa = {k: torch.where(kcnt > 0, ksum[k] / kcnt.clamp_min(1), torch.zeros_like(ksum[k])).float() for k in ALL}
    # ---- mean-ablation values ---------------------------------------------------------------------------------------------------------
    abl = {"set": set()}

    def z_ablate(l, z):
        for (al, h) in abl["set"]:
            if al == l:
                z = z.clone(); z[:, h] = means[(l, h)]
        return z

    state["z_edit"] = z_ablate; state["logit_edit"] = None
    value = {}
    for k in ALL:
        abl["set"] = {k}; c_, fw = ce_rows(ev); forwards += fw; value[f"{k[0]}.{k[1]}"] = c_ - native
    abl["set"] = set(ALL); c_all, fw = ce_rows(ev); forwards += fw; abl["set"] = set(); joint = c_all - native
    top = sorted(value, key=value.get, reverse=True)[:12]; vals = sorted(value.values()); median = vals[len(vals) // 2]
    print(f"mean-ablation values: top " + " ".join(f"{k}:{value[k]:.3f}" for k in top) + f" | median {median:.4f} | sum {sum(vals):.3f} | joint (all 162) {joint:.3f}")
    # ---- kernels-only program and the shape ladder ----------------------------------------------------------------------------------
    state["z_edit"] = None; prog = {"kap": kappa}

    def logit_kernel(l, logits):
        B, Hn, T, _ = logits.shape; pos = torch.arange(T, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (dmat > 0)
        out = logits.clone()
        for h in range(Hn):
            kap = prog["kap"][(l, h)]; out[:, h] = torch.where(off[None], kap[dmat][None].expand(B, T, T), logits[:, h])
        return out

    state["logit_edit"] = logit_kernel
    c_ko, fw = ce_rows(ev); forwards += fw
    prog["keep_col0"] = True

    def logit_kernel_col0(l, logits):
        out = logit_kernel(l, logits); out[:, :, :, 0] = logits[:, :, :, 0]; return out

    state["logit_edit"] = logit_kernel_col0; c_ko0, fw = ce_rows(ev); forwards += fw; state["logit_edit"] = logit_kernel
    print(f"kernels-only with the position-0 column native: {c_ko0 - native:+.4f} (vs kernels-only {c_ko - native:+.4f})")
    Kmat = torch.stack([kappa[k][1:512] for k in ALL]).double(); U, S, Vh = torch.linalg.svd(Kmat, full_matrices=False); energy = (S ** 2).cumsum(0) / (S ** 2).sum()

    def project(B):
        P = (Kmat @ B.T) @ B; out = {}
        for i_, k in enumerate(ALL):
            v = kappa[k].clone(); v[1:512] = P[i_].float(); out[k] = v
        return out

    ladder = {}
    for kk in KS:
        prog["kap"] = project(Vh[:kk]); c_, fw = ce_rows(ev); forwards += fw; ladder[str(kk)] = {"energy": float(energy[kk - 1]), "cost": c_ - native, "over_kernels_only": c_ - c_ko}
        print(f"k={kk}: energy {float(energy[kk - 1]):.4f} | kernels-only-with-k-shapes cost {c_ - native:+.4f} (over kernels-only {c_ - c_ko:+.4f})")
    dd = torch.arange(1, 512, dtype=torch.float64, device=dev); named = [torch.exp(-dd / 6), torch.exp(-dd / 3), dd ** -0.5, dd ** -1.0]
    Bn = torch.stack([v / v.norm() for v in named]); Qn, _ = torch.linalg.qr(Bn.T); prog["kap"] = project(Qn.T); c_named, fw = ce_rows(ev); forwards += fw
    e_named = float(((Kmat @ Qn) ** 2).sum() / (Kmat ** 2).sum())
    print(f"native {native:.5f} | kernels-only (closed form) {c_ko - native:+.4f} | named-4 basis: energy {e_named:.4f}, cost {c_named - native:+.4f} | SVD energy cum (1..8) {[round(float(energy[i]), 3) for i in range(8)]}")
    state["_restore"]()
    disk_guard.guard_torch_save({"means": {f"{k[0]}.{k[1]}": means[k].cpu() for k in ALL}, "kappa": {f"{k[0]}.{k[1]}": kappa[k].cpu() for k in ALL}, "svd_S": S.cpu(), "svd_Vh": Vh[:16].cpu()}, str(OUT_PT), "v751 gpt2-small means and kernels")
    predictions = {"pred_a_instrument_replays": abs(native - native_stock) <= REPLAY_TOL, "pred_b_joint_value_comparable": JOINT_RANGE[0] <= joint <= JOINT_RANGE[1],
                   "pred_c_kernels_only_cost": (c_ko - native) <= KO_MAX, "pred_d_kernel_energy_top4": float(energy[3]) >= E4_MIN, "pred_e_four_shapes_cheap": ladder["4"]["over_kernels_only"] <= K4_TOL}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "gpt2_values_kernels_result_v751", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native_stock": native_stock, "native": native, "mean_ablation_cost": value, "median_head": median, "joint_value": joint, "kernels_only_cost": c_ko - native, "kernels_only_col0_native_cost": c_ko0 - native,
                                          "svd_energy_cum": [float(energy[i]) for i in range(16)], "ladder": ladder, "named4": {"energy": e_named, "cost": c_named - native}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
