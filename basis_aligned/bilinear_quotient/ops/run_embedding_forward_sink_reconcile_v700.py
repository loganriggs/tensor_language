#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_two_instruments_agree pred_b_census_rows_position0 pred_c_skip7000_rows_position0 pred_d_mass_share_position0 pred_e_value_norm_position0
"""Embedding-forward lane, v700 (reconciliation): is head 5.7 an attention sink (ledger §432/§451: position 0 is the top read for 99.8% of
queries) or a broad aggregator (v625/v636: argmax at position 0 for 0% of queries, 26% of |mass|)? Same definition (argmax of |pattern| over
keys j < q), different rows and different instruments. This rung computes, for head 5.7 on 16 census rows (census_lib.rows(), T = 256 as the
ledger used) and 16 skip7000 rows (T = 512 as the lane used): (i) the fraction of queries (q = 4, 8, ..., every fourth) whose argmax |pattern|
is position 0, by the MODEL'S OWN pattern (captured inside squared_attention) and by the LEDGER'S recomputation (head_5_7_reads.py's q/k/rotary
arithmetic on the captured attention input); (ii) the |mass| share on position 0; (iii) the value norm at position 0 vs elsewhere (§432: 730 vs
197). Response only. Whichever instrument disagrees with the model's own pattern is the wrong one.
PREDICTIONS (scored as written; failures preserved)
    pred_a_two_instruments_agree  on every row set, the ledger-style recomputation and the model's own pattern give argmax-at-0 fractions within 0.05
    pred_b_census_rows_position0  on census rows (T = 256) the model's-own argmax-at-0 fraction >= 0.9 (the ledger's 99.8%). Prior: unsure
    pred_c_skip7000_rows_position0 on skip7000 rows (T = 512) the fraction <= 0.1 (the lane's 0%). Prior: unsure — if b and c both hold, the row sets differ
    pred_d_mass_share_position0   the |mass| share on position 0 (queries >= 8) on skip7000 rows is within 0.1 of v636's 0.257 (replay)
    pred_e_value_norm_position0   position 0's value norm exceeds the median elsewhere by >= 2x on census rows (§432: 730 vs 197). Prior: likely
PRICE (registered maximum): 4 forwards (16 census rows at T=256 and 16 skip7000 rows at T=512, batch 8); 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, sys, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_sink_reconcile_v700_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
CANDIDATE_ID = "embedding_forward.sink_reconcile_v700"
FORWARDS_MAX = 6
L, H_ = 5, 7
N_ROWS, B = 16, 8
PREDICTIONS = {"pred_a_two_instruments_agree": "within 0.05", "pred_b_census_rows_position0": ">= 0.9", "pred_c_skip7000_rows_position0": "<= 0.1",
               "pred_d_mass_share_position0": "0.257 +-0.1", "pred_e_value_norm_position0": ">= 2x"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "head": [L, H_]}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import census_lib as cl
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    D = model.config.n_embd; H = model.config.n_head; hd = D // H; dev = "cuda"; blk = model.transformer.h[L]; at = blk.attn; forwards = 0
    are = sys.modules[type(at).__module__].apply_rotary_emb
    sets = {"census_T256": cl.rows()[:N_ROWS, :257].long(), "skip7000_T512": torch.load(EVAL_ROWS, map_location="cpu").long()[:N_ROWS]}
    with torch.no_grad():
        cap = {}
        native_sq = at.squared_attention

        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
            cap.setdefault("own", []).append(pat[:, H_].detach().clone()); cap.setdefault("v", []).append(v[:, :, H_].detach().clone())
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)

        at.squared_attention = patched
        pre = at.register_forward_pre_hook(lambda m, a: cap.setdefault("X", []).append(a[0].detach().clone()))
        report = {}
        for name, rows in sets.items():
            cap.clear()
            for s in range(0, N_ROWS, B):
                idx = rows[s:s + B, :-1].to(dev); model(idx, rows[s:s + B, 1:].to(dev)); forwards += 1
            own = torch.cat(cap["own"]); X = torch.cat(cap["X"]); V_ = torch.cat(cap["v"]); Bn, T = own.shape[0], own.shape[-1]
            # ledger-style recomputation (head_5_7_reads.py arithmetic) from the captured attention input
            cos, sin = at.rotary(at.c_q(X).view(Bn, T, H, hd))
            qf = F.rms_norm(at.c_q(X).view(Bn, T, H, hd), (hd,))[:, :, H_]; kf = F.rms_norm(at.c_k(X).view(Bn, T, H, hd), (hd,))[:, :, H_]
            q2 = F.rms_norm(at.c_q2(X).view(Bn, T, H, hd), (hd,))[:, :, H_]; k2 = F.rms_norm(at.c_k2(X).view(Bn, T, H, hd), (hd,))[:, :, H_]
            qf = are(qf[:, :, None], cos, sin)[:, :, 0]; kf = are(kf[:, :, None], cos, sin)[:, :, 0]; q2 = are(q2[:, :, None], cos, sin)[:, :, 0]; k2 = are(k2[:, :, None], cos, sin)[:, :, 0]
            led = (torch.einsum("bqd,bkd->bqk", qf.float(), kf.float()) * torch.einsum("bqd,bkd->bqk", q2.float(), k2.float())) * torch.tril(torch.ones(T, T, device=dev))
            qs = list(range(4, T, 4))
            def frac0(P):
                hits = 0; n = 0
                for b in range(Bn):
                    for q in qs:
                        hits += int(P[b, q, :q].abs().argmax()) == 0; n += 1
                return hits / n
            f_own, f_led = frac0(own), frac0(led)
            pos = torch.arange(T, device=dev); off = (pos[:, None] > pos[None, :])[None]; qm = pos >= 8
            m = own.abs() * off; share0 = float(m[:, qm, 0].sum() / m[:, qm].sum())
            vn = V_.norm(dim=-1); v0 = float(vn[:, 0].mean()); vmed = float(vn[:, 1:].median())
            corr = float(torch.corrcoef(torch.stack([own[:, qm][:, :, :].flatten(), led[:, qm].flatten()]))[0, 1])
            report[name] = {"argmax0_model_own": f_own, "argmax0_ledger_recompute": f_led, "mass_share_pos0": share0, "value_norm_pos0": v0, "value_norm_median_other": vmed, "pattern_corr_own_vs_ledger": corr}
            print(f"{name}: argmax-at-0 fraction — model's own pattern {f_own:.3f}, ledger recomputation {f_led:.3f} (pattern corr {corr:.4f}); |mass| share at 0 {share0:.3f}; value norm at 0 {v0:.1f} vs median elsewhere {vmed:.1f}")
        at.squared_attention = native_sq; pre.remove()
    c, s7 = report["census_T256"], report["skip7000_T512"]
    predictions = {"pred_a_two_instruments_agree": all(abs(r["argmax0_model_own"] - r["argmax0_ledger_recompute"]) <= 0.05 for r in report.values()),
                   "pred_b_census_rows_position0": c["argmax0_model_own"] >= 0.9, "pred_c_skip7000_rows_position0": s7["argmax0_model_own"] <= 0.1,
                   "pred_d_mass_share_position0": abs(s7["mass_share_pos0"] - 0.257) <= 0.1, "pred_e_value_norm_position0": c["value_norm_pos0"] >= 2 * c["value_norm_median_other"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_sink_reconcile_result_v700", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
