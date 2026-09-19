#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_vc0_is_case pred_c_circuit_does_not_move_case pred_d_vc2_is_number pred_e_vc2_written_by_readers
"""The other pronoun-class axes: case (VC0) and number-within-pronouns (VC2) (v400). v397-v399: the number circuit's class-internal output lies mostly on VC1
(gendered singular vs the rest). The class's other residual axes: VC0 orders them / us / him / her against they / we / he / she / I (object vs subject case) and VC2
orders they / them / it / he against we / us (third-person plural vs first-person). Here the energies and writers along VC0 and VC2 on panel and text, to say what
the circuit leaves alone (case) and whether its plural signal is third-person-specific.
PREDICTIONS (scored as written; failures preserved; priors from v397 / v398)
    pred_a_closure                 per-writer final-token writes sum to the final residual within relative 1e-4 (instrument), panel and text
    pred_b_vc0_is_case             along VC0 the object pronouns them / us / him / her all lie on one side of the subject pronouns they / we / he / she / I
    pred_c_circuit_does_not_move_case  the final-state contrast's energy along VC0 is <= 0.05 of its energy along u, on panel and text
    pred_d_vc2_is_number           along VC2 they and them lie on the opposite side from we and us (third vs first person plural)
    pred_e_vc2_written_by_readers  the readers 9.6 / 12.4 / 15.1 carry >= 0.35 of the summed |contrast| along VC2 on the panel. Prior: unsure.
PRICE (registered maximum): 2 natural batches + 3 panel batches = 5 forwards; 0 backwards; 0 fits. Bar <= 7.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/subspace_vc0_vc2_v400_result.json"
CANDIDATE_ID = "subspace.vc0_vc2_v400"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, CASE_MAX, READER_MIN = 1e-4, 0.05, 0.35
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 7
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_vc0_is_case": "object vs subject", "pred_c_circuit_does_not_move_case": "<= 0.05 of u x 2", "pred_d_vc2_is_number": "they/them vs we/us", "pred_e_vc2_written_by_readers": ">= 0.35"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "case_max": CASE_MAX, "reader_min": READER_MIN}, "pronouns": list(PRONOUNS), "natural_rows": len(recs)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float().cpu()
    P = torch.tensor([b for _, b in pairs]); S = torch.tensor([a for a, _ in pairs]); UP, US = WU[P], WU[S]; mP, mS = UP.mean(0), US.mean(0); c = mP - mS
    RP = UP - mP; svP = torch.linalg.svd(RP, full_matrices=False); Vk = svP.Vh[:K]; RS = US - mS; svS = torch.linalg.svd(RS, full_matrices=False); VkS = svS.Vh[:K]
    mean_energy = float((mP.norm() ** 2) * len(P)); resid_topk_energy = float((svP.S[:K] ** 2).sum()); resid_total_energy = float((svP.S ** 2).sum())
    u = WU[L._single(" they")] - WU[L._single(" he")]; cos_cu = float(torch.nn.functional.cosine_similarity(c, u, dim=0))
    # fold back at the final token: per-writer writes (embedding, attention heads of blocks 9-15 per head, all MLPs, other attention blocks as totals)
    def capture(batches):
        W = {}; XF = []
        def add(k, v): W.setdefault(k, []).append(v)
        with torch.no_grad():
            for tokens, pf in batches:
                idx = torch.arange(tokens.shape[0]); x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0
                    for k in parts: parts[k] = block.lambdas[0] * parts[k]
                    parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                    if 9 <= l <= 15:
                        captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(xin_a, v1_)
                        finally: hook.remove()
                        Wp = block.attn.c_proj.weight.detach().float(); y = captured["y"].float()
                        for h in range(N_HEAD): parts[f"{l}.{h}"] = y[:, :, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T
                    else:
                        attention, v1_ = block.attn(xin_a, v1_); parts[f"attn{l}"] = attention.clone()
                    x = live + attention; m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
                for k, v in parts.items(): add(k, v[idx, pf].float().cpu())
                XF.append(x[idx, pf].float().cpu())
        W = {k: torch.cat(v) for k, v in W.items()}; XF = torch.cat(XF); return W, XF, float(((sum(W.values()) - XF).norm(dim=1) / XF.norm(dim=1)).max())
    nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); fin = [len(r_["ids"]) - 1 for r_ in recs]
    Wt, XFt, c1 = capture([(nat[s0:s0 + 64], torch.tensor(fin[s0:s0 + 64])) for s0 in range(0, len(recs), 64)]); forwards += 2
    Wp_, XFp, c2 = capture([(fw._tokens(rows[s0:s0 + BATCH]), torch.tensor([r_.final for r_ in rows[s0:s0 + BATCH]])) for s0 in range(0, len(rows), BATCH)]); forwards += 3; closure = max(c1, c2)
    tp = [i for i, r_ in enumerate(recs) if r_["label"] == "they"]; th = [i for i, r_ in enumerate(recs) if r_["label"] != "they"]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pp = [i for i, r_ in enumerate(rows) if r_.present]; ps = [partner[(rows[i].construction, rows[i].group, False)] for i in pp]
    un = u / u.norm(); C = torch.tensor([L._single(t) for t in PRONOUNS]); UC = WU[C]; mC = UC.mean(0); VC = torch.linalg.svd(UC - mC, full_matrices=False).Vh[:8]; mCn = mC / mC.norm()
    vc0, vc2 = VC[0], VC[2]; dt = XFt[tp].mean(0) - XFt[th].mean(0); dp = (XFp[pp] - XFp[ps]).mean(0)
    order0 = {p.strip(): float(WU[c] @ vc0) for p, c in zip(PRONOUNS, C)}; order2 = {p.strip(): float(WU[c] @ vc2) for p, c in zip(PRONOUNS, C)}
    obj = ["them", "us", "him", "her"]; subj = ["they", "we", "he", "she", "I"]; case_ok = max(order0[k] for k in subj) < min(order0[k] for k in obj) or min(order0[k] for k in subj) > max(order0[k] for k in obj)
    third = [order2["they"], order2["them"]]; first = [order2["we"], order2["us"]]; num_ok = max(third) < min(first) or min(third) > max(first)
    e = lambda d, v: float((d @ v) ** 2)
    def writers(W_, a_, b_, d, paired):
        if paired: return {k: float(((W_[k][a_] - W_[k][b_]).mean(0)) @ d) for k in W_}
        return {k: float((W_[k][a_].mean(0) - W_[k][b_].mean(0)) @ d) for k in W_}
    c2p = writers(Wp_, pp, ps, vc2, True); share = lambda c_: sum(abs(c_[k]) for k in READERS) / sum(abs(v) for v in c_.values()); top = lambda c_: sorted(c_, key=lambda k: -abs(c_[k]))[:8]
    report = {"closure_max": closure, "vc0_order": dict(sorted(order0.items(), key=lambda kv: kv[1])), "vc2_order": dict(sorted(order2.items(), key=lambda kv: kv[1])), "case_separated": bool(case_ok), "third_vs_first_separated": bool(num_ok),
              "energy": {"panel": {"vc0": e(dp, vc0), "vc2": e(dp, vc2), "u": e(dp, un)}, "text": {"vc0": e(dt, vc0), "vc2": e(dt, vc2), "u": e(dt, un)}}, "panel_writers_vc2_top8": {k: c2p[k] for k in top(c2p)}, "reader_share_vc2_panel": share(c2p)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_vc0_is_case": bool(case_ok), "pred_c_circuit_does_not_move_case": report["energy"]["panel"]["vc0"] <= CASE_MAX * report["energy"]["panel"]["u"] and report["energy"]["text"]["vc0"] <= CASE_MAX * report["energy"]["text"]["u"], "pred_d_vc2_is_number": bool(num_ok), "pred_e_vc2_written_by_readers": share(c2p) >= READER_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vc0_vc2_result_v400", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
