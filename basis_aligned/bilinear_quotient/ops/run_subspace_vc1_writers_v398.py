#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_vc1_writers_are_the_readers pred_c_vc1_writers_replay_on_text pred_d_vc1_component_orthogonal_to_u_is_written_by_readers pred_e_vc1_separates_plural_pronouns
"""The pronoun-internal axis VC1: who writes it, and what it is (v398). v397: within the pronoun class's unembedding residual, the second direction VC1 (cos 0.45
with u_they - u_he) carries the most of the number circuit's class-internal output on panel and text. Here: (i) the pronoun ordering along VC1 (which pronouns it
separates); (ii) writers along VC1 and along its component orthogonal to u (VC1 - (VC1.u)u, normalised) at the final token, plural - singular, panel and text;
(iii) whether the readers 9.6 / 12.4 / 15.1 carry it as they carry u.
PREDICTIONS (scored as written; failures preserved; priors from v380 / v397)
    pred_a_closure                     per-writer final-token writes sum to the final residual within relative 1e-4 (instrument), panel and text
    pred_b_vc1_writers_are_the_readers the readers 9.6 / 12.4 / 15.1 carry >= 0.35 of the summed |contrast| along VC1 on the panel
    pred_c_vc1_writers_replay_on_text  the same three heads are in the top 5 along VC1 on text
    pred_d_vc1_component_orthogonal_to_u_is_written_by_readers  along VC1's u-orthogonal component the readers still carry >= 0.25 (the pronoun circuit writes more than they - he). Prior: unsure.
    pred_e_vc1_separates_plural_pronouns  along VC1 the plural pronouns (they, we, them, us) all lie on one side of the singular ones (he, she, it, him, her, I, you) -- i.e. it is number, not case or person
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
OUT = ROOT / "circuits/followups/subspace_vc1_writers_v398_result.json"
CANDIDATE_ID = "subspace.vc1_writers_v398"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, READER_MIN, ORTH_MIN = 1e-4, 0.35, 0.25
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 7
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_vc1_writers_are_the_readers": ">= 0.35", "pred_c_vc1_writers_replay_on_text": "top 5 x 3", "pred_d_vc1_component_orthogonal_to_u_is_written_by_readers": ">= 0.25", "pred_e_vc1_separates_plural_pronouns": "plural on one side"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "reader_min": READER_MIN, "orth_min": ORTH_MIN}, "pronouns": list(PRONOUNS), "natural_rows": len(recs)}
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
    vc1 = VC[1]; orth = vc1 - float(vc1 @ un) * un; orth = orth / orth.norm()
    order = {p.strip(): float(WU[c] @ vc1) for p, c in zip(PRONOUNS, C)}; plural_set = {"they", "we", "them", "us"}
    pl_vals = [v for k, v in order.items() if k in plural_set]; sg_vals = [v for k, v in order.items() if k not in plural_set]; separated = max(pl_vals) < min(sg_vals) or min(pl_vals) > max(sg_vals)
    def writers(W_, a_, b_, d, paired):
        if paired: return {k: float(((W_[k][a_] - W_[k][b_]).mean(0)) @ d) for k in W_}
        return {k: float((W_[k][a_].mean(0) - W_[k][b_].mean(0)) @ d) for k in W_}
    cp, ct = writers(Wp_, pp, ps, vc1, True), writers(Wt, tp, th, vc1, False); co = writers(Wp_, pp, ps, orth, True)
    share = lambda c_: sum(abs(c_[k]) for k in READERS) / sum(abs(v) for v in c_.values()); top = lambda c_: sorted(c_, key=lambda k: -abs(c_[k]))[:8]
    report = {"closure_max": closure, "vc1_pronoun_order": dict(sorted(order.items(), key=lambda kv: kv[1])), "plural_pronouns_separated": separated, "cos_vc1_u": float(vc1 @ un),
              "panel_writers_vc1_top8": {k: cp[k] for k in top(cp)}, "text_writers_vc1_top8": {k: ct[k] for k in top(ct)}, "panel_writers_orth_top8": {k: co[k] for k in top(co)}, "reader_share": {"panel_vc1": share(cp), "text_vc1": share(ct), "panel_orth": share(co)}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_vc1_writers_are_the_readers": share(cp) >= READER_MIN, "pred_c_vc1_writers_replay_on_text": all(k in top(ct)[:5] for k in READERS), "pred_d_vc1_component_orthogonal_to_u_is_written_by_readers": share(co) >= ORTH_MIN, "pred_e_vc1_separates_plural_pronouns": bool(separated)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vc1_writers_result_v398", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
