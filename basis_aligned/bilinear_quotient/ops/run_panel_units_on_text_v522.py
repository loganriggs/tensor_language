#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_units_fire_less_with_number_on_text pred_c_pooled_contrast_small_on_text pred_d_terms_against_they_small_on_text pred_e_3547_column_replays
"""Do the panel's late units fire with number at natural pronoun slots (v522)? v521: on the panel units 3547 / 1747 / 4448 fire more on plural rows (+32 to +53)
and 3093 less (-30), each pushing against 'they'; v517: their swap does nothing on natural text. Here the same census at the natural pronoun answer (122 pairs):
their activation contrasts and terms against 'they' there, and MLP 17's pooled they - he contrast at the natural answer.
PREDICTIONS (scored as written; failures preserved; priors from v504 / v517 / v521)
    pred_a_unit_closure                     per-unit terms sum to MLP 17's term within relative 1e-3 on every row
    pred_b_units_fire_less_with_number_on_text  for each of the four units |activation contrast| on text is <= 0.5 x its panel value (53 / 30 / 32 / 40)
    pred_c_pooled_contrast_small_on_text    |MLP 17's pooled they - he contrast at the natural answer| <= 10 (panel: -25.3)
    pred_d_terms_against_they_small_on_text  each unit's |term against they| on text is <= 0.5 x its panel value (61 / 45 / 46 / 44)
    pred_e_3547_column_replays              cos(Down[:, 3547], W_U[he] - class mean) replays 0.526 +- 0.01 (weights; sanity)
PRICE (registered maximum): 4 text batches = 4 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/panel_units_on_text_v522_result.json"
CANDIDATE_ID = "chain.panel_units_on_text_v522"
N_HEAD, LAYER = 9, 17
CLOSURE_TOL, POOLED_MAX, HALF = 1e-3, 10.0, 0.5
PANEL_ACT = {1747: 53.226, 3093: -29.842, 3547: 32.33, 4448: 40.481}; PANEL_AGAINST = {1747: -61.05, 3093: -44.98, 3547: -45.59, 4448: -44.13}
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
FOUR = (3547, 1747, 3093, 4448)
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_units_fire_less_with_number_on_text": "<= 0.5 x panel, x 4", "pred_c_pooled_contrast_small_on_text": "|.| <= 10", "pred_d_terms_against_they_small_on_text": "<= 0.5 x panel, x 4", "pred_e_3547_column_replays": "0.526 +- 0.01"}


def main() -> None:
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]]; E = L.ENCODING
    def partner(tok):
        t = E.decode([tok])
        for c in (t + "s", t + "es", t[:-1] if t.endswith("s") else None, t[:-2] if t.endswith("es") else None, (t[:-3] + "y") if t.endswith("ies") else None, (t[:-1] + "ies") if t.endswith("y") else None):
            if c and c != t and len(E.encode(c)) == 1: return E.encode(c)[0]
        return None
    items = []
    for r_ in recs:
        c = r_["cue_offset"]; alt = partner(r_["ids"][c])
        if alt is None: continue
        sw = list(r_["ids"]); sw[c] = alt; plural, singular = (r_["ids"], sw) if r_["cue"] == "plural" else (sw, r_["ids"])
        items.append((list(plural), list(singular), len(plural) - 1))              # the natural pronoun answer position
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(items), "layer": LAYER, "reader": "direct (W_U), natural answer position", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "pooled_max": POOLED_MAX, "half": HALF, "four": list(FOUR)}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float(); u = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()
    m = (WU[L._single(" they")] - WU[L._single(" he")]).cuda()   # the pronoun direction on the final residual at the answer position
    mlp = blocks[LAYER].mlp; Dw = mlp.Down.weight.detach().float(); mD = m @ Dw                                     # (4608,)
    seqs = [p for p, _, _ in items] + [s for _, s, _ in items]; pos = [c for _, _, c in items] * 2; n = len(items)
    per_row_units, per_row_mlp, per_row_h, forwards = [], [], [], 0
    with torch.no_grad():
        for s0 in range(0, len(seqs), 64):
            tokens = torch.tensor([q + [0] * (max(len(t) for t in seqs[s0:s0 + 64]) - len(q)) for q in seqs[s0:s0 + 64]], device="cuda"); pp = torch.tensor(pos[s0:s0 + 64]); idx = torch.arange(tokens.shape[0])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; lam_after = 1.0
            for l, block in enumerate(list(blocks) + [None]):
                if block is None: rms = x[idx, pp].float().pow(2).mean(1).sqrt(); break
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                if l == 18: break
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin); h = (F.silu(Lx) * Rx) if model.config.gated else (Lx * Rx); h8 = h[idx, pp].float(); m8 = mlp(xin)[idx, pp].float()
                x = x + block.mlp(xin)
            # the lambda chain from block 8's output to block 9's live: block 9's lambdas[0] (blocks 9 only, since MLP 8 is added after block 8's live)
            lam = 1.0
            for l_ in range(LAYER + 1, 18): lam *= float(blocks[l_].lambdas[0])          # the lambda chain from block 8 output to the final residual
            per_row_units.append((lam * mD.cpu() * h8.cpu()) / rms.cpu()[:, None]); per_row_h.append(h8.cpu()); per_row_mlp.append((lam * (m8 @ m).cpu() - lam * float(m @ mlp.Down_bias.detach().float())) / rms.cpu())   # module output minus its bias = sum of unit terms
            forwards += 1
    U = torch.cat(per_row_units); M = torch.cat(per_row_mlp)
    closure = float(((U.sum(1) - M).abs() / M.abs().clamp_min(1e-6)).max())
    total = (U[:n] - U[n:]).sum(0); contrast = float(total.sum()); order = torch.argsort(total.abs(), descending=True)
    rank = {j: int((total.abs() > abs(total[j])).sum()) + 1 for j in (829, 953, 1030, 3152, 3943)}
    prev = json.loads((ROOT / "circuits/followups/value_copy_mlp8_units_text_v408_result.json").read_text())["report"]["top_units"][:20]; overlap = len({int(j) for j, _ in prev} & set(order[:20].tolist()))
    share = lambda k: float(total[order[:k]].sum()) / contrast
    report = {"closure_max": closure, "mlp8_contrast": contrast, "shares": {str(k): share(k) for k in (10, 20, 50, 100, 200)}, "top_units": [(int(j), float(total[j])) for j in order[:30]], "ranks": rank, "top20_overlap_with_v408": overlap}
    print(json.dumps(report, indent=1))
    top = int(order[0]); H = torch.cat(per_row_h); dtop = H[:n, top] - H[n:, top]; sign_const = float(max((dtop > 0).float().mean(), (dtop < 0).float().mean()))
    WUc = WU.cpu(); cm = WUc[torch.tensor([L._single(t) for t in PRONOUNS])].mean(0); Dc = Dw.cpu()
    tok = {t: WUc[L._single(" " + t)] - cm for t in ("he", "they", "she")}
    cos = lambda j, t: float(Dc[:, j] @ tok[t] / (Dc[:, j].norm() * tok[t].norm()))
    cols = {j: {t: cos(j, t) for t in tok} for j in FOUR}
    H = torch.cat(per_row_h); act = {j: float((H[:n, j] - H[n:, j]).mean()) for j in FOUR}
    against_they = {j: act[j] * float(Dc[:, j] @ tok["they"]) for j in FOUR}
    report["column_cosines"] = cols; report["activation_contrast"] = act; report["term_against_they"] = against_they
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_units_fire_less_with_number_on_text": all(abs(act[j]) <= HALF * abs(PANEL_ACT[j]) for j in FOUR), "pred_c_pooled_contrast_small_on_text": abs(contrast) <= POOLED_MAX,
                   "pred_d_terms_against_they_small_on_text": all(abs(against_they[j]) <= HALF * abs(PANEL_AGAINST[j]) for j in FOUR), "pred_e_3547_column_replays": abs(cols[3547]["he"] - 0.526) <= 0.01}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "panel_units_on_text_result_v522", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
