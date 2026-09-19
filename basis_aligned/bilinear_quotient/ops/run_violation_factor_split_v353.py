#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_one_factor_reads_the_verb pred_c_other_factor_reads_the_noun pred_d_product_reproduces_the_table pred_e_3040_and_114_split_the_same_way
"""The violation leaders' bilinear factors: which reads the verb, which the noun? (v353). v352: MLP-3 units 3040 and 114 read the verb's form from the token
alone and fire at the noun-verb mismatch; 565 amplifies "are" after a singular noun. The 2 x 2 design here is {" is", " are"} x {singular X, plural X} in the
frames "The X verb", read at the verb, over the 256 vocabulary pairs (1,024 rows); for each of 3040 / 114 / 565 / 3465 the exact two-way variance shares
of the L and R factors (verb form, noun number, interaction, residual) and the product-of-medians sign check.
PREDICTIONS (scored as written; failures preserved; priors from v352)
    pred_a_closure                     the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_one_factor_reads_the_verb   for each of 3040 / 114 / 565, one factor has >= 0.50 of its variance explained by verb form with < 0.20 by noun number
    pred_c_other_factor_reads_the_noun for each of the three, the other factor has >= 0.30 by noun number with < 0.30 by verb form. Prior: unsure.
    pred_d_product_reproduces_the_table  the product of the two factors' 2 x 2 median tables matches the unit's table in sign on all four cells, for each of the four units
    pred_e_3040_and_114_split_the_same_way  3040 and 114 have the verb-reading factor on the same side (both L or both R). Prior: unsure.
PRICE (registered maximum): 4 cells x 256 rows / 256 = 4 forwards (blocks 0-3); 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/violation_factor_split_v353_result.json"
CANDIDATE_ID = "pronoun_number.violation_factor_split_v353"
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, NOUN_MIN, NOUN_FRAME_MAX, FRAME_MIN, FRAME_NOUN_MAX = 1e-4, 0.50, 0.20, 0.30, 0.30
FRAMES = (" is", " are"); FOUR = ("mlp3.3040", "mlp3.114", "mlp3.565")
UNITS = {3: (3040, 114, 565, 3465)}; TOP = 3
V333 = {"mlp3.3465": -2.58, "mlp3.493": 2.65, "mlp5.1036": 2.01, "mlp8.829": 1.76, "mlp8.953": -0.06, "mlp8.1030": -0.04}
V333_SIDE = {"mlp3.3465": 0.96, "mlp3.493": 0.98, "mlp5.1036": 0.89, "mlp8.829": 0.82}
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 10
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_one_factor_reads_the_verb": ">= 0.50 verb, < 0.20 noun x 3", "pred_c_other_factor_reads_the_noun": ">= 0.30 noun, < 0.30 verb x 3", "pred_d_product_reproduces_the_table": "4 cells x 4 units", "pred_e_3040_and_114_split_the_same_way": "same side"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "noun_min": NOUN_MIN, "noun_frame_max": NOUN_FRAME_MAX, "frame_min": FRAME_MIN, "frame_noun_max": FRAME_NOUN_MAX}, "frames": list(FRAMES), "frame": "The X verb"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    closure = 0.0; FL, FR = {}, {}          # per (unit, frame): L-factor and R-factor tensors over the 512 tokens (singular forms then plural forms order = tokens list)
    for frame in FRAMES:
        pre = [L._single(frame)]; Hl, Hr, Hl2 = {}, {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                the = L._single("The"); ids = torch.tensor([[the, t] + pre for t in tokens[s0:s0 + 256]], device="cuda")
                x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l in UNITS:
                        lf, rf = block.mlp.Left(xin)[:, 2].float().cpu(), block.mlp.Right(xin)[:, 2].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 2].float().cpu()
                        for u in UNITS[l]: Hl.setdefault((l, u), []).append(lf[:, u]); Hr.setdefault((l, u), []).append(rf[:, u]); Hl2.setdefault((l, u), []).append(hh2[:, u])
                    if l == TOP: break
                    x = x + block.mlp(xin)
                forwards += 1
        for kk in Hl:
            lf, rf, hh2 = torch.cat(Hl[kk]), torch.cat(Hr[kk]), torch.cat(Hl2[kk]); closure = max(closure, float(((lf * rf - hh2).abs() / hh2.abs().clamp_min(1e-6)).max()))
            FL[(kk, frame)] = lf; FR[(kk, frame)] = rf
    tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a_] for a_, _ in pairs]); pi = torch.tensor([tindex[b_] for _, b_ in pairs])
    def anova(get):
        """balanced two-way decomposition over frames x {singular, plural} x 256 pairs: shares of variance for noun form, frame, interaction, residual."""
        cells = {(f, n): get(f)[si if n == "s" else pi] for f in FRAMES for n in ("s", "p")}; allv = torch.cat(list(cells.values())); gm = allv.mean(); sst = ((allv - gm) ** 2).sum()
        mean_n = {n: torch.cat([cells[(f, n)] for f in FRAMES]).mean() for n in ("s", "p")}; mean_f = {f: torch.cat([cells[(f, n)] for n in ("s", "p")]).mean() for f in FRAMES}
        ss_n = sum(((mean_n[n] - gm) ** 2) * 256 * len(FRAMES) for n in ("s", "p")); ss_f = sum(((mean_f[f] - gm) ** 2) * 256 * 2 for f in FRAMES)
        ss_int = sum(((cells[(f, n)].mean() - mean_n[n] - mean_f[f] + gm) ** 2) * 256 for f in FRAMES for n in ("s", "p"))
        return {"noun": float(ss_n / sst), "frame": float(ss_f / sst), "interaction": float(ss_int / sst), "residual": float(1 - (ss_n + ss_f + ss_int) / sst)}
    per = {}
    for l in UNITS:
        for u in UNITS[l]:
            key = f"mlp{l}.{u}"; kk = (l, u)
            aL, aR = anova(lambda f: FL[(kk, f)]), anova(lambda f: FR[(kk, f)])
            tabL = {f: {n: float(FL[(kk, f)][si if n == "s" else pi].median()) for n in ("s", "p")} for f in FRAMES}; tabR = {f: {n: float(FR[(kk, f)][si if n == "s" else pi].median()) for n in ("s", "p")} for f in FRAMES}
            tabH = {f: {n: float((FL[(kk, f)] * FR[(kk, f)])[si if n == "s" else pi].median()) for n in ("s", "p")} for f in FRAMES}
            sign_ok = all((tabL[f][n] * tabR[f][n]) * tabH[f][n] > 0 for f in FRAMES for n in ("s", "p"))
            noun_side = "L" if aL["noun"] >= aR["noun"] else "R"
            per[key] = {"L_variance": aL, "R_variance": aR, "L_medians": tabL, "R_medians": tabR, "h_medians": tabH, "product_sign_matches": sign_ok, "noun_reading_factor": noun_side, "verb_reading_factor": ("L" if aL["frame"] >= aR["frame"] else "R")}
    report = {"closure_max": closure, "per_unit": per}
    print(json.dumps({k: {a_: b_ for a_, b_ in v.items() if a_ in ("L_variance", "R_variance", "product_sign_matches", "noun_reading_factor")} for k, v in per.items()}, indent=1))
    def verb_ok(v):
        vf = v["L_variance"] if v["verb_reading_factor"] == "L" else v["R_variance"]; return vf["frame"] >= 0.50 and vf["noun"] < 0.20
    def noun2_ok(v):
        of = v["R_variance"] if v["verb_reading_factor"] == "L" else v["L_variance"]; return of["noun"] >= 0.30 and of["frame"] < 0.30
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_one_factor_reads_the_verb": all(verb_ok(per[k]) for k in FOUR), "pred_c_other_factor_reads_the_noun": all(noun2_ok(per[k]) for k in FOUR),
                   "pred_d_product_reproduces_the_table": all(per[k]["product_sign_matches"] for k in list(FOUR) + ["mlp3.3465"]), "pred_e_3040_and_114_split_the_same_way": per["mlp3.3040"]["verb_reading_factor"] == per["mlp3.114"]["verb_reading_factor"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "violation_factor_split_result_v353", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
