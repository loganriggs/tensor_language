#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_953_separates_verbs_by_number pred_c_829_separates_verbs_by_number pred_d_3465_493_ignore_verbs pred_e_have_has_pattern_matches_are_is
"""The verb site from single tokens (v346). v345: on natural sentences only unit 953 keeps a number trace at the verb between the noun and the pronoun; v266-v271
named MLP-8 unit 829 as shared with have / has agreement. Each verb token alone through blocks 0-8: plural-agreeing forms (" are", " were", " have", " do",
" go", " say", " make", " take") vs singular-agreeing forms (" is", " was", " has", " does", " goes", " says", " makes", " takes"); read h for 3465 / 493 / 1036 /
829 / 953 / 1030; pooled-std separation of plural-agreeing from singular-agreeing verbs; the six units' noun-axis for reference (v329's singular / plural
noun medians recomputed here). Question: which units read the VERB's own number from its lookup -- the generalised-number carrier 953, the shared unit 829,
or none.
PREDICTIONS (scored as written; failures preserved; priors from v332 / v345 / v266)
    pred_a_closure                          the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_953_separates_verbs_by_number    953 separates plural-agreeing from singular-agreeing verbs by >= 1 pooled std with its noun sign (plural side negative)
    pred_c_829_separates_verbs_by_number    829 separates them by >= 1 pooled std with its noun sign (plural side positive). Prior: unsure.
    pred_d_3465_493_ignore_verbs            3465 and 493 separate them by < 1 pooled std (strictly noun-position lexical detectors)
    pred_e_have_has_pattern_matches_are_is  for every unit, (h(have) - h(has)) has the same sign as (h(are) - h(is))
PRICE (registered maximum): 1 batch of <= 150 single tokens (blocks 0-8) = 1 forward; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/verb_tokens_up_the_chain_v346_result.json"
CANDIDATE_ID = "pronoun_number.verb_tokens_up_the_chain_v346"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_STD = 1e-4, 1.0
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_953_separates_verbs_by_number": ">= 1 std, negative", "pred_c_829_separates_verbs_by_number": ">= 1 std, positive", "pred_d_3465_493_ignore_verbs": "< 1 std x 2", "pred_e_have_has_pattern_matches_are_is": "same sign x 6"}


def main() -> None:
    spec = v287.SPEC; sing = [a for _, a, _ in spec["noun_pairs_lexicon"]][:64]; plur = [b for _, _, b in spec["noun_pairs_lexicon"]][:64]
    def T(xs):
        out = []
        for t in xs:
            try: out.append(L._single(t))
            except Exception: pass                      # multi-token entries are dropped and listed in the plan
        return out
    groups = {"singular": sing, "plural": plur, "verb_plural": T([" are", " were", " have", " do", " go", " say", " make", " take"]), "verb_singular": T([" is", " was", " has", " does", " goes", " says", " makes", " takes"])}
    tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_std": GAP_STD}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    ids = torch.tensor(tokens, device="cuda").unsqueeze(1); blocks = model.transformer.h
    H, H2 = {}, {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l in UNITS:
                hh = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu()
                for u in UNITS[l]: H[u] = hh[:, u]; H2[u] = hh2[:, u]
            if l == TOP: break
            x = x + block.mlp(xin)
        forwards += 1
    closure = max(float(((H2[u] - H[u]).abs() / H[u].abs().clamp_min(1e-6)).max()) for u in H); tindex = {t: i for i, t in enumerate(tokens)}
    per = {}
    for u, h in H.items():
        val = lambda g: torch.tensor([float(h[tindex[t]]) for t in groups[g]])
        s_, p_ = val("singular"), val("plural"); noun_sign = 1.0 if float(p_.median() - s_.median()) > 0 else -1.0
        vp, vs = val("verb_plural"), val("verb_singular"); std = float(torch.cat([vp - vp.mean(), vs - vs.mean()]).std()); sep = float((vp.median() - vs.median()) / std)
        have_has = float(h[tindex[L._single(" have")]] - h[tindex[L._single(" has")]]); are_is = float(h[tindex[L._single(" are")]] - h[tindex[L._single(" is")]])
        per[str(u)] = {"noun_sign_of_plural": noun_sign, "verb_separation_over_std": sep, "verb_separation_signed_by_noun": sep * noun_sign, "have_minus_has": have_has, "are_minus_is": are_is, "verb_plural_median": float(vp.median()), "verb_singular_median": float(vs.median())}
    report = {"closure_max": closure, "per_unit": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_953_separates_verbs_by_number": per["953"]["verb_separation_signed_by_noun"] >= GAP_STD, "pred_c_829_separates_verbs_by_number": per["829"]["verb_separation_signed_by_noun"] >= GAP_STD,
                   "pred_d_3465_493_ignore_verbs": abs(per["3465"]["verb_separation_over_std"]) < GAP_STD and abs(per["493"]["verb_separation_over_std"]) < GAP_STD, "pred_e_have_has_pattern_matches_are_is": all(v["have_minus_has"] * v["are_minus_is"] > 0 for v in per.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "verb_tokens_up_the_chain_result_v346", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
