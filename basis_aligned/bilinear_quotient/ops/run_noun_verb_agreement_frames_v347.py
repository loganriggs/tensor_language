#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_1036_at_verb_reads_verb_form pred_c_1036_at_verb_also_reads_noun_number pred_d_agreement_interaction_present pred_e_3465_at_verb_reads_ending_not_noun
"""At the verb after a noun: verb form, noun number, or their agreement? (v347). v346: from the token alone MLP-5 unit 1036 reads the verb's own number at 7.8
std, 1030 at 2.8, 953 at 1.7; v345: at the verb of natural sentences 1036 separates plural- from singular-cue rows by only 0.29 std. Frames [X, verb] for the
256 vocabulary pairs (X singular or plural) x verb in {" is", " are"}: four cells, 1,024 rows, read at the VERB position for 3465 / 493 / 1036 / 829 / 953 / 1030.
Exact two-way decomposition of each unit's medians: verb-form main effect, noun-number main effect, interaction (agreement: grammatical vs ungrammatical cells).
PREDICTIONS (scored as written; failures preserved; priors from v345 / v346)
    pred_a_closure                            the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_1036_at_verb_reads_verb_form       1036's verb-form main effect (are - is, pooled over noun number) is >= 2 pooled std
    pred_c_1036_at_verb_also_reads_noun_number 1036's noun-number main effect at the verb (plural - singular X, pooled over verb) is >= 0.5 pooled std
    pred_d_agreement_interaction_present      for at least one of the six units the interaction term (grammatical - ungrammatical) is >= 1 pooled std. Prior: unsure.
    pred_e_3465_at_verb_reads_ending_not_noun 3465's verb-form effect exceeds its noun-number effect at the verb (in |std|)
PRICE (registered maximum): 4 cells x 256 rows / 256 = 4 forwards (blocks 0-8); 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/noun_verb_agreement_frames_v347_result.json"
CANDIDATE_ID = "pronoun_number.noun_verb_agreement_frames_v347"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, FORM_MIN, NOUN_MIN, INTER_MIN = 1e-4, 2.0, 0.5, 1.0
VERBS = (" is", " are")
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_1036_at_verb_reads_verb_form": ">= 2 std", "pred_c_1036_at_verb_also_reads_noun_number": ">= 0.5 std", "pred_d_agreement_interaction_present": ">= 1 std for one unit", "pred_e_3465_at_verb_reads_ending_not_noun": "|form| > |noun|"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "form_min": FORM_MIN, "noun_min": NOUN_MIN, "inter_min": INTER_MIN}, "verbs": list(VERBS)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    cells = {}; closure = 0.0
    for vb in VERBS:
        vid = L._single(vb); H, H2 = {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                ids = torch.tensor([[t, vid] for t in tokens[s0:s0 + 256]], device="cuda")
                x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l in UNITS:
                        hh = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu()
                        for u in UNITS[l]: H.setdefault((l, u), []).append(hh[:, u]); H2.setdefault((l, u), []).append(hh2[:, u])
                    if l == TOP: break
                    x = x + block.mlp(xin)
                forwards += 1
        H = {kk: torch.cat(v) for kk, v in H.items()}; H2 = {kk: torch.cat(v) for kk, v in H2.items()}
        closure = max(closure, max(float(((H2[kk] - H[kk]).abs() / H[kk].abs().clamp_min(1e-6)).max()) for kk in H)); tindex = {t: i for i, t in enumerate(tokens)}
        for (l, u), h in H.items():
            cells[(f"mlp{l}.{u}", vb, "s")] = torch.tensor([float(h[tindex[a_]]) for a_, _ in pairs]); cells[(f"mlp{l}.{u}", vb, "p")] = torch.tensor([float(h[tindex[b_]]) for _, b_ in pairs])
    per = {}
    for key in sorted({k[0] for k in cells}):
        c = {(vb, n): cells[(key, vb, n)] for vb in VERBS for n in ("s", "p")}; allv = torch.cat(list(c.values())); std = float(torch.cat([v - v.mean() for v in c.values()]).std())
        med = {f"{vb.strip()}|{n}": float(v.median()) for (vb, n), v in c.items()}
        form = ((med["are|s"] + med["are|p"]) - (med["is|s"] + med["is|p"])) / 2 / std; noun = ((med["is|p"] + med["are|p"]) - (med["is|s"] + med["are|s"])) / 2 / std
        gram = (med["is|s"] + med["are|p"]) / 2; ungram = (med["is|p"] + med["are|s"]) / 2; inter = (gram - ungram) / std
        per[key] = {"medians": med, "pooled_std": std, "verb_form_effect_std": form, "noun_number_effect_std": noun, "agreement_interaction_std": inter}
    report = {"closure_max": closure, "per_unit": per}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_1036_at_verb_reads_verb_form": abs(per["mlp5.1036"]["verb_form_effect_std"]) >= FORM_MIN, "pred_c_1036_at_verb_also_reads_noun_number": abs(per["mlp5.1036"]["noun_number_effect_std"]) >= NOUN_MIN,
                   "pred_d_agreement_interaction_present": any(abs(v["agreement_interaction_std"]) >= INTER_MIN for v in per.values()), "pred_e_3465_at_verb_reads_ending_not_noun": abs(per["mlp3.3465"]["verb_form_effect_std"]) > abs(per["mlp3.3465"]["noun_number_effect_std"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "noun_verb_agreement_frames_result_v347", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
