#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: 8forwards2304seq; bracket rank2 keys times rank2 payload;3fits0backwards0updates.
"""Remove donor-state access using type key prototypes and the live recipient query."""
from __future__ import annotations
from collections import defaultdict
import hashlib, json, math, signal, sys, time
from pathlib import Path

RUNNER = Path(__file__).resolve(); OPS = RUNNER.parent; ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
TRAIN_ROWS = POLY / "BRACKET_NESTED_PENDING_OOD_V1_ROWS.json"
EVAL_ROWS = POLY / "BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json"
BUILDER = POLY / "build_bracket_layered_pending_ood_v1_rows.py"
PRIOR = POLY / "BRACKET_TRIPLE_PENDING_PAYLOAD_RANK2_INTERACTION_V2_RESULT.json"
PREREG = POLY / "BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_PREREGISTRATION.md"
BINDING = POLY / "BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_BINDING.json"
OUT = POLY / "BRACKET_LAYERED_PENDING_KEY_RANK2_INTERACTION_V1_RESULT.json"
TYPES = ("parenthesis", "square", "quote"); HEAD = 8; HEAD_D = 128
PRICE = {"forwards": 8, "sequences": 2304, "fits": 3, "backwards": 0, "updates": 0}
BARS = {
    "replay_max": 1e-5, "capability_accuracy_min": .75, "exact_positive_min": .90,
    "key_cosine_min": .90, "key_relative_l2_max": .50,
    "factor_cosine_min": .80, "factor_relative_l2_max": .60, "factor_sign_min": .85,
    "effect_cosine_min": .90, "effect_relative_l2_max": .40, "effect_sign_min": .90,
    "effect_norm_ratio_min": .60, "effect_norm_ratio_max": 1.40,
    "pair_cosine_min": .75, "pair_relative_l2_max": .50, "pair_sign_min": .85,
    "recipient_score_improvement_min": .10, "control_to_target_rms_max": .50,
}
PREDICTION_REGISTRY = {
    "pred_a_exact_instrument_and_capability": None,
    "pred_b_exact_joint_parent_live": None,
    "pred_c_rank2_key_sources_transfer": None,
    "pred_d_live_query_score_transfers": None,
    "pred_e_donor_free_interaction_transfers": None,
    "pred_f_key_substitution_beats_recipient_score": None,
    "pred_g_control_selectivity": None,
}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vector_metrics(actual, predicted):
    actual = [float(x) for x in actual]; predicted = [float(x) for x in predicted]
    dot = sum(a*b for a, b in zip(actual, predicted)); an = math.sqrt(sum(a*a for a in actual)); pn = math.sqrt(sum(p*p for p in predicted))
    return {"count": len(actual), "cosine": dot/max(an*pn, 1e-30),
            "relative_l2_error": math.sqrt(sum((a-p)**2 for a, p in zip(actual, predicted)))/max(an, 1e-30),
            "sign_agreement": sum((a > 0) == (p > 0) for a, p in zip(actual, predicted))/len(actual),
            "predicted_to_actual_norm_ratio": pn/max(an, 1e-30)}


def low_rank_type_table(values, torch):
    prototypes = torch.stack([torch.stack(values[name]).double().mean(0) for name in TYPES])
    mean = prototypes.mean(0); centered = prototypes - mean
    _left, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]; coefficients = centered @ basis.T
    rank = int((singular > singular[0] * 1e-10).sum().cpu())
    return {"mean": mean, "basis": basis, "coefficients": coefficients,
            "singular_values": singular, "rank": rank}


def score_from_query_keys(q1, q2, k1, k2):
    return ((q1*k1).sum(-1)/HEAD_D) * ((q2*k2).sum(-1)/HEAD_D)


def main():
    sys.path.insert(0, str(OPS))
    from circuit_exactness_preflight import managed_execution_mode, validate_literal_prediction_registry, validate_result_contract
    binding = json.loads(BINDING.read_text())
    paths = {"train_rows": TRAIN_ROWS, "eval_rows": EVAL_ROWS, "builder": BUILDER,
             "prior_result": PRIOR, "preregistration": PREREG}
    assert all(digest(paths[name]) == expected for name, expected in binding["files"].items())
    train = json.loads(TRAIN_ROWS.read_text()); evaluate = json.loads(EVAL_ROWS.read_text())
    assert canonical(train["rows"]) == train["row_manifest_sha256"] and canonical(evaluate["rows"]) == evaluate["row_manifest_sha256"]
    assert binding["bars"] == BARS and binding["price"] == PRICE
    validate_literal_prediction_registry(RUNNER.read_text(), PREDICTION_REGISTRY)
    if managed_execution_mode(__import__("os").environ) == "preflight":
        print(json.dumps({"dryrun": True, "model_loaded": False, "gpu_accessed": False,
                          "train_rows": train["row_count"], "eval_rows": evaluate["row_count"],
                          "arms": ["native", "replay", "exact", "payload_live", "key1", "key2", "both", "recipient_score"],
                          "bars": BARS, "price": PRICE, "predicates": list(PREDICTION_REGISTRY)}, sort_keys=True)); return
    assert not OUT.exists(); signal.alarm(600)
    import torch
    import run_bracket_l13h8_source_region_payload_factorial as exact
    from circuit_fast_screen_managed_runner import atomic_create_json
    torch.set_num_threads(2); tm, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    tagged = [("train", row) for row in train["rows"]] + [("eval", row) for row in evaluate["rows"]]
    endpoints = [(split, row, side) for split, row in tagged for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for _split, row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device="cuda")
    finals, sources = [], []
    for i, (_split, row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]; tokens[i, :len(ids)] = torch.tensor(ids, device="cuda")
        finals.append(len(ids)-1); sources.append(row[f"{side}_open_position"])
    finals_t = torch.tensor(finals, device="cuda"); sources_t = torch.tensor(sources, device="cuda")
    ar = torch.arange(len(endpoints), device="cuda"); counts = [0, 0]

    def count(_module, args, _output): counts[0] += 1; counts[1] += len(args[0])
    handle = model.transformer.h[0].attn.register_forward_hook(count); tic = time.perf_counter()

    def extended_factor_forward():
        captured = {}
        def attention(event):
            if event.site != 13: return event.block.attn(event.state, event.first_value)
            state, first, attn = event.state, event.first_value, event.block.attn
            batch, seq, width = state.shape; heads = 9
            q = exact._linear(state, attn.c_q.weight, F).view(batch, seq, heads, HEAD_D)
            k = exact._linear(state, attn.c_k.weight, F).view(batch, seq, heads, HEAD_D)
            q2 = exact._linear(state, attn.c_q2.weight, F).view(batch, seq, heads, HEAD_D)
            k2 = exact._linear(state, attn.c_k2.weight, F).view(batch, seq, heads, HEAD_D)
            raw = exact._linear(state, attn.c_v.weight, F).view(batch, seq, heads, HEAD_D)
            value = (1-attn.lamb)*raw + attn.lamb*first.view_as(raw)
            cos, sin = attn.rotary(q); rotate = sys.modules[type(attn).__module__].apply_rotary_emb
            qn, kn = F.rms_norm(q, (HEAD_D,)), F.rms_norm(k, (HEAD_D,))
            q2n, k2n = F.rms_norm(q2, (HEAD_D,)), F.rms_norm(k2, (HEAD_D,))
            qr, kr = rotate(qn, cos, sin), rotate(kn, cos, sin)
            q2r, k2r = rotate(q2n, cos, sin), rotate(k2n, cos, sin)
            score1 = torch.einsum("bqhd,bkhd->bhqk", qr, kr)/HEAD_D
            score2 = torch.einsum("bqhd,bkhd->bhqk", q2r, k2r)/HEAD_D
            pattern = (score1*score2).masked_fill(~torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=state.device)), 0)
            all_heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
            write = exact._linear(all_heads.transpose(1,2).contiguous().view(batch,seq,width), attn.c_proj.weight, F)
            weight = attn.c_proj.weight[:, HEAD*HEAD_D:(HEAD+1)*HEAD_D]
            u = exact._linear(value[:,:,HEAD].float(), weight.float(), F)
            p = pattern[ar, HEAD, finals_t]; head = torch.einsum("bk,bkd->bd", p.float(), u)
            captured.update({"p": p.float().detach(), "u": u.detach(), "head": head.detach(),
                             "q1": qr[ar, finals_t, HEAD].detach(), "q2": q2r[ar, finals_t, HEAD].detach(),
                             "k1_pre": kn[ar, sources_t, HEAD].detach(), "k2_pre": k2n[ar, sources_t, HEAD].detach(),
                             "s1": score1[ar, HEAD, finals_t, sources_t].float().detach(),
                             "s2": score2[ar, HEAD, finals_t, sources_t].float().detach(),
                             "cos": cos.detach(), "sin": sin.detach(), "rotate": rotate})
            return write, first
        logits = facade.forward_with_dispatch(model, tokens, attention, lambda event: event.block.mlp(event.state), require_production=False).float()
        return logits, captured

    try:
        with torch.inference_mode():
            native = exact.native_logits(model, tokens, tm, F).cpu()
            replay_gpu, factors = extended_factor_forward(); replay = replay_gpu.cpu(); del replay_gpu; torch.cuda.empty_cache()
            value_tables = {name: defaultdict(list) for name in ("payload", "key1", "key2")}
            for i, (split, row, side) in enumerate(endpoints):
                if split == "train" and row["program_role"] == "target":
                    delimiter = row[f"{side}_type"]
                    value_tables["payload"][delimiter].append(factors["u"][i, sources_t[i]])
                    value_tables["key1"][delimiter].append(factors["k1_pre"][i])
                    value_tables["key2"][delimiter].append(factors["k2_pre"][i])
            tables = {name: low_rank_type_table(values, torch) for name, values in value_tables.items()}
            p, u, s1, s2 = factors["p"][ar, sources_t], factors["u"][ar, sources_t], factors["s1"], factors["s2"]
            donor = ar ^ 1; pd, ud, a_d, b_d = p[donor], u[donor], s1[donor], s2[donor]
            du_hat = torch.zeros_like(u); key1_hat = factors["k1_pre"].clone(); key2_hat = factors["k2_pre"].clone()
            for i, (split, row, side) in enumerate(endpoints):
                if split != "eval": continue
                if row["program_role"] == "target":
                    other = "donor" if side == "base" else "base"; recipient_type = row[f"{side}_type"]; donor_type = row[f"{other}_type"]
                else: recipient_type = donor_type = row["inner_type"]
                ri, di = TYPES.index(recipient_type), TYPES.index(donor_type)
                payload = tables["payload"]; du_hat[i] = ((payload["coefficients"][di]-payload["coefficients"][ri]) @ payload["basis"]).to(u.dtype)
                for name, target in (("key1", key1_hat), ("key2", key2_hat)):
                    table = tables[name]; target[i] = (table["mean"] + table["coefficients"][di] @ table["basis"]).to(target.dtype)
            cos_at = factors["cos"][0, sources_t, 0][:,None,None,:]; sin_at = factors["sin"][0, sources_t, 0][:,None,None,:]
            rotate = factors["rotate"]
            key1_rot = rotate(key1_hat[:,None,None,:], cos_at, sin_at)[:,0,0]
            key2_rot = rotate(key2_hat[:,None,None,:], cos_at, sin_at)[:,0,0]
            a_hat = (factors["q1"]*key1_rot).sum(-1)/HEAD_D; b_hat = (factors["q2"]*key2_rot).sum(-1)/HEAD_D
            p_hat = score_from_query_keys(factors["q1"], factors["q2"], key1_rot, key2_rot)
            payload_hat = u + du_hat
            replacements = {
                "exact": pd[:,None]*ud, "payload_live": pd[:,None]*payload_hat,
                "key1": (a_hat*b_d)[:,None]*payload_hat, "key2": (a_d*b_hat)[:,None]*payload_hat,
                "both": p_hat[:,None]*payload_hat, "recipient_score": p[:,None]*payload_hat,
            }
            arms = {}
            for name, replacement in replacements.items():
                logits, unused = exact.factor_forward(model, tokens, finals_t, {}, tm, F, facade,
                                                      replacement_terms=replacement, source_positions=sources_t)
                arms[name] = logits.cpu(); del logits, unused; torch.cuda.empty_cache()
    finally: handle.remove()
    replay_error = max(float((native[i, finals[i]]-replay[i, finals[i]]).abs().max()) for i in range(len(endpoints)))
    records, capability_cells = [], defaultdict(list); source_vectors = defaultdict(lambda: [[], []]); factor_values = defaultdict(lambda: [[], []])
    for i, (split, row, side) in enumerate(endpoints):
        if split != "eval": continue
        answer = int(row[f"{side}_answer_id"]); other = "donor" if side == "base" else "base"
        capkey = (row["program_role"], answer, side) if row["program_role"] == "control" else ("target", answer, int(row[f"{other}_answer_id"]))
        capability_cells[capkey].append(float(exact.closer_margin(native[i, finals[i]], answer)))
        rec = {"row_id": row["row_id"], "side": side, "program_role": row["program_role"]}
        if row["program_role"] == "target":
            direction = "base_to_donor" if side == "base" else "donor_to_base"; rec["ordered_pair"] = f"{answer}->{int(row[f'{other}_answer_id'])}"
            for name in replacements: rec[name+"_effect"] = float(exact.endpoint_change(replay[i,finals[i]], arms[name][i,finals[i]], row, direction))
            for name, actual, predicted in (("key1", factors["k1_pre"][donor[i]], key1_hat[i]), ("key2", factors["k2_pre"][donor[i]], key2_hat[i])):
                source_vectors[name][0].extend(actual.double().cpu().tolist()); source_vectors[name][1].extend(predicted.double().cpu().tolist())
            for name, actual, predicted in (("factor1", a_d[i], a_hat[i]), ("factor2", b_d[i], b_hat[i]), ("score", pd[i], p_hat[i])):
                factor_values[name][0].append(float(actual.cpu())); factor_values[name][1].append(float(predicted.cpu()))
        else:
            before = float(exact.closer_margin(replay[i,finals[i]], answer)); rec["control_changes"] = {name: float(exact.closer_margin(arms[name][i,finals[i]], answer)-before) for name in replacements}
        records.append(rec)
    capability = {"|".join(map(str,key)): {"n":len(v),"accuracy":sum(x>0 for x in v)/len(v),"mean_closer_margin":sum(v)/len(v)} for key,v in sorted(capability_cells.items(),key=lambda z:str(z[0]))}
    targets = [r for r in records if r["program_role"]=="target"]; controls = [r for r in records if r["program_role"]=="control"]
    effect_metrics = {name: vector_metrics([r["exact_effect"] for r in targets],[r[name+"_effect"] for r in targets]) for name in replacements if name!="exact"}
    key_metrics = {name: vector_metrics(*values) for name,values in source_vectors.items()}; factor_metrics = {name: vector_metrics(*values) for name,values in factor_values.items()}
    by_pair = {}
    for pair in sorted({r["ordered_pair"] for r in targets}):
        items=[r for r in targets if r["ordered_pair"]==pair]; by_pair[pair]={"n":len(items),"exact_positive_fraction":sum(r["exact_effect"]>0 for r in items)/len(items),"both_vs_exact":vector_metrics([r["exact_effect"] for r in items],[r["both_effect"] for r in items])}
    target_rms=math.sqrt(sum(r["exact_effect"]**2 for r in targets)/len(targets)); control_rms=math.sqrt(sum(r["control_changes"]["both"]**2 for r in controls)/len(controls))
    ranks={name:table["rank"] for name,table in tables.items()}; capable=all(v["accuracy"]>=BARS["capability_accuracy_min"] and v["mean_closer_margin"]>0 for v in capability.values())
    instrument=counts==[PRICE["forwards"],PRICE["sequences"]] and replay_error<=BARS["replay_max"] and max(ranks.values())<=2 and len(targets)==len(controls)==72 and capable
    live=all(v["exact_positive_fraction"]>=BARS["exact_positive_min"] for v in by_pair.values())
    keys_ok=all(v["cosine"]>=BARS["key_cosine_min"] and v["relative_l2_error"]<=BARS["key_relative_l2_max"] for v in key_metrics.values())
    factors_ok=all(v["cosine"]>=BARS["factor_cosine_min"] and v["relative_l2_error"]<=BARS["factor_relative_l2_max"] and v["sign_agreement"]>=BARS["factor_sign_min"] for v in factor_metrics.values())
    both=effect_metrics["both"]; effect_ok=both["cosine"]>=BARS["effect_cosine_min"] and both["relative_l2_error"]<=BARS["effect_relative_l2_max"] and both["sign_agreement"]>=BARS["effect_sign_min"] and BARS["effect_norm_ratio_min"]<=both["predicted_to_actual_norm_ratio"]<=BARS["effect_norm_ratio_max"] and all(v["both_vs_exact"]["cosine"]>=BARS["pair_cosine_min"] and v["both_vs_exact"]["relative_l2_error"]<=BARS["pair_relative_l2_max"] and v["both_vs_exact"]["sign_agreement"]>=BARS["pair_sign_min"] for v in by_pair.values())
    improvement=effect_metrics["recipient_score"]["relative_l2_error"]-both["relative_l2_error"]; selective=control_rms/max(target_rms,1e-30)<=BARS["control_to_target_rms_max"]
    predictions={"pred_a_exact_instrument_and_capability":bool(instrument),"pred_b_exact_joint_parent_live":bool(instrument and live),"pred_c_rank2_key_sources_transfer":bool(instrument and keys_ok),"pred_d_live_query_score_transfers":bool(instrument and factors_ok),"pred_e_donor_free_interaction_transfers":bool(instrument and effect_ok),"pred_f_key_substitution_beats_recipient_score":bool(instrument and improvement>=BARS["recipient_score_improvement_min"]),"pred_g_control_selectivity":bool(instrument and selective)}
    if not (counts==[PRICE["forwards"],PRICE["sequences"]] and replay_error<=BARS["replay_max"] and max(ranks.values())<=2): terminal="invalid"
    elif not capable: terminal="fifth_construction_capability_null"
    elif all(predictions.values()): terminal="key_rank2_payload_rank2_interaction_transfer"
    else: terminal="key_rank2_interaction_null"
    program={name:{"rank":table["rank"],"mean":table["mean"].cpu().tolist(),"basis":table["basis"].cpu().tolist(),"type_coefficients":table["coefficients"].cpu().tolist(),"singular_values":table["singular_values"].cpu().tolist()} for name,table in tables.items()}
    result={"schema":"bracket_layered_pending_key_rank2_interaction_v1_result","terminal":terminal,"predictions":predictions,"instrument":{"native_factor_replay_max_logit_error":replay_error,"table_ranks":ranks},"capability_cells":capability,"key_source_metrics":key_metrics,"score_factor_metrics":factor_metrics,"effect_metrics":effect_metrics,"by_ordered_pair":by_pair,"relative_l2_improvement_over_recipient_score":improvement,"target_exact_effect_rms":target_rms,"compressed_control_rms":control_rms,"control_to_target_rms":control_rms/max(target_rms,1e-30),"program":program,"price":{**PRICE,"observed_forwards":counts[0],"observed_sequences":counts[1]},"claim_boundary":"Third-authority rank-two key and payload type tables with live recipient query/payload, tested prospectively on a fifth construction without donor-state access; native suffix and delimiter label remain external.","train_rows_sha256":digest(TRAIN_ROWS),"eval_rows_sha256":digest(EVAL_ROWS),"binding_sha256":digest(BINDING),"runner_sha256":digest(RUNNER),"checkpoint_sha256":checkpoint.weights_sha256,"wall_seconds":time.perf_counter()-tic,"records":records}
    validate_result_contract(result,PREDICTION_REGISTRY); atomic_create_json(OUT,result)
    print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","key_source_metrics","score_factor_metrics","effect_metrics","relative_l2_improvement_over_recipient_score","control_to_target_rms","price")},indent=2)); assert terminal!="invalid"


if __name__ == "__main__": main()
