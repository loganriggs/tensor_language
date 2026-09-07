#!/usr/bin/env python3
"""Family-held-out, hard-feasible full-vocabulary-KL refinement of rank-8 response projectors."""
# BQGATE: EXPERIMENT pred_a_authority_instrument_hash_finiteness_and_price pred_b_nonbaseline_family_feasible_checkpoint_exists pred_c_selected_lowers_selection_secondary pred_d_selected_generalizes_to_sealed_families pred_e_refinement_is_stable_and_weight_readable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as sealed_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as sealed_i
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_target_contrast_response_basis_v1 as v1
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_frozen_response_ood_regularization_v2 as ood

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_family_feasible_kl_refinement_v1.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
OOD = ROOT / "circuits/followups/temporal_five_mlp_frozen_response_ood_regularization_v2_result.json"
STB = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
STC = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
SIB = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
SIC = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_family_feasible_kl_refinement_v1_result.json"
EXPECTED = {
    "prior": "52923b93a4f51361984b50e08a7337b9bb9a9d069fd5f926495f4706f57544ef",
    "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
    "ood": "0f5060d10aec960988ebbe02610e7c2d7558e08e5a08a470a32bfe7a27965428",
    "sealed_t_builder": "f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450",
    "sealed_t_capability": "0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026",
    "sealed_i_builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "sealed_i_capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "ood_runner": "b7768b1d1c060effc9e4f516e0bc85e2e524cf742e5ea594a571110dd6820127",
}
STEPS, CHECKPOINTS, LR, BARRIER = 40, (0, 10, 20, 30, 40), .02, 100.0
STARTS = ("even_fit", "odd_fit", "crossfit_consensus")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def task_name(row): return "temporal" if str(row["task_id"]).startswith("temporal_auxiliary.") else "iswas"


def forward_projectors(backend, batch, base, donor, bases, *, complement=False, grad=False):
    """Differentiable exact model forward with the v1 prefix-patching semantics."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    positions = list(batch.semantic_positions); n = len(positions)
    width = model.config.n_embd // model.config.n_head
    attention = {}; mlps = {}
    for site, q in bases.items():
        kind, layer, head = atlasrun.site_parts(site)
        (attention if kind == "attn" else mlps).setdefault(layer, []).append((head, q))
    with torch.set_grad_enabled(grad):
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0 = x; vcache = None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0
            def ahook(_m, args, layer=layer):
                if layer not in attention: return None
                value = args[0]; changed = value.clone()
                for head, q in attention[layer]:
                    a, z = head * width, (head + 1) * width
                    for i, pos in enumerate(positions):
                        m = int(pos) + 1
                        delta = (donor["attention"][layer][i, :m, a:z] - base["attention"][layer][i, :m, a:z]).to(value).float()
                        projected = (delta @ q) @ q.T
                        if complement: projected = delta - projected
                        changed[i, :m, a:z] = base["attention"][layer][i, :m, a:z].to(value) + projected.to(value)
                return (changed,) + tuple(args[1:])
            handle = block.attn.c_proj.register_forward_pre_hook(ahook)
            try: attn_out, vcache = block.attn(F.rms_norm(live, (model.config.n_embd,)), vcache)
            finally: handle.remove()
            x = live + attn_out
            mlp_out = block.mlp(F.rms_norm(x, (model.config.n_embd,)))
            if layer in mlps:
                changed = mlp_out.clone()
                for _head, q in mlps[layer]:
                    for i, pos in enumerate(positions):
                        m = int(pos) + 1
                        delta = (donor["mlp"][layer][i, :m] - base["mlp"][layer][i, :m]).to(mlp_out).float()
                        projected = (delta @ q) @ q.T
                        if complement: projected = delta - projected
                        changed[i, :m] = base["mlp"][layer][i, :m].to(mlp_out) + projected.to(mlp_out)
                mlp_out = changed
            x = x + mlp_out
        idx = torch.arange(n, device=backend.device)
        last = torch.as_tensor([length - 1 for length in lengths], device=backend.device)
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
        return x[idx, torch.as_tensor(positions, device=backend.device)].float(), logits[idx, last].float()


def context(backend, rows, *, full):
    batch, base_out, base, _donor_out, donor = v1.cap(backend, rows)
    base_state = atlasrun.states(backend.torch, backend, base_out, rows)
    base_logits = das.head_logits(backend, base_state).float()
    item = {"rows": rows, "batch": batch, "base": base, "donor": donor,
            "base_state": base_state, "base_logits": base_logits}
    if full:
        output, _ = atlasrun.run_patch(backend, batch, donor, comp.SITES)
        full_state = atlasrun.states(backend.torch, backend, output, rows)
        item.update(full_state=full_state, full_logits=das.head_logits(backend, full_state).float())
    return item


def groups(rows):
    out = {}
    for i, row in enumerate(rows): out.setdefault((task_name(row), row["transform_id"]), []).append(i)
    return out


def centered(x): return x - x.mean(-1, keepdim=True)


def secondary(backend, target, control, bases, *, grad):
    torch, F = backend.torch, backend.F
    ps, pl = forward_projectors(backend, target["batch"], target["base"], target["donor"], bases, grad=grad)
    cs, cl = forward_projectors(backend, target["batch"], target["base"], target["donor"], bases, complement=True, grad=grad)
    _zs, zl = forward_projectors(backend, control["batch"], control["base"], control["donor"], bases, grad=grad)
    logp, logc, logz = [F.log_softmax(x, -1) for x in (pl, cl, zl)]
    logb, logf = F.log_softmax(target["base_logits"], -1), F.log_softmax(target["full_logits"], -1)
    scores = []
    for key, ids in groups(target["rows"]).items():
        ix = torch.as_tensor(ids, device=backend.device)
        refv = (centered(target["full_logits"][ix]) - centered(target["base_logits"][ix])).square().mean().clamp_min(1e-12)
        refk = F.kl_div(logb[ix], logf[ix], log_target=True, reduction="batchmean").clamp_min(1e-12)
        vm = (centered(pl[ix]) - centered(target["full_logits"][ix])).square().mean() / refv
        vi = (centered(cl[ix]) - centered(target["base_logits"][ix])).square().mean() / refv
        km = F.kl_div(logp[ix], logf[ix], log_target=True, reduction="batchmean") / refk
        ki = F.kl_div(logc[ix], logb[ix], log_target=True, reduction="batchmean") / refk
        scores.append(vm + vi + km + ki)
    control_logb = F.log_softmax(control["base_logits"], -1)
    scales = json.loads(comp.GENERIC.read_text())["target_behavior_rms_scales"]
    for key, ids in groups(control["rows"]).items():
        ix = torch.as_tensor(ids, device=backend.device)
        kl = F.kl_div(logz[ix], control_logb[ix], log_target=True, reduction="batchmean") / .02
        a = torch.as_tensor([control["rows"][j]["donor_answer_id"] for j in ids], device=backend.device)
        f = torch.as_tensor([control["rows"][j]["donor_foil_id"] for j in ids], device=backend.device)
        ar = torch.arange(len(ids), device=backend.device)
        movement = ((zl[ix][ar, a] - zl[ix][ar, f]) - (control["base_logits"][ix][ar, a] - control["base_logits"][ix][ar, f])).square().mean() / (scales[key[0]] ** 2)
        scores.append(kl + movement)
    stack = torch.stack(scores)
    return stack.max(), {"family_scores": stack, "projected_state": ps, "projected_logits": pl,
                         "complement_state": cs, "complement_logits": cl, "control_logits": zl}


def target_metrics(backend, ctx, state, logits, reader):
    torch = backend.torch; rows = ctx["rows"]
    a = torch.as_tensor([r["donor_answer_id"] for r in rows], device=backend.device)
    f = torch.as_tensor([r["donor_foil_id"] for r in rows], device=backend.device)
    ix = torch.arange(len(rows), device=backend.device)
    base_margin = ctx["base_logits"][ix, a] - ctx["base_logits"][ix, f]
    full_margin = ctx["full_logits"][ix, a] - ctx["full_logits"][ix, f]
    margin = logits[ix, a] - logits[ix, f]
    full_delta, delta = full_margin - base_margin, margin - base_margin
    full_modes, modes = (ctx["full_state"] - ctx["base_state"]) @ reader, (state - ctx["base_state"]) @ reader
    report = {}
    for task in ("temporal", "iswas"):
        ids = [j for j, row in enumerate(rows) if task_name(row) == task]
        jx = torch.as_tensor(ids, device=backend.device)
        cells = [(delta[jx] - full_delta[jx]).square().sum() / full_delta[jx].square().sum().clamp_min(1e-30)]
        cells += [(modes[jx, k] - full_modes[jx, k]).square().sum() / full_modes[jx, k].square().sum().clamp_min(1e-30) for k in range(2)]
        report[task] = {"signed_projection": delta[jx] @ full_delta[jx] / full_delta[jx].square().sum().clamp_min(1e-30), "worst_residual": torch.stack(cells).max()}
    return report


def feasible(report): return all(float(x["signed_projection"]) >= .80 and float(x["worst_residual"]) <= .05 for x in report.values())


def control_metrics(backend, ctx, logits):
    torch, F = backend.torch, backend.F
    lb, lp = ctx["base_logits"], logits
    logb, logp = F.log_softmax(lb, -1), F.log_softmax(lp, -1)
    kl = (logb.exp() * (logb - logp)).sum(-1)
    scales = json.loads(comp.GENERIC.read_text())["target_behavior_rms_scales"]
    margin = {}
    for task in sorted({task_name(r) for r in ctx["rows"]}):
        ids = [j for j, r in enumerate(ctx["rows"]) if task_name(r) == task]
        ix = torch.as_tensor(ids, device=backend.device); ar = torch.arange(len(ids), device=backend.device)
        a = torch.as_tensor([ctx["rows"][j]["donor_answer_id"] for j in ids], device=backend.device)
        f = torch.as_tensor([ctx["rows"][j]["donor_foil_id"] for j in ids], device=backend.device)
        move = (lp[ix][ar, a] - lp[ix][ar, f]) - (lb[ix][ar, a] - lb[ix][ar, f])
        margin[task] = float(move.square().mean().sqrt()) / scales[task]
    return {"margin_rms_fraction": margin, "median_kl": float(kl.median()), "max_kl": float(kl.max()),
            "top1_flip_fraction": float((lb.argmax(-1) != lp.argmax(-1)).float().mean())}


def basis_union(torch, a, b): return torch.linalg.qr(torch.cat([a, b], dim=1), mode="reduced").Q


def bases_from_raw(unions, raws): return {site: unions[site] @ torch_qr(raws[site]) for site in comp.SITES}


def torch_qr(raw): return __import__("torch").linalg.qr(raw, mode="reduced").Q


def fit(backend, train_t, train_c, select_t, select_c, reader, unions, start_raw, name):
    torch = backend.torch
    raws = {site: value.detach().clone().requires_grad_(True) for site, value in start_raw.items()}
    optimizer = torch.optim.Adam(list(raws.values()), lr=LR); trace = []; candidates = []
    def checkpoint(step):
        with torch.no_grad():
            bases = bases_from_raw(unions, raws); sec, parts = secondary(backend, select_t, select_c, bases, grad=False)
            rep = target_metrics(backend, select_t, parts["projected_state"], parts["projected_logits"], reader)
            item = {"step": step, "secondary_worst": float(sec), "feasible": feasible(rep),
                    "target": {k: {q: float(v) for q, v in x.items()} for k, x in rep.items()},
                    "bases": {site: q.detach().clone() for site, q in bases.items()}}
            trace.append({k: v for k, v in item.items() if k != "bases"}); candidates.append(item)
    checkpoint(0)
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True); bases = bases_from_raw(unions, raws)
        sec, parts = secondary(backend, train_t, train_c, bases, grad=True)
        rep = target_metrics(backend, train_t, parts["projected_state"], parts["projected_logits"], reader)
        violations = []
        for item in rep.values():
            violations += [torch.relu(.80 - item["signed_projection"]).square(), torch.relu(item["worst_residual"] - .05).square()]
        loss = sec + BARRIER * torch.stack(violations).sum(); loss.backward(); optimizer.step()
        if step in CHECKPOINTS: checkpoint(step)
    okay = [x for x in candidates if x["feasible"]]
    best = min(okay or candidates, key=lambda x: (not x["feasible"], x["secondary_worst"], x["step"]))
    return {"name": name, "trace": trace, "best": best}


def main():
    paths = {"prior": PRIOR, "interface": INTERFACE, "ood": OOD, "sealed_t_builder": STB,
             "sealed_t_capability": STC, "sealed_i_builder": SIB, "sealed_i_capability": SIC,
             "ood_runner": Path(ood.__file__)}
    if {k: sha(v) for k, v in paths.items()} != EXPECTED: raise RuntimeError("family-feasible KL authority changed")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_family_feasible_kl_refinement_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "rank": 8,
           "sites": comp.SITES, "starts": STARTS, "steps": STEPS, "checkpoints": CHECKPOINTS,
           "model_forwards_max": 500, "fit_updates": 120, "model_updates": 120, "transformer_backwards": 120}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    iface, oldood = json.loads(INTERFACE.read_text()), json.loads(OOD.read_text())
    if iface["terminal"] != "stable_weight_readable_response_program" or oldood["terminal"] != "memorization_fragility_null": raise RuntimeError("upstream decision changed")
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    old_tcap, old_icap = json.loads(comp.TCAP.read_text()), json.loads(comp.ICAP.read_text())
    old_tr, old_ir, *_ = greedy.rows_and_controls(old_tcap, old_icap); te, to = old_tr[::2] + old_ir[::2], old_tr[1::2] + old_ir[1::2]
    controls = matched.control_rows(); ce, co = controls["discovery"], controls["validation"]
    ceb, _, ce0, _, ce1 = v1.cap(backend, ce); cob, _, co0, _, co1 = v1.cap(backend, co)
    teb, _, te0, _, te1 = v1.cap(backend, te); tob, _, to0, _, to1 = v1.cap(backend, to)
    cbe, _ = comp.fit_bases(backend, ceb, ce0, ce1); cbo, _ = comp.fit_bases(backend, cob, co0, co1)
    qe, _ = v1.fit_target(backend, teb, te0, te1, cbe); qo, _ = v1.fit_target(backend, tob, to0, to1, cbo)
    hashes_ok = all([interface.thash(qe[s]), interface.thash(qo[s])] == iface["records"][s]["fit_basis_sha256"] for s in comp.SITES)
    unions = {s: basis_union(torch, qe[s], qo[s]) for s in comp.SITES}
    consensus = {s: torch.linalg.svd(torch.cat([qe[s], qo[s]], 1), full_matrices=False).U[:, :8] for s in comp.SITES}
    starts = {"even_fit": {s: unions[s].T @ qe[s] for s in comp.SITES},
              "odd_fit": {s: unions[s].T @ qo[s] for s in comp.SITES},
              "crossfit_consensus": {s: unions[s].T @ consensus[s] for s in comp.SITES}}
    train_t = context(backend, old_tr + old_ir, full=True); train_c = context(backend, ce, full=False)
    with torch.no_grad():
        manual_state, manual_logits = forward_projectors(
            backend, train_t["batch"], train_t["base"], train_t["donor"], {}, grad=False)
    instrument = {
        "base_state_max_abs": float((manual_state - train_t["base_state"]).abs().max()),
        "base_logits_max_abs": float((manual_logits - train_t["base_logits"]).abs().max()),
    }
    stcap, sicap = json.loads(STC.read_text()), json.loads(SIC.read_text())
    selection_rows = sum((population.capable_rows(ood.fresh_t, json.loads(ood.TC.read_text()), p, 12) for p in ("A1", "A2")), []) + sum((population.capable_rows(ood.fresh_i, json.loads(ood.IC.read_text()), p, 12) for p in ("A1", "A2")), [])
    selection_controls = [r for r in ood.fresh_t.build_rows() if r["transform_id"] == "P"][:16] + co
    sealed_rows = sum((population.capable_rows(sealed_t, stcap, p, 12) for p in ("A1", "A2")), []) + sum((population.capable_rows(sealed_i, sicap, p, 12) for p in ("A1", "A2")), [])
    sealed_controls = [r for r in sealed_t.build_rows() if r["transform_id"] == "P"][:16]
    fit_ids = {r["row_id"] for r in train_t["rows"] + selection_rows}
    authority_ok = (len(selection_rows) == len(sealed_rows) == 48 and not fit_ids.intersection(r["row_id"] for r in sealed_rows)
                    and all(r["base_semantic_position"] == r["donor_semantic_position"] for r in selection_rows + selection_controls + sealed_rows + sealed_controls))
    if not authority_ok: raise RuntimeError("family split changed")
    select_t, select_c = context(backend, selection_rows, full=True), context(backend, selection_controls, full=False)
    fits = [fit(backend, train_t, train_c, select_t, select_c, reader, unions, starts[name], name) for name in STARTS]
    feasible_candidates = [(fitrow, point) for fitrow in fits for point in [fitrow["best"]] if point["feasible"]]
    selected_fit, selected = min(feasible_candidates, key=lambda x: x[1]["secondary_worst"]) if feasible_candidates else min(((f, f["best"]) for f in fits), key=lambda x: (not x[1]["feasible"], x[1]["secondary_worst"]))
    baseline = min(point["secondary_worst"] for fitrow in fits for point in [next(x for x in fitrow["trace"] if x["step"] == 0)] if point["feasible"])
    sealed_tctx, sealed_cctx = context(backend, sealed_rows, full=True), context(backend, sealed_controls, full=False)
    with torch.no_grad():
        _sec, sparts = secondary(backend, sealed_tctx, sealed_cctx, selected["bases"], grad=False)
        sealed_target = target_metrics(backend, sealed_tctx, sparts["projected_state"], sparts["projected_logits"], reader)
        sealed_control = control_metrics(backend, sealed_cctx, sparts["control_logits"])
    best_bases = [f["best"]["bases"] for f in fits]
    overlaps = []
    for i in range(len(best_bases)):
        for j in range(i + 1, len(best_bases)):
            overlaps.append(min(float(torch.linalg.svdvals(best_bases[i][s].T @ best_bases[j][s]).square().mean()) for s in comp.SITES))
    closure = []
    generator = torch.Generator(device="cpu").manual_seed(907)
    for site, q in selected["bases"].items():
        kind, layer, head = atlasrun.site_parts(site)
        w = (backend.model.transformer.h[layer].attn.c_proj.weight.detach().float()[:, head*128:(head+1)*128]
             if kind == "attn" else backend.model.transformer.h[layer].mlp.Down.weight.detach().float())
        if kind == "attn":
            x = torch.randn((17, q.shape[0]), generator=generator).to(backend.device)
            direct = ((x @ q) @ q.T) @ w.T; compiled = (x @ q) @ (w @ q).T
        else:
            x = torch.randn((17, w.shape[1]), generator=generator).to(backend.device)
            direct = ((x @ w.T @ q) @ q.T); compiled = (x @ (w.T @ q)) @ q.T
        closure.append(float((direct - compiled).square().sum() / direct.square().sum().clamp_min(1e-30)))
    finite = [x for f in fits for point in f["trace"] for x in [point["secondary_worst"]] + [v for r in point["target"].values() for v in r.values()]]
    pa = (authority_ok and hashes_ok and reader_ok and orientation <= 1e-6
          and max(instrument.values()) <= 1e-4 and all(math.isfinite(x) for x in finite))
    pb = any(point["step"] > 0 and point["feasible"] for f in fits for point in f["trace"])
    pc = selected["step"] > 0 and selected["secondary_worst"] <= .90 * baseline
    sd = {k: {q: float(v) for q, v in x.items()} for k, x in sealed_target.items()}
    pd = feasible(sealed_target) and max(sealed_control["margin_rms_fraction"].values()) <= .10 and sealed_control["median_kl"] <= .02 and sealed_control["top1_flip_fraction"] <= .05
    pe = sorted(overlaps, reverse=True)[0] >= .70 and max(closure) <= 1e-10
    predictions = {"pred_a_authority_instrument_hash_finiteness_and_price": bool(pa), "pred_b_nonbaseline_family_feasible_checkpoint_exists": bool(pb), "pred_c_selected_lowers_selection_secondary": bool(pc), "pred_d_selected_generalizes_to_sealed_families": bool(pd), "pred_e_refinement_is_stable_and_weight_readable": bool(pe)}
    terminal = "invalid" if not pa else "family_stable_kl_refined_response_program" if all(predictions.values()) else "family_memorization" if pc and not pd else "frozen_projector_frontier" if selected["step"] == 0 else "kl_refinement_inconclusive"
    result = {"schema": "temporal_five_mlp_family_feasible_kl_refinement_result_v1", "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "instrument": instrument, "selection": {"selected_start": selected_fit["name"], "selected_step": selected["step"], "selected_secondary": selected["secondary_worst"], "baseline_secondary": baseline,
              "fits": [{"name": f["name"], "trace": f["trace"]} for f in fits]}, "sealed": {"target": sd, "control": sealed_control}, "stability": {"pair_min_site_overlaps": overlaps, "max_weight_closure_rse": max(closure)},
              "selected_basis_sha256": {s: interface.thash(q) for s, q in selected["bases"].items()}, "selected_bases": {s: q.detach().cpu().tolist() for s, q in selected["bases"].items()},
              "predictions": predictions, "terminal": terminal, "price": {"model_forwards_max": 500, "fit_updates": 120, "model_updates": 120, "transformer_backwards": 120}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("selection", "sealed", "stability", "predictions", "terminal", "price") if k != "selection"} | {"selection": {k:v for k,v in result["selection"].items() if k != "fits"}}, sort_keys=True))


if __name__ == "__main__": main()
