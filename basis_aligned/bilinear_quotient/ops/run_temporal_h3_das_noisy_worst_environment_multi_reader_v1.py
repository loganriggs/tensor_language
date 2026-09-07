#!/usr/bin/env python3
"""Fit a rank-one H3 causal operator against several readers and environments."""

# BQGATE: EXPERIMENT pred_a_authority_closure_finiteness_and_price pred_b_odd_row_objective_improves_pooled_seed pred_c_sealed_mean_and_worst_improve_pooled_axis pred_d_sealed_beats_dim_at_every_reader_family pred_e_restart_stability_or_seeded_reproducibility
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fit_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as test_v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v9 as test_v9
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g
import run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2 as evaluator

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_h3_das_noisy_worst_environment_multi_reader_v1.json"
POOLED = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
TOURNAMENT = ROOT / "circuits/followups/temporal_h3_das_multi_reader_frozen_axis_tournament_v1_result.json"
CAP8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CAP9 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v9_capability_v1_result.json"
V1 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
V2 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
V8 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
V9 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v9.py"
EVALUATOR = ROOT / "ops/run_temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
OUT = ROOT / "circuits/followups/temporal_h3_das_noisy_worst_environment_multi_reader_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_das_noisy_worst_environment_multi_reader_v1"
EXPECTED = {
    "prior": "e50759b8115d47afa483a1f026b996c8cf5eb7227226dea51af6ebc79bf3b6b9",
    "pooled": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "tournament": "1d900618b28fc3421c6f3496f0b676742882c3460bab12d73cd3c4da2bdd764c",
    "cap8": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "cap9": "828d8b15d9bcf048de32d74384e2f4bc62972f20289515f8eb6c576302262392",
    "v1": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "v2": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "v8": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "v9": "9b771713c5803082c95a3566bc41120587e60f99c4c8bacc291602516bbe01a5",
    "evaluator": "561b40b093e0a46469fa95b01c23e1b0d7d294201aaccb63b918b86900359303",
    "unit_lib": "8a5389dfee9f239509ef2440409854b1115e46a729193cf0ceba548443a1d254",
}
STEPS, LR, SIGMA, TAU, ANCHOR_WEIGHT = 50, 0.025, 0.03, 0.10, 0.02
RESTARTS = ("pooled_aligned", "pooled_dim", "random_907")
CHECKPOINTS = tuple(range(0, STEPS + 1, 10))
PRICE = {"model_forwards": 2662, "example_evaluations": 43372,
         "transformer_backward_forwards": 2400, "model_updates": 150,
         "fit_parameters": 128, "evaluation_records": 620}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
def tensor_sha(tensor): return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def manual_readers(backend, batch, delta, projection, *, grad=False):
    """Exact differentiable H3 intervention with final logits and raw L15 H5/H1 readers."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    n_head, width = int(model.config.n_head), int(model.config.n_embd // model.config.n_head)
    query = torch.tensor([int(x) for x in batch.semantic_positions], device=backend.device)
    index = torch.arange(len(lengths), device=backend.device)
    captured = {}
    with torch.set_grad_enabled(grad):
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0

            def patch(_module, arguments):
                flattened = arguments[0]
                changed = flattened.clone().view(len(batch.row_ids), flattened.shape[1], n_head, width)
                changed[:, :, 3] = changed[:, :, 3] + (delta @ projection).to(changed)
                return (changed.reshape_as(flattened),) + tuple(arguments[1:])

            def read15(_module, arguments):
                heads = arguments[0].view(len(batch.row_ids), arguments[0].shape[1], n_head, width)
                captured["l15"] = torch.cat((heads[index, query, 5], heads[index, query, 1]), dim=1).float()

            patch_handle = block.attn.c_proj.register_forward_pre_hook(patch) if layer == 11 else None
            read_handle = block.attn.c_proj.register_forward_pre_hook(read15) if layer == 15 else None
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                if patch_handle is not None: patch_handle.remove()
                if read_handle is not None: read_handle.remove()
            x = live + attention
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
        last = logits[index, torch.tensor([length-1 for length in lengths], device=backend.device)].float()
        answer = last[index, torch.tensor(batch.answer_ids, device=backend.device)]
        foil = last[index, torch.tensor(batch.foil_ids, device=backend.device)]
        margin = answer - foil
        if not grad:
            margin, last, captured["l15"] = margin.detach(), last.detach(), captured["l15"].detach()
        return {"margin": margin, "l15": captured["l15"], "logits": last}


def attach_targets(backend, rows):
    torch, F = backend.torch, backend.F
    context = evaluator.capture_context(backend, rows)
    zero, identity = torch.zeros((128,128), device=backend.device), torch.eye(128, device=backend.device)
    base = manual_readers(backend, context["base_batch"], context["delta"], zero)
    full = manual_readers(backend, context["base_batch"], context["delta"], identity)
    refs = {}
    for key in ("margin", "l15"):
        refs[key] = (full[key]-base[key]).square().mean().clamp_min(1e-12).detach()
    base_centered = base["logits"] - base["logits"].mean(dim=1, keepdim=True)
    full_centered = full["logits"] - full["logits"].mean(dim=1, keepdim=True)
    refs["vector"] = (full_centered-base_centered).square().mean().clamp_min(1e-12).detach()
    refs["kl"] = F.kl_div(F.log_softmax(base["logits"],dim=-1), F.log_softmax(full["logits"],dim=-1),
                           log_target=True,reduction="batchmean").clamp_min(1e-12).detach()
    native = torch.tensor(context["base_output"].answer_foil,device=backend.device)
    native_margin = native[:,0]-native[:,1]
    closure = float((native_margin-base["margin"]).abs().max())
    context.update({"base_readers":base,"full_readers":full,"refs":refs,"manual_base_margin_max_abs":closure})
    return context


def environment_loss(backend, context, projection, *, grad):
    torch, F = backend.torch, backend.F
    identity = torch.eye(128,device=backend.device)
    sub = manual_readers(backend,context["base_batch"],context["delta"],projection,grad=grad)
    comp = manual_readers(backend,context["base_batch"],context["delta"],identity-projection,grad=grad)
    base, full, refs = context["base_readers"],context["full_readers"],context["refs"]
    pieces = {}
    for key in ("margin","l15"):
        pieces[f"{key}_match"]=(sub[key]-full[key]).square().mean()/refs[key]
        pieces[f"{key}_inert"]=(comp[key]-base[key]).square().mean()/refs[key]
    log_base,log_full=F.log_softmax(base["logits"],dim=-1),F.log_softmax(full["logits"],dim=-1)
    kl_match=F.kl_div(F.log_softmax(sub["logits"],dim=-1),log_full,log_target=True,reduction="batchmean")/refs["kl"]
    kl_inert=F.kl_div(F.log_softmax(comp["logits"],dim=-1),log_base,log_target=True,reduction="batchmean")/refs["kl"]
    center=lambda x:x-x.mean(dim=1,keepdim=True)
    vector_match=(center(sub["logits"])-center(full["logits"])).square().mean()/refs["vector"]
    vector_inert=(center(comp["logits"])-center(base["logits"])).square().mean()/refs["vector"]
    pieces["vocab_match"]=.5*(kl_match+vector_match)
    pieces["vocab_inert"]=.5*(kl_inert+vector_inert)
    return sum(pieces.values()),pieces


def projector(raw):
    q=raw/raw.norm().clamp_min(1e-30)
    return q@q.T,q


def objective(backend,contexts,raw,anchor,*,grad,noise=False,generator=None):
    torch=backend.torch
    _p,unit=projector(raw)
    candidates=(unit,)
    if noise:
        z=torch.randn(unit.shape,generator=generator).to(backend.device)
        tangent=z-unit*(unit.T@z)
        tangent=tangent/tangent.norm().clamp_min(1e-30)
        candidates=tuple((unit+s*SIGMA*tangent)/(unit+s*SIGMA*tangent).norm().clamp_min(1e-30) for s in (-1.,1.))
    values=[]
    for context in contexts:
        variants=[environment_loss(backend,context,q@q.T,grad=grad)[0] for q in candidates]
        values.append(sum(variants)/len(variants))
    values=torch.stack(values)
    smooth=TAU*torch.logsumexp(values/TAU,dim=0)-TAU*math.log(len(contexts))
    anchor_penalty=1.0-(unit.T@anchor).square().squeeze()
    return smooth+ANCHOR_WEIGHT*anchor_penalty,values,anchor_penalty


def fit_restart(backend,train,heldout,initial,anchor,name):
    torch=backend.torch
    generator=torch.Generator(device="cpu").manual_seed(20260907+sum(map(ord,name)))
    raw=initial.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam([raw],lr=LR)
    trace=[];best=None
    def checkpoint(step):
        nonlocal best
        with torch.no_grad():
            joint,values,penalty=objective(backend,heldout,raw,anchor,grad=False)
            selection=float(values.max()+ANCHOR_WEIGHT*penalty)
        report={"step":step,"joint":float(joint),"selection":selection,"worst_environment":float(values.max()),"mean_environment":float(values.mean()),"anchor_distance":float(penalty)}
        trace.append(report)
        if best is None or selection<best[0]["selection"]:best=(report,(raw/raw.norm()).detach().clone())
    checkpoint(0)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        loss,_values,_penalty=objective(backend,train,raw,anchor,grad=True,noise=True,generator=generator)
        loss.backward();opt.step()
        if step in CHECKPOINTS:checkpoint(step)
    return {"name":name,"best":best[0],"axis":best[1],"trace":trace}


def rows_for(builder,capability):
    allowed={x for ids in capability["jointly_capable_row_ids"].values() for x in ids}
    return [r for r in builder.build_rows() if r["transform_id"] in ("A1","A2") and r["row_id"] in allowed]


def cell_score(item,panel):
    behavior=item["behavior_fraction_of_full_h3"][panel];transport=item["downstream_fraction_of_full_h3"][panel];kl=item["full_vocabulary_kl"][panel]
    pieces={"behavior_match_squared":(1-behavior["h3_regularized_rank7"])**2,"behavior_complement_squared":behavior["h3_regularized_rank7_orthogonal"]**2,"l15_transport_match_squared":(1-transport)**2,"full_vocab_match_fraction":kl["student_match_fraction"],"full_vocab_complement_fraction":kl["complement_inert_fraction"]}
    return sum(pieces.values()),pieces


def main():
    paths={"prior":PRIOR,"pooled":POOLED,"tournament":TOURNAMENT,"cap8":CAP8,"cap9":CAP9,"v1":V1,"v2":V2,"v8":V8,"v9":V9,"evaluator":EVALUATOR,"unit_lib":UNIT_LIB}
    if {k:sha(v) for k,v in paths.items()}!=EXPECTED:raise RuntimeError("multi-reader optimizer authority changed")
    prior,pooled,tournament,cap8,cap9=[json.loads(p.read_text()) for p in (PRIOR,POOLED,TOURNAMENT,CAP8,CAP9)]
    groups={(bank,panel):[r for r in builder.build_rows() if r["transform_id"]==panel] for bank,builder in (("v1",fit_v1),("v2",fit_v2)) for panel in ("A1","A2")}
    rows8,rows9=rows_for(test_v8,cap8),rows_for(test_v9,cap9)
    if prior.get("candidate_id")!=CANDIDATE_ID or pooled.get("terminal")!="task_conditioned" or tournament.get("terminal")!="optimizer_licensed" or len(rows8)!=60 or len(rows9)!=64 or any(len(x)!=32 for x in groups.values()):raise RuntimeError("authority terminal or population changed")
    dryrun={"candidate_id":CANDIDATE_ID,"dryrun":True,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"rank":1,"training_environments":[f"{b}_{p}_even" for b,p in groups],"selection_environments":[f"{b}_{p}_odd" for b,p in groups],"rows_per_environment_split":16,"readers":["answer_foil_margin","layer15_h5_h1","centered_full_vocabulary_kl_and_vector"],"restarts":list(RESTARTS),"steps_per_restart":STEPS,"checkpoints":list(CHECKPOINTS),"noise_sigma":SIGMA,"smooth_worst_tau":TAU,"anchor_weight":ANCHOR_WEIGHT,"sealed_banks":{"v8":len(rows8),"v9":len(rows9)},**PRICE}
    if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dryrun,sort_keys=True));return
    if OUT.exists():raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc,started=utc_now(),time.perf_counter();backend=producer.Bilin18TorchBackend.load("cuda");torch=backend.torch
    for parameter in backend.model.parameters():parameter.requires_grad_(False)
    anchor=torch.tensor(pooled["axis_artifacts"]["pooled_aligned_rank1"],device=backend.device).float().unsqueeze(1);anchor=anchor/anchor.norm()
    dim_rows=groups[("v1","A1")][0::2];prep=g.prepare(backend,dim_rows);dim=g.diff_in_means_direction(backend,prep,("attn:11:head:03",));dim=dim/dim.norm()
    generator=torch.Generator(device="cpu").manual_seed(907);random=torch.randn((128,1),generator=generator).to(backend.device);random=random/random.norm()
    train=[attach_targets(backend,rows[0::2]) for rows in groups.values()]
    heldout=[attach_targets(backend,rows[1::2]) for rows in groups.values()]
    closure=max(c["manual_base_margin_max_abs"] for c in train+heldout)
    with torch.no_grad():
        pooled_train=objective(backend,train,anchor,anchor,grad=False);pooled_heldout=objective(backend,heldout,anchor,anchor,grad=False)
    fits=[fit_restart(backend,train,heldout,initial,anchor,name) for name,initial in zip(RESTARTS,(anchor,dim,random))]
    selected=min(fits,key=lambda x:x["best"]["selection"]);q=selected["axis"];projection=q@q.T
    cosines=[float((fits[i]["axis"].T@fits[j]["axis"]).abs()) for i in range(3) for j in range(i+1,3)]
    evaluations={name:evaluator.evaluate_bank(backend,name,rows,q,projection) for name,rows in (("v8",rows8),("v9",rows9))}
    objectives={};cells=[]
    for bank in ("v8","v9"):
        for panel in ("A1","A2"):
            score,pieces=cell_score(evaluations[bank],panel);cells.append(score);objectives[f"{bank}_{panel}"]={"joint":score,**pieces}
    sealed={"mean":sum(cells)/4,"worst":max(cells),"by_cell":objectives}
    old=tournament["objectives"]
    families=("behavior_match_squared","behavior_complement_squared","l15_transport_match_squared","full_vocab_match_fraction","full_vocab_complement_fraction")
    family_sums={f:sum(c[f] for c in objectives.values()) for f in families}
    dim_family={f:sum(c[f] for c in old["dim"]["by_cell"].values()) for f in families}
    max_instrument=max(v for report in evaluations.values() for v in report["instrument"].values())
    fit_values=[v for fit in fits for point in fit["trace"] for v in point.values() if isinstance(v,(int,float))]
    pred_a=closure<=1e-4 and max_instrument<=1e-4 and all(math.isfinite(x) for x in fit_values)
    pred_b=selected["best"]["worst_environment"]<float(pooled_heldout[1].max())
    pred_c=sealed["mean"]<old["pooled_aligned"]["mean"] and sealed["worst"]<old["pooled_aligned"]["worst"]
    pred_d=all(family_sums[f]<dim_family[f] for f in families)
    pred_e=min(cosines)>=.80 or selected["name"]=="pooled_aligned"
    predictions={"pred_a_authority_closure_finiteness_and_price":bool(pred_a),"pred_b_odd_row_objective_improves_pooled_seed":bool(pred_b),"pred_c_sealed_mean_and_worst_improve_pooled_axis":bool(pred_c),"pred_d_sealed_beats_dim_at_every_reader_family":bool(pred_d),"pred_e_restart_stability_or_seeded_reproducibility":bool(pred_e)}
    terminal="invalid" if not pred_a else "multi_reader_das_candidate" if all(predictions.values()) else "task_memorization" if pred_b and not pred_c else "objective_miss" if not pred_b else "partial_generalization"
    result={"schema":"temporal_h3_das_noisy_worst_environment_multi_reader_result_v1","candidate_id":CANDIDATE_ID,"execution_policy":"managed_queue_only","started_utc":started_utc,"finished_utc":utc_now(),"serial_seconds":time.perf_counter()-started,"authority_sha256":EXPECTED,"dryrun":dryrun,"instrument":{"training_manual_base_margin_max_abs":closure,"sealed_max_closure_or_reconstruction_abs":max_instrument},"fit":{"pooled_seed_train":{"joint":float(pooled_train[0]),"worst":float(pooled_train[1].max())},"pooled_seed_heldout":{"joint":float(pooled_heldout[0]),"worst":float(pooled_heldout[1].max())},"selected_restart":selected["name"],"selected_axis_sha256":tensor_sha(q),"selected_axis":q.flatten().cpu().tolist(),"restart_min_axis_cosine":min(cosines),"restart_pair_cosines":cosines,"restarts":[{k:v for k,v in fit.items() if k!="axis"} for fit in fits]},"sealed":{"optimized":sealed,"frozen_pooled_aligned":old["pooled_aligned"],"frozen_dim":old["dim"],"optimized_family_sums":family_sums,"dim_family_sums":dim_family},"evaluations":{b:{k:v for k,v in report.items() if k not in ("records","forwards","evaluations")} for b,report in evaluations.items()},"predictions":predictions,"terminal":terminal,"price":PRICE}
    atomic_create_json(OUT,result);print(json.dumps({"candidate_id":CANDIDATE_ID,"instrument":result["instrument"],"fit":{k:v for k,v in result["fit"].items() if k not in ("selected_axis","restarts")},"sealed":result["sealed"],"predictions":predictions,"terminal":terminal,"price":PRICE},sort_keys=True))


if __name__=="__main__":main()
