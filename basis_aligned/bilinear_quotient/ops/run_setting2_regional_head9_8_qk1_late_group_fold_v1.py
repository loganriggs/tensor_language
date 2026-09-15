#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_late_self_material pred_c_three_block_compact pred_d_late_self_leads pred_e_cross_boundary_matters
"""Group the exact regional head9.8 QK1 carry fold into four late/remainder blocks."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_managed_runner as managed
from folded_normalized_router_v1 import EPS, rotary
from regional_cue_row_check_v1 import validate

RUNNER=Path(__file__).resolve()
PREREG=P/"SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FOLD_V1_PREREGISTRATION.md"
ROWS=P/"FIRST_TOKEN_PATH_FRESH_V1_ROWS.json"
BINDING=P/"SETTING2_REGIONAL_HEAD9_8_QK1_LATE_GROUP_FOLD_V1_BINDING.json"
PARENT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_carry_source_fold_v1_result.json"
OUT=ROOT/"basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json"
SOURCE_NAMES=("embedding",)+tuple(x for layer in range(8) for x in (f"attn{layer}",f"mlp{layer}"))
LATE_NAMES=("attn5","mlp5","mlp6","mlp7")
GROUP_NAMES=("late","remainder")
TERM_NAMES=tuple(f"{u}_x_{v}" for u in GROUP_NAMES for v in GROUP_NAMES)
PRICE={"physical_prefix_forwards":14,"sequences":96,"carry_sources":17,"grouped_sources":2,"ordered_group_terms":4,"fits":0,"backwards":0,"parameter_updates":0}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(actual,expected): return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))


def coefficients(lambdas):
    embed=1.0
    for layer in range(10): embed=float(lambdas[layer,0])*embed+float(lambdas[layer,1])
    writes={}
    for layer in range(8):
        c=float(torch.prod(lambdas[layer+1:10,0])); writes[f"attn{layer}"]=c; writes[f"mlp{layer}"]=c
    return embed,writes


def load_bound():
    binding=json.loads(BINDING.read_text())
    files={"preregistration":PREREG,"rows":ROWS,"row_check":P/"regional_cue_row_check_v1.py","parent_result":PARENT}
    if binding["files"]!={k:sha(v) for k,v in files.items()} or binding["price"]!=PRICE: raise ValueError("bound input or price changed")
    parent=json.loads(PARENT.read_text())
    if parent["terminal"]!="valid_qk1_carry_source_fold": raise ValueError("parent QK1 carry source fold is invalid")
    rows=json.loads(ROWS.read_text())["rows"]; checks=validate(rows); buckets={}
    for i,row in enumerate(rows): buckets.setdefault(len(row["ids"]),[]).append(i)
    if len(rows)!=96 or sum(math.ceil(len(v)/8) for v in buckets.values())!=14: raise ValueError("row/forward price changed")
    return {**binding,"row_checks":checks,"buckets":buckets},rows,parent


def plan():
    bound,rows,_=load_bound()
    return {"schema":"setting2_regional_head9_8_qk1_late_group_fold_v1_plan","model_loaded":False,"gpu_accessed":False,"queue_touched":False,"rows":len(rows),"source_names":SOURCE_NAMES,"late_names":LATE_NAMES,"term_names":TERM_NAMES,"price":PRICE,"row_checks":bound["row_checks"],"binding_sha256":sha(BINDING)}


def head_project(parts,total,weight):
    projected=[p@weight.T for p in parts]; complete=total@weight.T
    denominator=(complete.square().mean(-1,keepdim=True)+EPS).sqrt()
    return [p/denominator for p in projected],complete/denominator


@torch.no_grad()
def main():
    planned=plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned,sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval(); bound,rows,parent=load_bound(); device=next(model.parameters()).device
    state=model.state_dict(); lambdas=torch.stack([b.lambdas.detach().double().cpu() for b in model.transformer.h])
    embed_c,write_c=coefficients(lambdas)
    coefficient9=float(torch.prod(lambdas[10:18,0])); coefficient16=float(lambdas[17,0])
    projection9=state["transformer.h.9.attn.c_proj.weight"].double().cpu()
    q1,k1,q2,k2=[state[f"transformer.h.9.attn.c_{n}.weight"][8*128:9*128].double().cpu() for n in ("q","k","q2","k2")]
    left,right,down=[state[f"transformer.h.17.mlp.{n}.weight"].double().cpu() for n in ("Left","Right","Down")]
    unembed=state["lm_head.weight"].double().cpu()
    folded=torch.empty(4,96,dtype=torch.float64); parent_fold=torch.empty(96,dtype=torch.float64); complete_fold=torch.empty(96,dtype=torch.float64)
    carry_num=carry_den=score_num=score_den=0.0; forward_count=0
    late_indices={SOURCE_NAMES.index(name) for name in LATE_NAMES}
    for _,indices in sorted(bound["buckets"].items()):
        for offset in range(0,len(indices),8):
            selected=indices[offset:offset+8]; tokens=torch.tensor([rows[i]["ids"] for i in selected],device=device)
            x=F.rms_norm(model.transformer.wte(tokens),(1152,)); x0=x; first_value=None; writes={}
            for layer,block in enumerate(model.transformer.h[:8]):
                r=block.lambdas[0]*x+block.lambdas[1]*x0
                a,first_value=block.attn(F.rms_norm(r,(1152,)),first_value)
                m=block.mlp(F.rms_norm(r+a,(1152,))); writes[f"attn{layer}"]=a; writes[f"mlp{layer}"]=m; x=r+a+m
            block8,block9=model.transformer.h[8],model.transformer.h[9]
            r8=block8.lambdas[0]*x+block8.lambdas[1]*x0; a8,first_value=block8.attn(F.rms_norm(r8,(1152,)),first_value); m8=block8.mlp(F.rms_norm(r8+a8,(1152,)))
            carry_native=block9.lambdas[0]*r8+block9.lambdas[1]*x0
            parts=[embed_c*x0.double().cpu()]+[write_c[name]*writes[name].double().cpu() for name in SOURCE_NAMES[1:]]
            carry=sum(parts); carry_num+=float((carry-carry_native.double().cpu()).square().sum()); carry_den+=float(carry_native.double().cpu().square().sum())
            r9=carry+block9.lambdas[0].double().cpu()*a8.double().cpu()+block9.lambdas[0].double().cpu()*m8.double().cpu()
            denom=(r9.square().mean(-1,keepdim=True)+EPS).sqrt(); normalized=[p/denom for p in parts]; total=r9/denom
            grouped=[sum(normalized[i] for i in range(17) if i in late_indices),sum(normalized[i] for i in range(17) if i not in late_indices)]
            qp,qtotal=head_project(grouped,total,q1); kp,ktotal=head_project(grouped,total,k1)
            length=tokens.shape[1]; rotations=torch.stack([rotary(pos,128) for pos in range(length)])
            qp=[torch.einsum("btd,tde->bte",p,rotations.transpose(1,2)) for p in qp]; kp=[torch.einsum("btd,tde->bte",p,rotations.transpose(1,2)) for p in kp]
            qtotal=torch.einsum("btd,tde->bte",qtotal,rotations.transpose(1,2)); ktotal=torch.einsum("btd,tde->bte",ktotal,rotations.transpose(1,2))
            components=torch.stack([torch.einsum("bd,btd->bt",qp[u][:,-1],kp[v])/128 for u in range(2) for v in range(2)])
            parent_score=torch.einsum("bd,btd->bt",sum(qp)[:,-1],sum(kp))/128
            score_num+=float((components.sum(0)-parent_score).square().sum()); score_den+=float(parent_score.square().sum())
            _,q2total=head_project([],total,q2); _,k2total=head_project([],total,k2)
            q2total=torch.einsum("btd,tde->bte",q2total,rotations.transpose(1,2)); k2total=torch.einsum("btd,tde->bte",k2total,rotations.transpose(1,2))
            score2=torch.einsum("bd,btd->bt",q2total[:,-1],k2total)/128
            score1=torch.einsum("bd,btd->bt",qtotal[:,-1],ktotal)/128
            norm9_native=F.rms_norm((carry_native+block9.lambdas[0]*a8+block9.lambdas[0]*m8),(1152,))
            value=block9.attn.c_v(norm9_native).view(len(selected),length,9,128)
            value=((1-block9.attn.lamb)*value+block9.attn.lamb*first_value.view_as(value))[:,:,8].double().cpu()
            zterms=torch.stack([((c*score2)[:,:,None]*value).sum(1) for c in components])
            zparent=((parent_score*score2)[:,:,None]*value).sum(1); zcomplete=((score1*score2)[:,:,None]*value).sum(1)
            x=r8+a8+m8; x,first_value=block9(x,first_value,x0); mlp16=None
            for layer in range(10,17):
                block=model.transformer.h[layer]; r=block.lambdas[0]*x+block.lambdas[1]*x0; a,first_value=block.attn(F.rms_norm(r,(1152,)),first_value); m=block.mlp(F.rms_norm(r+a,(1152,))); x=r+a+m
                if layer==16: mlp16=m[:,-1].double().cpu()
            last=model.transformer.h[17]; raw17=last.lambdas[0]*x+last.lambdas[1]*x0; a17,first_value=last.attn(F.rms_norm(raw17,(1152,)),first_value); pre17=(raw17+a17)[:,-1].double().cpu()
            p=coefficient16*mlp16; readers=torch.stack([unembed[rows[i]["uk_id"]]-unembed[rows[i]["us_id"]] for i in selected]); fd=readers@down; lp,rp=p@left.T,p@right.T; d17=pre17.square().mean(-1)+EPS
            residual_reader=((rp*fd)@left+(lp*fd)@right)/d17[:,None]
            head_reader=coefficient9*residual_reader@projection9[:,8*128:9*128]
            folded[:,selected]=(zterms*head_reader.unsqueeze(0)).sum(-1)
            parent_fold[selected]=(zparent*head_reader).sum(-1); complete_fold[selected]=(zcomplete*head_reader).sum(-1); forward_count+=1
    carry_error=math.sqrt(carry_num/max(carry_den,1e-30)); score_error=math.sqrt(score_num/max(score_den,1e-30)); fold_error=rel(folded.sum(0),parent_fold)
    delta=parent_fold[1::2]-parent_fold[0::2]; delta_complete=complete_fold[1::2]-complete_fold[0::2]; dt=folded[:,1::2]-folded[:,0::2]; denom=delta.square().sum().clamp_min(1e-30)
    reports={}; families=sorted(set(r["family_name"] for r in rows))
    for term,name in enumerate(TERM_NAMES):
        fr={}
        for family in families:
            ids=[i//2 for i in range(0,96,2) if rows[i]["family_name"]==family]; fr[family]=float(dt[term,ids].norm()/delta[ids].norm().clamp_min(1e-30))
        reports[name]={"change_norm_ratio":float(dt[term].norm()/delta.norm().clamp_min(1e-30)),"aligned_fraction":float((dt[term]*delta).sum()/denom),"family_change_norm_ratios":fr}
    ranking=sorted(reports,key=lambda n:reports[n]["change_norm_ratio"],reverse=True)
    three_block=dt[:3].sum(0); three_block_error=rel(three_block,delta)
    cross_boundary=dt[1]+dt[2]; cross_boundary_ratio=float(cross_boundary.norm()/delta.norm().clamp_min(1e-30))
    parent_ratio=float(delta.norm()/delta_complete.norm().clamp_min(1e-30)); expected=parent["metrics"]["parent_change_norm_ratio"]; parent_bridge=abs(parent_ratio-expected)
    late=reports["late_x_late"]; instrument=forward_count==14 and carry_error<=1e-6 and score_error<=1e-8 and fold_error<=1e-8 and parent_bridge<=1e-6
    predictions={
        "pred_a_exact_instrument":bool(instrument),
        "pred_b_late_self_material":bool(instrument and late["change_norm_ratio"]>=.35 and min(late["family_change_norm_ratios"].values())>=.20),
        "pred_c_three_block_compact":bool(instrument and three_block_error<=.30),
        "pred_d_late_self_leads":bool(instrument and ranking[0]=="late_x_late" and all(late["family_change_norm_ratios"][f]>=max(reports[n]["family_change_norm_ratios"][f] for n in TERM_NAMES if n!="late_x_late") for f in families)),
        "pred_e_cross_boundary_matters":bool(instrument and cross_boundary_ratio>=.25),
    }
    result={"schema":"setting2_regional_head9_8_qk1_late_group_fold_v1_result","terminal":"valid_qk1_late_group_fold" if instrument else "invalid","predictions":predictions,"metrics":{"carry_source_reconstruction_relative_error":carry_error,"grouped_qk1_score_sum_relative_error":score_error,"folded_sum_relative_error":fold_error,"parent_change_norm_ratio":parent_ratio,"parent_ratio_absolute_bridge":parent_bridge,"three_block_replay_relative_error":three_block_error,"cross_boundary_change_norm_ratio":cross_boundary_ratio,"physical_prefix_forwards":forward_count,"sequences":96,"pair_count":48},"source_names":SOURCE_NAMES,"late_source_names":LATE_NAMES,"term_names":TERM_NAMES,"ranking":ranking,"term_reports":reports,"price":PRICE,"row_checks":planned["row_checks"],"outcome_access":{"behavioral_logits":False,"new_rows":False,"fits":False},"checkpoint_weights_sha256":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3","binding_sha256":sha(BINDING),"runner_sha256":sha(RUNNER),"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"scope":"Exact late/remainder four-block grouping inside the QK1 carry×carry term of the regional head9.8×MLP16×MLP17 path; fixed denominators, QK2, value, suffix, and reader."}
    managed.atomic_create_json(OUT,result); print(json.dumps({"terminal":result["terminal"],"predictions":predictions,"metrics":result["metrics"],"ranking":ranking,"term_reports":reports},indent=2)); assert instrument

if __name__=="__main__": main()
