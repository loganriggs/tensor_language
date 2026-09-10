#!/usr/bin/env python3
# BQGATE: jointQK feature subspaces,28forwards224seq,first8fit/last8eval.
"""pred_a instrument/native capability; pred_b own joint subspace routing error
<=.20 and recovery>=.80; pred_c other subspace routing effect<=.20, all3panels.
Both QK halves always used. Product-port intervention, no native weight ablation.
"""
import os,json,sys,time,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(ROOT),str(POLY)]
import torch
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import correlative_joint_qk_executor_v1 as J
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
from run_correlative_three_factor_v1 import margin,bridge,center,serial
OUT=POLY/'CORRELATIVE_JOINT_QK_SUBSPACES_V1_RESULT.json';BIND=POLY/'CORRELATIVE_JOINT_QK_SUBSPACES_V1_BINDING.json'
def metric(effect,reference):
    e=center(effect.double());r=center(reference.double());en=(e-r).square().sum(-1);rn=r.square().sum(-1);vn=e.square().sum(-1)
    return dict(relative_error=float(en.sum().sqrt()/rn.sum().sqrt().clamp_min(1e-30)),relative_effect=float(vn.sum().sqrt()/rn.sum().sqrt().clamp_min(1e-30)),projection_recovery=float((e*r).sum()/rn.sum().clamp_min(1e-30)),reference_norm=float(rn.sum().sqrt()),error_squared=serial(en),reference_squared=serial(rn),effect_squared=serial(vn))
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text())['panels'];artifact=torch.load(POLY/'CORRELATIVE_REPLAYABLE_INTERFACE_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)
    for name in ['A1','A2','C']:
        assert len(panels[name])==16 and len({r['row_id'] for r in panels[name]})==16
        for r in panels[name]:assert len(r['base_ids'])==len(r['donor_ids']) and r['base_semantic_position']==r['donor_semantic_position']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,body_forwards=28,sequences=224,fit_rows=16,evaluation_rows=24,max_basis_columns=48)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False);backend=P.Bilin18TorchBackend.load('cuda');counts=[0,0];caps=[];spaces={};fits={};reports={};checks={}
    def count(_m,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=28 and counts[1]<=224
    hook=backend.model.transformer.h[0].attn.register_forward_pre_hook(count)
    def run(rows,side,**kw):
        af,z,c=J.forward(backend,g.batch_of(rows,side),artifact,**kw);caps.append({k:v for k,v in c.items() if k in ['errors','coordinate_error']});return af,z,c
    try:
        for fam,name in [('A','A1'),('B','C')]:
            rows=panels[name][:8];_,_,bc=run(rows,'base');_,_,dc=run(rows,'donor');spaces[fam],fits[fam]=J.build_spaces(bc,dc)
        for name in ['A1','A2','C']:
            rows=panels[name][8:];branch='R' if name=='C' else 'P';own='B' if name=='C' else 'A';other='A' if own=='B' else 'B'
            ba,bz,bc=run(rows,'base');da,dz,dc=run(rows,'donor')
            va,vz,_=run(rows,'base',donor=dc,mode='value',branch=branch);fa,fz,_=run(rows,'base',donor=dc,mode='full',branch=branch)
            evaluated={}
            for fam in ['A','B']:
                aa,zz,_=run(rows,'base',donor=dc,mode='space',space=spaces[fam],branch=branch);evaluated[fam]=metric(zz-vz,fz-vz)
            _,noop,_=run(rows,'base',donor=bc,mode='space',space=spaces[own],branch=branch)
            _,ref=g.forward_units(backend,g.batch_of(rows,'base'),units=artifact['units'],donor_cache=dc['cache'],q={k:v.cuda() for k,v in artifact['q'].items()},complement=branch=='R',return_logits=True)
            checks[name+'_noop']=bridge(noop,bz);checks[name+'_full_reference']=bridge(fz,ref)
            reports[name]=dict(own_family=own,other_family=other,capable=int(((margin(ba)>0)&(margin(da)>0)).sum()),families=evaluated,base_margin=serial(margin(ba)),donor_margin=serial(margin(da)),value_margin=serial(margin(va)),full_margin=serial(margin(fa)),row_ids=[r['row_id'] for r in rows])
    finally:hook.remove()
    overlap={}
    for unit in spaces['A']:
        overlap[unit]={}
        for role in ['query','key']:
            a=spaces['A'][unit][role];b=spaces['B'][unit][role];cross=a.T@b;overlap[unit][role]=dict(rank_A=a.shape[1],rank_B=b.shape[1],normalized_overlap=float(cross.square().sum()/max(1,min(a.shape[1],b.shape[1]))))
    valid=counts==[28,224] and all(c['max_abs']<=1e-3 and c['relative_l2']<=1e-5 for c in checks.values()) and all(c['coordinate_error']<=1e-10 and all(v['relative_l2']<=1e-5 and v['max_scaled']<=1 for v in c['errors'].values()) for c in caps) and all(d['orthogonality']<=1e-9 and d['fit_relative']<=1e-5 for fam in fits.values() for unit in fam.values() for d in unit.values()) and all(r['capable']==8 for r in reports.values())
    ownok=valid and all(r['families'][r['own_family']]['reference_norm']>1e-4 and r['families'][r['own_family']]['relative_error']<=.2 and r['families'][r['own_family']]['projection_recovery']>=.8 for r in reports.values())
    otherok=valid and all(r['families'][r['other_family']]['reference_norm']>1e-4 and r['families'][r['other_family']]['relative_effect']<=.2 for r in reports.values())
    ap=POLY/'CORRELATIVE_JOINT_QK_SUBSPACES_V1_BASES.pt';assert not ap.exists();torch.save({f:{u:{k:v.cpu() for k,v in roles.items()} for u,roles in units.items()} for f,units in spaces.items()},ap)
    result=dict(schema='correlative.joint_qk_subspaces.v1',predictions={'pred_a_instrument':bool(valid),'pred_b_own_routing_subspace':bool(ownok),'pred_c_cross_subspace_selectivity':bool(otherok)},reports=reports,checks=checks,fit_diagnostics=fits,overlap=overlap,worst_factor_relative=max(v['relative_l2'] for c in caps for v in c['errors'].values()),worst_factor_scaled=max(v['max_scaled'] for c in caps for v in c['errors'].values()),coordinate_error=max(c['coordinate_error'] for c in caps),artifact_sha256=digest(ap),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),wall_seconds=time.perf_counter()-tic,price=dict(body_forwards=counts[0],sequences=counts[1],basis_bytes=ap.stat().st_size,native_weight_saving=0,native_parameters=sum(p.numel() for p in backend.model.parameters())),scope='Held out from first8 basis fitting only; previously opened examples. Explicit joint-product-port intervention, not native raw-input or weight ablation. Both score factors retained; no independent extraction or removal claim.')
    atomic_create_json(OUT,result);print(json.dumps(dict(predictions=result['predictions'],effects={n:{f:{k:v for k,v in m.items() if k in ['relative_error','relative_effect','projection_recovery']} for f,m in r['families'].items()} for n,r in reports.items()},wall_seconds=result['wall_seconds'],price=result['price'])))
if __name__=='__main__':main()
