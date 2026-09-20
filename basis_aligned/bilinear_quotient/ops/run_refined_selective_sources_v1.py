#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_shared_native pred_c_oracle_native
"""Fixed role-level source selector versus a per-input linear feasibility ceiling."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'refined_selective_sources_v1_result.json'
PLAN=dict(prefix=12,double_suffix=24,gradient=72,native_suffix=28)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from refined_native_sources import build_refined as build, PARENTS, NAMES
    from native_source_observables import source_observables
    from shared_selective_source_lp import choose
    binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
    prior=json.loads((A/'source_ood_v2_result.json').read_text())
    parent_prior=json.loads((A/'shared_selective_sources_v1_result.json').read_text())
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};cache=[];gradients=[];replays=[];fdchecks=[]
    reference=np.array([0.,0.,1.,1.,1.,0.])[PARENTS]
    for context in binding['contexts']:
        panel,template=context['panel'],context['template'];c=build(model,context,modal);counts['prefix']+=c['prefix_calls'];cache.append((context,c))
        for role in ['subject','attractor']:
            pos,ds=c['source_components'][role];zero=torch.zeros(len(ds),23,device='cuda',dtype=torch.float64)
            def evaluate(a):
                counts['double_suffix']+=1
                return source_observables(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],a)
            with torch.enable_grad():
                a=zero.clone().requires_grad_();base=evaluate(a)
                g=torch.stack([torch.autograd.grad(base[:,o].sum(),a,retain_graph=o<8)[0] for o in range(9)],dim=1);counts['gradient']+=9
            old=next(v for v in parent_prior['gradients'] if v['panel']==panel and v['template']==template and v['role']==role)
            folded=torch.stack([g[:,:,[i for i,p in enumerate(PARENTS) if p==parent]].sum(-1) for parent in range(6)],dim=-1)
            replays.append(float((folded-torch.tensor(old['gradient'],device='cuda',dtype=torch.float64)).abs().max()))
            v=torch.linspace(-.6,.7,23,device='cuda',dtype=torch.float64).expand_as(zero)
            fd=(evaluate(.001*v)-evaluate(-.001*v))/.002;exact=torch.einsum('boi,bi->bo',g,v)
            fdchecks.append(float(((fd-exact).norm(dim=0)/exact.norm(dim=0).clamp_min(1e-10)).max()))
            gradients.append(dict(panel=panel,template=template,role=role,families=[r['family'] for r in c['entries']],gradient=g.tolist()))
    shared={};fits={}
    for role in ['subject','attractor']:
        fit=np.concatenate([np.array(g['gradient']) for g in gradients if g['role']==role and g['panel']=='opposite'])
        shared[role],fits[role]=choose(fit,reference)
    records=[];oracle_checks=[];amplitude_records=[]
    for context,c in cache:
        panel,template=context['panel'],context['template'];raw,x0,first,read,pairs,batch=[c[k] for k in ['raw','x0','first','read','pairs','batch']];n=len(raw)
        def native(ds,pos,a):
            counts['native_suffix']+=1;x=raw.clone();x[batch,pos]=(raw[batch,pos].double()+torch.einsum('bi,bid->bd',a,ds)).float()
            for layer in range(11,18):
                b=model.transformer.h[layer]
                if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                attention,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
            logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
            z=logits.gather(1,pairs.reshape(n,-1)).reshape(n,9,2)
            return (z[:,:,0]-z[:,:,1]).double()
        dspos=c['source_components']['subject'];baseline=native(dspos[1],dspos[0],torch.zeros(n,23,device='cuda',dtype=torch.float64))
        for role in ['subject','attractor']:
            pos,ds=c['source_components'][role];group=next(g for g in gradients if g['panel']==panel and g['template']==template and g['role']==role)
            ga=np.array(group['gradient']);individual=[]
            for row in ga:
                aa,check=choose(row,reference);individual.append(aa);oracle_checks.append(dict(panel=panel,template=template,role=role,**check))
            arms=dict(unitB=np.tile(reference,(n,1)),shared=np.tile(shared[role],(n,1)),oracle=np.array(individual))
            targets={arm:baseline-native(ds,pos,torch.tensor(av,device='cuda',dtype=torch.float64)) for arm,av in arms.items()}
            amplitude_records.append(dict(panel=panel,template=template,role=role,amplitudes={k:v.tolist() for k,v in arms.items()}))
            for family in dict.fromkeys(group['families']):
                ids=[i for i,f in enumerate(group['families']) if f==family];ref=targets['unitB'][ids,0];den=float(ref.norm())
                oldref=next(r for r in prior['records'] if r['panel']==panel and r['role']==role and r['family']==family and r['arm']=='unitB')
                replays.append(float((targets['unitB'][ids,:4]-torch.tensor(oldref['target'],device='cuda',dtype=torch.float64)).abs().max()))
                for arm in arms:
                    effect=targets[arm][ids];number=float(effect[:,0].norm());retention=float(effect[:,0]@ref)/max(den*den,1e-30)
                    leakage=float(effect[:,1:].norm(dim=0).max())/max(number,1e-30)
                    records.append(dict(panel=panel,role=role,family=family,arm=arm,retention=retention,collateral_ratio=leakage,number_norm=number,reference_norm=den,effect=effect.tolist(),linear_prediction=(-np.einsum('boi,bi->bo',ga[ids],arms[arm][ids])).tolist()))
    refinement_checks=[v for _,c in cache for v in c['refinement_checks']]
    refined_ok=max(c['collapse_error'] for c in refinement_checks)<=1e-10 and max(v['relative'] for c in refinement_checks for v in c['corrections'])<=.01
    instrument=refined_ok and counts==PLAN and max(replays)<=1e-8 and max(fdchecks)<=1e-3
    passes=lambda rr:all(r['retention']>=.8 and r['collateral_ratio']<=.1 for r in rr)
    held=[r for r in records if r['arm']=='shared' and r['panel']=='congruent'];oracle=[r for r in records if r['arm']=='oracle']
    result=dict(source_names=NAMES,parent_mapping=PARENTS,refinement_checks=refinement_checks,plan=PLAN,counts=counts,gradients=gradients,shared={k:v.tolist() for k,v in shared.items()},fits=fits,oracle_checks=oracle_checks,amplitudes=amplitude_records,records=records,max_replay=max(replays),finite_difference_checks=fdchecks,predictions=dict(pred_a_instrument=instrument,pred_b_shared_native=instrument and passes(held),pred_c_oracle_native=instrument and passes(oracle)),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','counts','shared','fits','max_replay','finite_difference_checks','seconds']}))
if __name__=='__main__':main()
