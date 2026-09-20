#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_new_controls pred_c_old_controls
"""Prospective control outputs for frozen source edits; shared setup replay."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'expanded_source_controls_v1_result.json'
PLAN=dict(prefix=12,double_suffix=68,native_suffix=12)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from frozen_two_site_context import build
    from native_source_observables import source_observables
    from two_site_composition_scoring import key
    binding_path=P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json';binding=json.loads(binding_path.read_text())
    control_path=P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json';controls=json.loads(control_path.read_text())
    assert hashlib.sha256(binding_path.read_bytes()).hexdigest()==controls['source_binding_sha256']
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((A/'source_ood_v2_result.json').read_bytes()).hexdigest()==binding['source_sha256']
    prior=json.loads((A/'two_site_composition_v1r1_result.json').read_text());previous={(g['panel'],g['template']):g for g in prior['groups']}
    budgets={(r['panel'],r['family']):r for r in binding['budgets']}
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};groups=[];records=[];replays=[];numerics=[]
    for context in binding['contexts']:
        panel,template=context['panel'],context['template'];c=build(model,context,modal);counts['prefix']+=c['prefix_calls']
        raw,x0,first,directions,read,pairs,batch=[c[k] for k in ['raw','x0','first','directions','read','pairs','batch']]
        n=len(raw);dummy_d=torch.zeros(n,1,raw.shape[-1],device='cuda',dtype=torch.float64);dummy_p=torch.zeros(n,device='cuda',dtype=torch.long);dummy_a=torch.zeros(n,1,device='cuda',dtype=torch.float64)
        def state(coordinate):return raw.double()+torch.einsum('btdp,p->btd',directions,torch.tensor(coordinate,device='cuda',dtype=torch.float64))
        def evaluate(coordinate):
            counts['double_suffix']+=1
            return source_observables(model,state(coordinate),x0,first,dummy_d,dummy_p,read,pairs,dummy_a)
        def native(coordinate):
            counts['native_suffix']+=1;x=state(coordinate).float()
            for layer in range(11,18):
                b=model.transformer.h[layer]
                if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                attention,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
            logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
            v=logits.gather(1,pairs.reshape(n,-1)).reshape(n,len(pairs[0]),2)
            return (v[:,:,0]-v[:,:,1]).double()
        base=evaluate([0,0]);native_base=native([0,0]);targets={'0,0':torch.zeros_like(base)};actual={}
        for coordinate in binding['coordinates']:
            label=key(coordinate);targets[label]=base-evaluate(coordinate)
            old_target=torch.tensor(previous[panel,template]['double'][label],device='cuda',dtype=torch.float64)
            replays.append(float((targets[label][:,:4]-old_target).abs().max()))
            if coordinate in [[1,0],[0,1]]:actual[label]=native_base-native(coordinate)
        families=[r['family'] for r in c['entries']]
        for role,coordinate in [('subject',[1,0]),('attractor',[0,1])]:
            label=key(coordinate)
            for family in dict.fromkeys(families):
                ids=[i for i,f in enumerate(families) if f==family];budget=budgets[panel,family][role+'_number_norm']
                effects=targets[label][ids].norm(dim=0);native_effects=actual[label][ids].norm(dim=0)
                precision=(targets[label][ids]-actual[label][ids]).norm(dim=0)/budget;numerics.append(float(precision.max()))
                records.append(dict(panel=panel,family=family,role=role,budget=budget,effect_norms=effects.tolist(),relative_effects=(effects/budget).tolist(),native_effect_norms=native_effects.tolist(),native_relative_effects=(native_effects/budget).tolist(),numerical_ratios=precision.tolist()))
        groups.append(dict(panel=panel,template=template,families=families,targets={k:v.tolist() for k,v in targets.items()}))
    instrument=counts==PLAN and max(replays)<=1e-10 and max(numerics)<=.01
    new=all(max(r['native_relative_effects'][4:])<=.1 for r in records);old=all(max(r['native_relative_effects'][1:4])<=.1 for r in records)
    result=dict(plan=PLAN,counts=counts,control_binding_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((P/'frozen_two_site_context.py').read_bytes()).hexdigest(),predictions=dict(pred_a_instrument=instrument,pred_b_new_controls=instrument and new,pred_c_old_controls=instrument and old),max_old_replay=max(replays),max_numerical_ratio=max(numerics),controls=controls,records=records,groups=groups,seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','counts','max_old_replay','max_numerical_ratio','seconds']}))
if __name__=='__main__':main()
