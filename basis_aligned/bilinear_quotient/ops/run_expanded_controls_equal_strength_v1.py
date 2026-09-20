#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_strength pred_c_selectivity pred_d_relative_advantage
"""Compare frozen candidate and plain source at equal native number-effect norm."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'expanded_controls_equal_strength_v1_result.json'
PLAN=dict(prefix=12,native_suffix=108,bisection_steps=80)
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
    binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
    prior=json.loads((A/'source_ood_v2_result.json').read_text());expanded=json.loads((A/'expanded_source_controls_v1_result.json').read_text())
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((A/'source_ood_v2_result.json').read_bytes()).hexdigest()==binding['source_sha256']
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
    model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};records=[];replays=[];brackets=[]
    for context in binding['contexts']:
        panel,template=context['panel'],context['template'];c=build(model,context,modal);counts['prefix']+=c['prefix_calls']
        raw,x0,first,read,pairs,batch=[c[k] for k in ['raw','x0','first','read','pairs','batch']];n=len(raw)
        def native(direction,scale):
            counts['native_suffix']+=1;x=(raw.double()+direction*scale[:,None,None]).float()
            for layer in range(11,18):
                b=model.transformer.h[layer]
                if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                attention,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
            logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
            v=logits.gather(1,pairs.reshape(n,-1)).reshape(n,len(pairs[0]),2)
            return (v[:,:,0]-v[:,:,1]).double()
        one=torch.ones(n,device='cuda',dtype=torch.float64);base=native(torch.zeros_like(raw,dtype=torch.float64),one)
        families=[r['family'] for r in c['entries']];index={f:[i for i,v in enumerate(families) if v==f] for f in dict.fromkeys(families)}
        for axis,role in enumerate(['subject','attractor']):
            candidate=c['directions'][...,axis];pos,six=c['source_components'][role]
            plain=torch.zeros_like(raw,dtype=torch.float64);plain[batch,pos]=six[:,2:5].sum(1)
            full=base-native(candidate,one);reference=base-native(plain,one)
            targets={family:float(reference[ids,0].norm()) for family,ids in index.items()}
            lo={family:0. for family in index};hi={family:1. for family in index}
            for family,ids in index.items():
                oldref=next(r for r in prior['records'] if r['panel']==panel and r['role']==role and r['family']==family and r['arm']=='unitB')
                replays.append(float((reference[ids,:4]-torch.tensor(oldref['target'],device='cuda',dtype=torch.float64)).abs().max()))
                oldfull=next(r for r in expanded['records'] if r['panel']==panel and r['role']==role and r['family']==family)
                replays.append(float((full[ids].norm(dim=0)-torch.tensor(oldfull['native_effect_norms'],device='cuda',dtype=torch.float64)).abs().max()))
                brackets.append(float(full[ids,0].norm())>=targets[family]>0)
            history=[]
            for step in range(10):
                mid={f:(lo[f]+hi[f])/2 for f in index};scales=torch.tensor([mid[f] for f in families],device='cuda',dtype=torch.float64)
                number=(base-native(candidate,scales))[:,0];counts['bisection_steps']+=1
                observed={f:float(number[ids].norm()) for f,ids in index.items()};history.append(dict(amplitudes=mid,number_norms=observed))
                for f in index:
                    if observed[f]<targets[f]:lo[f]=mid[f]
                    else:hi[f]=mid[f]
            chosen={f:(lo[f]+hi[f])/2 for f in index};scales=torch.tensor([chosen[f] for f in families],device='cuda',dtype=torch.float64);matched=base-native(candidate,scales)
            for family,ids in index.items():
                target=targets[family];cn=matched[ids].norm(dim=0);bn=reference[ids].norm(dim=0)
                records.append(dict(panel=panel,family=family,role=role,amplitude=chosen[family],target_norm=target,number_cosine=float(torch.nn.functional.cosine_similarity(matched[ids,0],reference[ids,0],dim=0)),strength_error=abs(float(cn[0])-target)/target,candidate_norms=cn.tolist(),baseline_norms=bn.tolist(),candidate_ratios=(cn/target).tolist(),baseline_ratios=(bn/target).tolist(),candidate_effect=matched[ids].tolist(),baseline_effect=reference[ids].tolist(),calibration=[dict(amplitude=h['amplitudes'][family],number_norm=h['number_norms'][family]) for h in history]))
    instrument=counts==PLAN and max(replays)<=1e-4 and all(brackets);strength=all(r['strength_error']<=.05 and r['number_cosine']>=.95 for r in records)
    selective=all(max(r['candidate_ratios'][4:])<=.1 for r in records)
    advantage=all(max(r['candidate_norms'][4:])<=.75*max(r['baseline_norms'][4:]) for r in records)
    result=dict(plan=PLAN,counts=counts,records=records,max_prior_replay=max(replays),brackets=brackets,predictions=dict(pred_a_instrument=instrument,pred_b_strength=instrument and strength,pred_c_selectivity=instrument and strength and selective,pred_d_relative_advantage=instrument and strength and advantage),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps({k:result[k] for k in ['predictions','counts','max_prior_replay','seconds']}))
if __name__=='__main__':main()
