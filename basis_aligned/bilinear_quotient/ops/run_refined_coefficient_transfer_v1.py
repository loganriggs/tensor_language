#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_lexical_transfer pred_c_noun_transfer
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'refined_coefficient_transfer_v1_result.json'
PLAN=dict(prefix=6,native_suffix=14)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from refined_native_sources import build_refined
    binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
    source=A/'refined_selective_sources_v1_result.json';prior=json.loads(source.read_text());assert not OUT.exists()
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};records=[];replays=[];checks=[]
    for context in binding['contexts']:
        if context['panel']!='congruent':continue
        template=context['template'];c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];checks.extend(c['refinement_checks'])
        raw,x0,first,read,pairs,batch=[c[k] for k in ['raw','x0','first','read','pairs','batch']];n=len(raw)
        def native(ds,pos,a):
            counts['native_suffix']+=1;x=raw.clone();x[batch,pos]=(raw[batch,pos].double()+torch.einsum('bi,bid->bd',a,ds)).float()
            for layer in range(11,18):
                b=model.transformer.h[layer]
                if layer>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
                attention,_=b.attn(F.rms_norm(x,(x.shape[-1],)),first);x=x+attention;x=x+b.mlp(F.rms_norm(x,(x.shape[-1],)))
            logits=30*torch.tanh(model.lm_head(F.rms_norm(x[batch,read],(x.shape[-1],)))/30)
            z=logits.gather(1,pairs.reshape(n,-1)).reshape(n,9,2);return (z[:,:,0]-z[:,:,1]).double()
        pos,ds=c['source_components']['subject'];base=native(ds,pos,torch.zeros(n,23,device='cuda',dtype=torch.float64));families=[r['family'] for r in c['entries']]
        for role in ['subject','attractor']:
            donor=next(g for g in prior['amplitudes'] if g['panel']=='opposite' and g['role']==role and g['template']==template)
            held=next(g for g in prior['amplitudes'] if g['panel']=='congruent' and g['role']==role and g['template']==template)
            aa=np.array(donor['amplitudes']['oracle']);arms=dict(same_lexical_pair=aa,next_noun_same_number=np.roll(aa,-2,axis=0),held_oracle=np.array(held['amplitudes']['oracle']));pos,ds=c['source_components'][role]
            for arm,amplitudes in arms.items():
                effect=base-native(ds,pos,torch.tensor(amplitudes,device='cuda',dtype=torch.float64))
                for family in dict.fromkeys(families):
                    ids=[i for i,f in enumerate(families) if f==family];old=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='unitB');ref=torch.tensor(old['effect'],device='cuda',dtype=torch.float64)[:,0];v=effect[ids]
                    if arm=='held_oracle':
                        oldoracle=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='oracle');replays.append(float((v-torch.tensor(oldoracle['effect'],device='cuda',dtype=torch.float64)).abs().max()))
                    retention=float(v[:,0]@ref/ref.square().sum());leak=float(v[:,1:].norm(dim=0).max()/v[:,0].norm().clamp_min(1e-30))
                    records.append(dict(role=role,family=family,template=template,arm=arm,retention=retention,collateral_ratio=leak,effect=v.tolist(),passed=bool(retention>=.8 and leak<=.1)))
    instrument=counts==PLAN and max(replays)<=1e-4 and max(c['collapse_error'] for c in checks)<=1e-10
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),plan=PLAN,counts=counts,checks=checks,max_replay=max(replays),records=records,predictions=dict(pred_a_instrument=instrument,pred_b_lexical_transfer=instrument and all(r['passed'] for r in records if r['arm']=='same_lexical_pair'),pred_c_noun_transfer=instrument and all(r['passed'] for r in records if r['arm']=='next_noun_same_number')),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,max_replay=max(replays),passes={arm:sum(r['passed'] for r in records if r['arm']==arm) for arm in arms},seconds=result['seconds'])))
if __name__=='__main__':main()
