#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_role_native pred_c_role_prediction pred_d_template_native
"""Factor source response into residual readers and recipient source vectors."""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'residual_reader_transfer_v1_result.json';TENSORS=A/'residual_reader_transfer_v1_tensors.pt'
PLAN=dict(prefix=12,double_suffix=4,reader_gradient=36,native_suffix=18)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import numpy as np
    import torch
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write,guard_torch_save
    sys.path.insert(0,str(P))
    from refined_native_sources import build_refined
    from native_source_observables import source_observables,source_observables32
    from shared_selective_source_lp import choose
    binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
    source=A/'refined_selective_sources_v1_result.json';prior=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.])[prior['parent_mapping']]
    assert not OUT.exists() and not TENSORS.exists()
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');is_id=enc.encode(' is')[0];old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};cache={};factors={};replays=[];records=[];selection_checks=[]
    for context in binding['contexts']:
        panel,template=context['panel'],context['template'];c=build_refined(model,context,modal);cache[panel,template]=c;counts['prefix']+=c['prefix_calls'];n=len(c['raw']);d=c['raw'].shape[-1]
        zero_d=torch.zeros(n,1,d,device='cuda',dtype=torch.float64);zero_p=torch.zeros(n,device='cuda',dtype=torch.long);zero_a=torch.zeros(n,1,device='cuda',dtype=torch.float64)
        with torch.enable_grad():
            raw=c['raw'].double().detach().requires_grad_()
            values=source_observables(model,raw,c['x0'],c['first'],zero_d,zero_p,c['read'],c['pairs'],zero_a);counts['double_suffix']+=1
            readers=torch.stack([torch.autograd.grad(values[:,o].sum(),raw,retain_graph=o<8)[0] for o in range(9)],dim=2);counts['reader_gradient']+=9
        orientation=torch.where(c['pairs'][:,0,0]==is_id,1.,-1.).double()
        for role in ['subject','attractor']:
            pos,ds=c['source_components'][role];reader=readers[c['batch'],pos].detach()
            gradient=torch.einsum('bod,bkd->bok',reader,ds)
            oldg=next(g for g in prior['gradients'] if g['panel']==panel and g['role']==role and g['template']==template)
            replays.append(float((gradient-torch.tensor(oldg['gradient'],device='cuda',dtype=torch.float64)).abs().max()))
            canonical=reader.clone();canonical[:,0]*=orientation[:,None]
            factors[panel,template,role]=dict(reader=canonical,sources=ds,orientation=orientation,families=[r['family'] for r in c['entries']])
    role_banks={role:torch.cat([v['reader'] for k,v in factors.items() if k[0]=='opposite' and k[2]==role]).mean(0) for role in ['subject','attractor']}
    template_banks={(template,role):factors['opposite',template,role]['reader'].mean(0) for template,role in {(k[1],k[2]) for k in factors}}
    for (panel,template),c in cache.items():
        if panel!='congruent':continue
        n=len(c['raw'])
        def native(role,amplitudes):
            counts['native_suffix']+=1;pos,ds=c['source_components'][role]
            return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(amplitudes,device='cuda',dtype=torch.float64))
        baseline=native('subject',np.zeros((n,23)))
        for role in ['subject','attractor']:
            current=factors[panel,template,role];donor=factors['opposite',template,role];indices=torch.arange(n,device='cuda')
            if role=='attractor':indices=indices^1
            banks=dict(donor_reader=donor['reader'][indices],role_bank=role_banks[role][None].expand(n,-1,-1),template_bank=template_banks[template,role][None].expand(n,-1,-1),oracle=current['reader'])
            gradients={};arms={}
            for name,reader in banks.items():
                g=torch.einsum('bod,bkd->bok',reader,current['sources']);g[:,0]*=current['orientation'][:,None];gradients[name]=g
                if name=='oracle':
                    arms[name]=np.array(next(g for g in prior['amplitudes'] if g['panel']=='congruent' and g['template']==template and g['role']==role)['amplitudes']['oracle'])
                else:
                    weights=[]
                    for row in g.cpu().numpy():
                        aa,check=choose(row,reference);weights.append(aa);selection_checks.append(check)
                    arms[name]=np.array(weights)
            for arm,aa in arms.items():
                effect=baseline-native(role,aa);prediction=-torch.einsum('bok,bk->bo',gradients[arm],torch.tensor(aa,device='cuda',dtype=torch.float64))
                families=current['families']
                for family in dict.fromkeys(families):
                    ids=[i for i,f in enumerate(families) if f==family]
                    old=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='unitB');ref=torch.tensor(old['effect'],device='cuda',dtype=torch.float64)[:,0];v=effect[ids]
                    if arm=='oracle':
                        previous=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='oracle');replays.append(float((v-torch.tensor(previous['effect'],device='cuda',dtype=torch.float64)).abs().max()))
                    den=v[:,0].norm().clamp_min(1e-30);ret=float(v[:,0]@ref/ref.square().sum());leak=float(v[:,1:].norm(dim=0).max()/den)
                    err=(prediction[ids]-v).norm(dim=0)/den
                    records.append(dict(role=role,family=family,arm=arm,retention=ret,control_ratio=leak,passed=bool(ret>=.8 and leak<=.1),prediction_errors=err.tolist(),effect=v.tolist(),amplitudes=aa[ids].tolist()))
    instrument=counts==PLAN and max(replays)<=1e-8
    def passes(arm):return all(r['passed'] for r in records if r['arm']==arm)
    role_prediction=all(r['prediction_errors'][0]<=.1 and max(r['prediction_errors'][1:])<=.05 for r in records if r['arm']=='role_bank')
    bundle=dict(groups=[dict(panel=k[0],template=k[1],role=k[2],**{name:(value.cpu() if torch.is_tensor(value) else value) for name,value in v.items()}) for k,v in factors.items()],role_banks={k:v.cpu() for k,v in role_banks.items()},template_banks={k:v.cpu() for k,v in template_banks.items()})
    guard_torch_save(bundle,str(TENSORS))
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tensor_sha256=hashlib.sha256(TENSORS.read_bytes()).hexdigest(),plan=PLAN,counts=counts,max_replay=max(replays),records=records,selection_checks=selection_checks,reader_coefficients=dict(role=2*9*1152,template=4*9*1152),predictions=dict(pred_a_instrument=instrument,pred_b_role_native=instrument and passes('role_bank'),pred_c_role_prediction=instrument and role_prediction,pred_d_template_native=instrument and passes('template_bank')),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,max_replay=max(replays),passes={arm:sum(r['passed'] for r in records if r['arm']==arm) for arm in arms},seconds=result['seconds'])))
if __name__=='__main__':main()
