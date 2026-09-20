#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_selectivity pred_c_prediction pred_d_capability
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'source_ood_v3_role_bank_v1_result.json'
PLAN=dict(prefix=18,native_suffix=30)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import numpy as np
    import torch
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from refined_native_sources import build_refined,PARENTS
    from native_source_observables import source_observables32
    from shared_selective_source_lp import choose
    binding_path=P/'SOURCE_OOD_V3_ROLE_BANK_BINDING.json';binding=json.loads(binding_path.read_text());bank_path=P/binding['bank_file']
    assert hashlib.sha256(bank_path.read_bytes()).hexdigest()==binding['bank_sha256']
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    banks=torch.load(bank_path,map_location='cpu',weights_only=True)['role_banks']
    oldbinding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());oldresult=json.loads((A/'residual_reader_transfer_v1_result.json').read_text());oldunit=json.loads((A/'refined_selective_sources_v1_result.json').read_text())
    contexts=[dict(dataset='v2_replay',**c) for c in oldbinding['contexts'] if c['panel']=='congruent']
    for panel in ['opposite','congruent']:
        filename=f'SOURCE_OOD_V3_{panel.upper()}_ROWS.json';rows=json.loads((P/filename).read_text())
        for template in dict.fromkeys(r['template'] for r in rows):
            n=sum(r['template']==template for r in rows)
            contexts.append(dict(dataset='v3',panel=panel,template=template,rows_file=filename,amplitudes={role:[[0.,0.,1.,1.,1.,0.]]*n for role in ['subject','attractor']}))
    assert not OUT.exists();torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');is_id=enc.encode(' is')[0];old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text());modal=torch.tensor(old+controls['token_ids'],device='cuda');banks={k:v.to('cuda') for k,v in banks.items()}
    reference=np.array([0.,0.,1.,1.,1.,0.])[PARENTS];counts={k:0 for k in PLAN};records=[];replays=[];checks=[];selection=[]
    for context in contexts:
        dataset,panel,template=context['dataset'],context['panel'],context['template'];c=build_refined(model,context,modal);counts['prefix']+=c['prefix_calls'];checks.extend(c['refinement_checks']);n=len(c['raw'])
        def native(role,amplitudes):
            counts['native_suffix']+=1;pos,ds=c['source_components'][role]
            return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(amplitudes,device='cuda',dtype=torch.float64))
        base=native('subject',np.zeros((n,23)));families=[r['family'] for r in c['entries']];orientation=torch.where(c['pairs'][:,0,0]==is_id,1.,-1.).double()
        for role in ['subject','attractor']:
            pos,ds=c['source_components'][role];g=torch.einsum('od,bkd->bok',banks[role],ds);g[:,0]*=orientation[:,None]
            weights=[]
            for row in g.cpu().numpy():
                aa,check=choose(row,reference);weights.append(aa);selection.append(dict(dataset=dataset,panel=panel,template=template,role=role,**check))
            weights=np.array(weights);ref=base-native(role,np.tile(reference,(n,1)));effect=base-native(role,weights)
            prediction=-torch.einsum('bok,bk->bo',g,torch.tensor(weights,device='cuda',dtype=torch.float64))
            for family in dict.fromkeys(families):
                ids=[i for i,f in enumerate(families) if f==family];v=effect[ids];b=ref[ids,0];den=v[:,0].norm().clamp_min(1e-30)
                if dataset=='v2_replay':
                    prior=next(r for r in oldresult['records'] if r['family']==family and r['role']==role and r['arm']=='role_bank');replays.append(float((v-torch.tensor(prior['effect'],device='cuda',dtype=torch.float64)).abs().max()))
                    prior=next(r for r in oldunit['records'] if r['panel']==panel and r['family']==family and r['role']==role and r['arm']=='unitB');replays.append(float((ref[ids]-torch.tensor(prior['effect'],device='cuda',dtype=torch.float64)).abs().max()))
                retention=float(v[:,0]@b/b.square().sum().clamp_min(1e-30));collateral=float(v[:,1:].norm(dim=0).max()/den);error=(prediction[ids]-v).norm(dim=0)/den
                records.append(dict(dataset=dataset,panel=panel,family=family,role=role,retention=retention,control_ratio=collateral,prediction_errors=error.tolist(),capability=float((base[ids,0]>0).double().mean()),baseline_margins=base[ids,0].tolist(),effect=v.tolist(),reference_effect=ref[ids].tolist(),prediction=prediction[ids].tolist(),amplitudes=weights[ids].tolist()))
    instrument=counts==PLAN and max(replays)<=1e-4 and max(c['collapse_error'] for c in checks)<=1e-10 and max(v['relative'] for c in checks for v in c['corrections'])<=.01
    fresh=[r for r in records if r['dataset']=='v3'];assert len(fresh)==16
    result=dict(binding_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),plan=PLAN,counts=counts,max_old_replay=max(replays),refinement_checks=checks,selection_checks=selection,records=records,predictions=dict(pred_a_instrument=instrument,pred_b_selectivity=instrument and all(r['retention']>=.8 and r['control_ratio']<=.1 for r in fresh),pred_c_prediction=instrument and all(r['prediction_errors'][0]<=.1 and max(r['prediction_errors'][1:])<=.05 for r in fresh),pred_d_capability=all(r['capability']>=.9 for r in fresh)),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,max_old_replay=max(replays),selectivity_passes=sum(r['retention']>=.8 and r['control_ratio']<=.1 for r in fresh),prediction_passes=sum(r['prediction_errors'][0]<=.1 and max(r['prediction_errors'][1:])<=.05 for r in fresh),seconds=result['seconds'])))
if __name__=='__main__':main()
