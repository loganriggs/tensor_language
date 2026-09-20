#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_recipient_root pred_c_gradient_transfer
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';A=ROOT/'basis_aligned/bilinear_quotient/circuits/followups';OUT=A/'readout_field_transfer_v1_result.json'
PLAN=dict(prefix=12,double_suffix=8,field_gradient=152,native_suffix=14)
def main():
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(PLAN));return
    import numpy as np
    import torch
    import tiktoken
    import circuit_fast_screen_producer as producer
    from disk_guard import guard_write
    sys.path.insert(0,str(P))
    from refined_native_sources import build_refined
    from native_source_observables import source_observables,source_observables32
    from final_readout_field_program import pack_fields,evaluate_fields
    from readout_field_differential import contract
    from align_readout_fields import align
    from shared_selective_source_lp import choose
    binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text());controls=json.loads((P/'EXPANDED_SOURCE_CONTROLS_V1_BINDING.json').read_text())
    source=A/'refined_selective_sources_v1_result.json';prior=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.])[prior['parent_mapping']]
    assert not OUT.exists()
    for filename,digest in binding['row_sha256'].items():assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==digest
    torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
    for parameter in model.parameters():parameter.requires_grad_(False)
    enc=tiktoken.get_encoding('gpt2');old=[[enc.encode(' '+w)[0] for w in pair] for pair in [['can','will'],['may','might'],['should','could']]]
    modal=torch.tensor(old+controls['token_ids'],device='cuda');counts={k:0 for k in PLAN};cache={};field_groups={};replays=[];records=[];gradient_records=[];selection_checks=[]
    for context in binding['contexts']:
        panel,template=context['panel'],context['template'];c=build_refined(model,context,modal);cache[panel,template]=c;counts['prefix']+=c['prefix_calls']
        for role in ['subject','attractor']:
            pos,ds=c['source_components'][role]
            with torch.enable_grad():
                amplitudes=torch.zeros(len(ds),23,device='cuda',dtype=torch.float64,requires_grad=True);capture={}
                output=source_observables(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],amplitudes,capture);counts['double_suffix']+=1
                state=capture['mlp_outputs'][17][c['batch'],c['read']]
                fields=pack_fields(state,model.lm_head.weight[c['pairs']].double())
                derivatives=torch.stack([torch.autograd.grad(fields[:,j].sum(),amplitudes,retain_graph=j<18)[0] for j in range(19)],dim=1);counts['field_gradient']+=19
            fields=fields.detach();derivatives=derivatives.detach()
            oldg=next(g for g in prior['gradients'] if g['panel']==panel and g['template']==template and g['role']==role)
            replays.extend([float((contract(fields,derivatives)-torch.tensor(oldg['gradient'],device='cuda',dtype=torch.float64)).abs().max()),float((evaluate_fields(fields)-output.detach()).abs().max())])
            field_groups[panel,template,role]=dict(fields=fields,derivatives=derivatives,tokens=c['pairs'])
    for (panel,template),c in cache.items():
        if panel!='congruent':continue
        n=len(c['raw']);families=[r['family'] for r in c['entries']]
        def native(role,amplitudes):
            counts['native_suffix']+=1;pos,ds=c['source_components'][role]
            return source_observables32(model,c['raw'],c['x0'],c['first'],ds,pos,c['read'],c['pairs'],torch.as_tensor(amplitudes,device='cuda',dtype=torch.float64))
        baseline=native('subject',np.zeros((n,23)))
        for role in ['subject','attractor']:
            donor=field_groups['opposite',template,role];current=field_groups[panel,template,role];indices=torch.arange(n,device='cuda')
            if role=='attractor':indices=indices^1
            df,dg=align(donor['fields'][indices],donor['derivatives'][indices],donor['tokens'][indices],current['tokens'])
            exact=contract(current['fields'],current['derivatives']);predicted=contract(current['fields'],dg);stale=contract(df,dg);root_only=contract(df,current['derivatives'])
            for family in dict.fromkeys(families):
                ids=[i for i,f in enumerate(families) if f==family];norm=exact[ids,0].norm()
                gradient_records.append(dict(role=role,family=family,errors={mode:dict(number=float((g-exact)[ids,0].norm()/norm.clamp_min(1e-30)),controls=float((g-exact)[ids,1:].permute(1,0,2).flatten(1).norm(dim=1).max()/norm.clamp_min(1e-30))) for mode,g in [('stale',stale),('recipient_root',predicted),('donor_root_only',root_only)]}))
            candidate=[]
            for row in predicted.cpu().numpy():
                weights,check=choose(row,reference);candidate.append(weights);selection_checks.append(check)
            donor_a=next(g for g in prior['amplitudes'] if g['panel']=='opposite' and g['template']==template and g['role']==role)['amplitudes']['oracle']
            own_a=next(g for g in prior['amplitudes'] if g['panel']=='congruent' and g['template']==template and g['role']==role)['amplitudes']['oracle']
            arms=dict(donor=np.array(donor_a)[indices.cpu().numpy()],recipient_root=np.array(candidate),oracle=np.array(own_a))
            for arm,aa in arms.items():
                effect=baseline-native(role,aa)
                for family in dict.fromkeys(families):
                    ids=[i for i,f in enumerate(families) if f==family]
                    old=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='unitB');ref=torch.tensor(old['effect'],device='cuda',dtype=torch.float64)[:,0];v=effect[ids]
                    if arm=='oracle':
                        previous=next(r for r in prior['records'] if r['panel']=='congruent' and r['role']==role and r['family']==family and r['arm']=='oracle');replays.append(float((v-torch.tensor(previous['effect'],device='cuda',dtype=torch.float64)).abs().max()))
                    retention=float(v[:,0]@ref/ref.square().sum());leak=float(v[:,1:].norm(dim=0).max()/v[:,0].norm().clamp_min(1e-30))
                    records.append(dict(role=role,family=family,arm=arm,retention=retention,control_ratio=leak,passed=bool(retention>=.8 and leak<=.1),effect=v.tolist()))
    instrument=counts==PLAN and max(replays)<=1e-8
    serialized=[dict(panel=k[0],template=k[1],role=k[2],**{name:t.tolist() for name,t in value.items()}) for k,value in field_groups.items()]
    result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),plan=PLAN,counts=counts,max_replay=max(replays),fields=serialized,gradient_records=gradient_records,records=records,selection_checks=selection_checks,predictions=dict(pred_a_instrument=instrument,pred_b_recipient_root=instrument and all(r['passed'] for r in records if r['arm']=='recipient_root'),pred_c_gradient_transfer=instrument and all(r['errors']['recipient_root']['number']<=.1 and r['errors']['recipient_root']['controls']<=.05 for r in gradient_records)),seconds=time.perf_counter()-tic)
    payload=json.dumps(result,separators=(',',':'))+'\n';guard_write(len(payload.encode()),label=OUT.name);OUT.write_text(payload)
    print(json.dumps(dict(predictions=result['predictions'],counts=counts,max_replay=max(replays),passes={arm:sum(r['passed'] for r in records if r['arm']==arm) for arm in arms},seconds=result['seconds'])))
if __name__=='__main__':main()
