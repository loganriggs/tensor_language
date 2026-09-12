#!/usr/bin/env python3
# BQGATE:5bodyforwards,40sequences,max18tokens;2frozenbranches;300sec.
"""pred_a native replay/complete matched lengths; pred_b quote RMS<=20% frozen progressive;
pred_c native capability>=75%. Original unmatched control remains failed.
No fitting or exclusions, frozen branch8 plus descriptive branch3.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
STEM='MATCHED_PARTNER_QUOTE_CONTROL_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    authority=json.loads((P/(STEM+'_ROWS.json')).read_text());rows=authority['rows'];sequences=[];buckets={}
    for row in rows:
        for side in ('base','donor'):
            ids=row[side+'_ids'];assert row[side+'_prediction_position']==len(ids)-1
            i=len(sequences);sequences.append(ids);buckets.setdefault(len(ids),[]).append(i)
    assert all(len(row['base_ids'])==len(row['donor_ids'])==18 for row in rows)
    assert len(rows)==16 and len(sequences)==32 and max(buckets)==18 and sum((len(v)+7)//8 for v in buckets.values())==4
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=5,sequences=40,max_length=18,fitting=False)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PORTS.pt');assert not out.exists() and not ap.exists()
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17]
    captured={};counts=[0,0]
    def input_hook(module,args):captured['x16']=args[0].detach()
    def count_hook(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=5 and args[0].shape[1]<=18
    handles=[model.transformer.h[16].mlp.register_forward_pre_hook(input_hook),model.transformer.h[0].attn.register_forward_pre_hook(count_hook)]
    ports={k:torch.empty(32,1152) for k in ('input16','pre','native_output')};replay=None
    try:
        for length,indices in sorted(buckets.items()):
            for start in range(0,len(indices),8):
                selected=indices[start:start+8];tokens=torch.tensor([sequences[i] for i in selected],device='cuda')
                x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
                for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
                residual=last.lambdas[0]*x+last.lambdas[1]*x0
                attention,v1=last.attn(F.rms_norm(residual,(1152,)),v1);pre=residual+attention;native=last.mlp(F.rms_norm(pre,(1152,)))
                for key,value in [('input16',captured['x16']),('pre',pre),('native_output',native)]:ports[key][selected]=value[:,-1].cpu()
                if replay is None:
                    assert len(selected)==8
                    def head_hook(module,args,value):captured['head']=value[:,-1].detach()
                    hook=model.lm_head.register_forward_hook(head_hook)
                    try:model(tokens,tokens)
                    finally:hook.remove()
                    manual=30*torch.tanh(model.lm_head(F.rms_norm(pre[:,-1]+native[:,-1],(1152,)))/30)
                    physical=30*torch.tanh(captured['head']/30);replay=float((manual-physical).norm()/physical.norm())
    finally:
        for hook in handles:hook.remove()
    assert counts==[5,40]
    program=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True)
    l,r=[getattr(model.transformer.h[16].mlp,n).weight.double() for n in ('Left','Right')];xx=ports['input16'].double().cuda()
    values=((xx@l.T)*(xx@r.T))@program['compiled_producer_readers'][:,:9].cuda()
    den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps;alpha=values[:,0,None]*values/den[:,None]
    state=ports['pre']+ports['native_output'];u=model.lm_head.weight.detach().cpu();reports=[];writes={};finite=True
    for branch in (3,8):
        write=(alpha[:,branch,None]*program['output_writers'][:,branch].cuda()[None,:]).cpu();writes[str(branch)]=write
        effects=score([write,write,write],state,rows,u);finite=finite and effects['pred_a']
        swaps=torch.tensor(effects['reference_effects']['swaps'][0],dtype=torch.float64);zero=torch.tensor(effects['reference_effects']['zero_ce'][0],dtype=torch.float64);families=[]
        for family in sorted({row['family'] for row in rows}):
            ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
            families.append(dict(family=family,swap_mean=float(swaps[ids].mean()),swap_rms=float(swaps[ids].square().mean().sqrt()),negative_fraction=float((swaps[ids]<0).double().mean()),
                                 zero_ce_mean=float(zero[ep].mean()),zero_ce_meanabs=float(zero[ep].abs().mean())))
        reports.append(dict(branch=branch,families=families,swaps=swaps.tolist(),zero_ce=zero.tolist()))
    capability=[]
    for family in sorted({row['family'] for row in rows}):
        ids=[i for i,row in enumerate(rows) if row['family']==family];sides=[]
        for side,offset in [('base',0),('donor',1)]:
            readers=torch.stack([u[[rows[i][side+'_answer_id'],rows[i][side+'_foil_id']]] for i in ids]);z=torch.einsum('nd,nkd->nk',F.rms_norm(state[torch.tensor(ids)*2+offset],(1152,)),readers)
            sides.append(float((z[:,0]>z[:,1]).float().mean()))
        capability.append(dict(family=family,base_donor=sides))
    cells={f['family']:f for f in reports[1]['families']}
    result={'pred_a':finite and replay<=1e-5 and all(bool(torch.isfinite(v).all()) for v in ports.values()),
            'pred_b':cells['matched_quote']['swap_rms']<=.2*authority['reference_progressive_swap_rms'],
            'pred_c':all(min(c['base_donor'])>=.75 for c in capability)}
    torch.save(dict(ports=ports,writes=writes,values=values.cpu(),rows_sha256=digest(P/(STEM+'_ROWS.json')),program_sha256=digest(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt')),ap)
    result.update(reports=reports,native_capability=capability,counts=counts,replay=replay,artifact_sha=digest(ap),source_shas=binding,
                  execution_seconds=time.perf_counter()-tic,scope='Frozen weight-discovered branch on new lexemes/constructions, no refit/exclusions; conditional native background, not corpus OOD.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','reports')}),flush=True)

if __name__=='__main__':main()
