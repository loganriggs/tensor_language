#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;1024cachedendpoints,10branchinterventions;300sec.
"""pred_a execution; pred_b task concentration2x in6/8contexts with RMS>=1e-3;
pred_c target scalar relative change>=.01; pred_d joint swap additivity<=.1.
Frozen self+8 SVD branches. Null: token readout is not selective computation.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
STEM='MATCHED_PARTNER_BRANCH_SCREEN_V1'

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=1024,branch_interventions=10)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    program=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True)
    cache=torch.load(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_PORTS.pt',weights_only=True,mmap=True)
    assert cache['rows_sha256']==digest(P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json')
    rows=json.loads((P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json').read_text())['rows'];assert len(rows)==512 and len({r['family'] for r in rows})==32
    ports=cache['ports'];x=ports['input16'].double().cuda()
    l,r=[sd[f'transformer.h.16.mlp.{name}.weight'].double().cuda() for name in ('Left','Right')]
    c=program['compiled_producer_readers'][:,:9].cuda();w=program['output_writers'][:,:9].cuda()
    values=((x@l.T)*(x@r.T))@c;den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    alpha=values[:,0,None]*values/den[:,None]
    writes=[alpha[:,j,None]*w[:,j][None,:] for j in range(9)]
    joint=writes[3]+writes[8];joint_direct=alpha[:,[3,8]]@w[:,[3,8]].T
    identity=float((joint-joint_direct).norm()/joint.norm());writes.append(joint)
    torch.set_default_dtype(torch.float32)
    state=ports['pre']+ports['native_output'];u=sd['lm_head.weight'].float()
    reports=[];finite=True
    for j,write in enumerate(writes):
        effects=score([write.cpu(),write.cpu(),write.cpu()],state,rows,u)
        finite=finite and effects['pred_a'];zero=torch.tensor(effects['reference_effects']['zero_ce'][0],dtype=torch.float64);swap=torch.tensor(effects['reference_effects']['swaps'][0],dtype=torch.float64)
        families=[]
        for family in sorted({row['family'] for row in rows}):
            ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
            cell=dict(family=family,swap_rms=float(swap[ids].square().mean().sqrt()),swap_mean=float(swap[ids].mean()),
                      zero_ce_mean=float(zero[ep].mean()),zero_ce_meanabs=float(zero[ep].abs().mean()))
            if j<9:
                aa=alpha[:,j].cpu();drms=(aa[2*ids+1]-aa[2*ids]).square().mean().sqrt();rms=aa[ep].square().mean().sqrt()
                cell.update(coefficient_rms=float(rms),coefficient_delta_rms=float(drms),coefficient_relative_change=float(drms/rms.clamp_min(1e-30)))
            families.append(cell)
        reports.append(dict(branch=j if j<9 else 'joint3+8',families=families,swaps=swap.tolist(),zero_ce=zero.tolist()))
    targets=[]
    for branch,task in [(3,'past'),(8,'progressive')]:
        cells={f['family']:f for f in reports[branch]['families']};contexts=[]
        for context in range(8):
            target=cells[f'{task}:context{context}'];other=max(cells[f'{t}:context{context}']['swap_rms'] for t in ('A1','A2','past','progressive') if t!=task)
            contexts.append(dict(context=context,target_rms=target['swap_rms'],max_other_rms=other,ratio=target['swap_rms']/max(other,1e-30),
                                 concentrated=target['swap_rms']>=max(.001,2*other),coefficient_relative_change=target['coefficient_relative_change']))
        targets.append(dict(branch=branch,task=task,contexts=contexts,passed_contexts=sum(v['concentrated'] for v in contexts)))
    interaction=[];a,b,joint=[torch.tensor(reports[k]['swaps'],dtype=torch.float64) for k in (3,8,9)]
    for family in sorted({row['family'] for row in rows}):
        ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family]);err=joint[ids]-a[ids]-b[ids];ref=joint[ids]
        interaction.append(dict(family=family,relative_rms=float(err.norm()/ref.norm().clamp_min(1e-30)),joint_rms=float(ref.square().mean().sqrt()),signed_interaction_mean=float(err.mean())))
    capability=[]
    for family in sorted({row['family'] for row in rows}):
        ids=[i for i,row in enumerate(rows) if row['family']==family];sides=[]
        for side,offset in [('base',0),('donor',1)]:
            readers=torch.stack([u[[rows[i][side+'_answer_id'],rows[i][side+'_foil_id']]] for i in ids]);logits=torch.einsum('nd,nkd->nk',F.rms_norm(state[torch.tensor(ids)*2+offset],(1152,)),readers)
            sides.append(float((logits[:,0]>logits[:,1]).float().mean()))
        capability.append(dict(family=family,base_donor=sides))
    result={'pred_a':finite and identity<=1e-8,'pred_b':all(t['passed_contexts']>=6 for t in targets),
            'pred_c':all(c['coefficient_relative_change']>=.01 for t in targets for c in t['contexts']),
            'pred_d':all(i['relative_rms']<=.1 and i['joint_rms']>=1e-4 for i in interaction)}
    result.update(reports=reports,targets=targets,interaction=interaction,native_capability=capability,joint_identity=identity,
                  source_shas=binding,execution_seconds=time.perf_counter()-tic,scope='Frozen weights; inspected context panel; conditional output-port manipulation, no OOD or semantic promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','reports','native_capability','interaction')}),flush=True)

if __name__=='__main__':main()
