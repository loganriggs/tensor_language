#!/usr/bin/env python3
# BQGATE: 0 body forwards, 1632 tail rows, frozen cached developmental panels, no fitting.
"""Physical comparison of within-span and ambient mean-null output branches.
A old whole swap/zero CE replay<=1e-4 nats, finite and bound sources.
B ambient-private swap>=80%whole and>=.05 each task/direction/panel.
C ambient-private neighbor zero meanabsCE<=.05 and absolute contrast attenuation<=5%nativegap eachfamily.
D ambient-private target contrast attenuation>=80%whole and>=10%nativegap eachtask/direction/panel.
E inside-private swap<=50%whole each task/panel, directions pooled.
Null: mean-null weights fail tokenwise preservation or lose task effect.
Two materialized branch writer matrices cost6912floats versus3456; same45products/nativebackground.
Known panels are developmental validation, not independent confirmation.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
from native_relation_split_v1 import evaluate
STEM='NATIVE_RELATION_OUTPUT_SPLIT_PHYSICAL_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    saved=torch.load(P/'NATIVE_RELATION_OUTPUT_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    program=torch.load(P/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    composition=json.loads((P/'NATIVE_RELATION_COMPOSITION_V1_RESULT.json').read_text())
    neighbor=json.loads((P/'NATIVE_RELATION_NEIGHBOR_V1_RESULT.json').read_text())
    panels=[]
    for panel,stem in [('original','FROZEN_BRANCH_MORPHOLOGY_V1'),('holdout','NATIVE_RELATION_HOLDOUT_V1'),('neighbor','NATIVE_RELATION_NEIGHBOR_V1')]:
        rows=json.loads((P/(stem+'_ROWS.json')).read_text())['rows']
        cache=torch.load(P/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu')
        assert cache['rows_sha256']==digest(P/(stem+'_ROWS.json'))
        lead,rest=evaluate(program,cache['ports']['input'].double());scalar=lead+rest
        writers=[saved['original_writers']]+[saved['splits'][split][branch+'_writers'] for split in ('inside_span','ambient') for branch in ('private','shared')]
        writes=[scalar@v.T for v in writers]
        panels.append((panel,rows,cache,writes))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,tail_rows=1632,fitting=False)));return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    started=time.perf_counter();torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].float().cuda();records=[];replay=[];tail_rows=0
    for panel,rows,cache,writes in panels:
        h=(cache['ports']['pre']+cache['ports']['native_output']).cuda()
        for i,row in enumerate(rows):
            if panel!='neighbor' and row['family'] not in ('A1','A2'):continue
            b,d=2*i,2*i+1;states=[h[b],h[d]]
            for w in writes:
                states.extend([h[b]+(w[d]-w[b]).float().cuda(),h[b]-w[b].float().cuda(),h[d]-w[d].float().cuda()])
            z=30*torch.tanh(F.linear(F.rms_norm(torch.stack(states),(1152,)),u)/30);tail_rows+=len(states)
            margins=(z[:,row['donor_answer_id']]-z[:,row['donor_foil_id']]).double()
            targets=torch.tensor([row['donor_answer_id'] if j==1 or (j>=2 and (j-2)%3==2) else row['base_answer_id'] for j in range(17)],device='cuda')
            ce=F.cross_entropy(z.double(),targets,reduction='none')
            arms=[]
            for arm in range(5):
                k=2+3*arm
                arms.append(dict(swap_effect=float(margins[k]-margins[0]),
                    zero_ce=[float(ce[k+1]-ce[0]),float(ce[k+2]-ce[1])],
                    contrast_attenuation=float((margins[1]-margins[0])-(margins[k+2]-margins[k+1]))))
            if panel=='neighbor':
                old=neighbor['records'][i]
                for side in (0,1):replay.append(abs(arms[0]['zero_ce'][side]-(old['endpoints'][side]['ce'][3]-old['endpoints'][side]['ce'][0])))
            # Original/holdout share the same prior composition receipt schema.
            else:
                old=composition['panels'][panel]['records'][i]
                assert old['row_id']==row['row_id']
                replay.append(abs(arms[0]['swap_effect']-old['swap_margin_effects'][2]))
                for side,key in enumerate(('base_ce_change','donor_ce_change')):
                    replay.append(abs(arms[0]['zero_ce'][side]-old['removals']['whole'][key]))
            records.append(dict(panel=panel,family=row['family'],direction=row.get('direction','base_to_inflected'),
                row_id=row['row_id'],native_gap=float(margins[1]-margins[0]),arms=arms))
    assert tail_rows==1632
    cells=[]
    for panel in ('original','holdout','neighbor'):
        families=('past','progressive') if panel=='neighbor' else ('A1','A2')
        directions=('base_to_inflected',) if panel=='neighbor' else ('base_to_suffix','suffix_to_base')
        for family in families:
            for direction in directions:
                local=[r for r in records if r['panel']==panel and r['family']==family and r['direction']==direction]
                gap=sum(r['native_gap'] for r in local)/len(local)
                arms=[dict(swap_effect=sum(r['arms'][j]['swap_effect'] for r in local)/len(local),
                    zero_meanabs_ce=sum(abs(v) for r in local for v in r['arms'][j]['zero_ce'])/(2*len(local)),
                    contrast_attenuation=sum(r['arms'][j]['contrast_attenuation'] for r in local)/len(local)) for j in range(5)]
                cells.append(dict(panel=panel,family=family,direction=direction,n=len(local),native_gap=gap,arms=arms))
    target=[c for c in cells if c['panel']!='neighbor'];control=[c for c in cells if c['panel']=='neighbor']
    weak=[]
    for panel in ('original','holdout'):
        for family in ('A1','A2'):
            local=[c for c in target if c['panel']==panel and c['family']==family]
            whole=sum(c['arms'][0]['swap_effect'] for c in local);private=sum(c['arms'][1]['swap_effect'] for c in local)
            weak.append(whole>0 and private<=.5*whole)
    a=max(replay)<=1e-4 and all(torch.isfinite(torch.tensor([r['native_gap']]+[n for v in r['arms'] for n in [v['swap_effect'],v['contrast_attenuation']]+v['zero_ce']])).all() for r in records)
    result={'pred_a':bool(a),
        'pred_b':bool(a) and all(c['arms'][0]['swap_effect']>0 and c['arms'][3]['swap_effect']>=max(.05,.8*c['arms'][0]['swap_effect']) for c in target),
        'pred_c':bool(a) and all(c['native_gap']>0 and c['arms'][3]['zero_meanabs_ce']<=.05 and abs(c['arms'][3]['contrast_attenuation'])<=.05*c['native_gap'] for c in control),
        'pred_d':bool(a) and all(c['native_gap']>0 and c['arms'][0]['contrast_attenuation']>0 and c['arms'][3]['contrast_attenuation']>=max(.8*c['arms'][0]['contrast_attenuation'],.1*c['native_gap']) for c in target),
        'pred_e':bool(a) and all(weak),'cells':cells,'records':records,'max_replay_nats':max(replay),
        'arm_order':['whole','inside_private','inside_shared','ambient_private','ambient_shared'],
        'tail_rows':tail_rows,'seconds':time.perf_counter()-started,'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'scope':'Developmental known-panel validation of frozen weight-defined output splits. No new OOD, factor fitting, independent circuit identification or replacement adoption.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
