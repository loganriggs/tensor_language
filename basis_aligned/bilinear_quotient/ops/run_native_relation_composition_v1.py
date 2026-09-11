#!/usr/bin/env python3
# BQGATE: 0 body forwards, 1408 native tail rows, two fixed cached panels, no fitting.
"""Composition and removal of the fixed leading/remainder suffix split.
A bound split, finite and previous full-swap margin replay<=1e-4 nats.
B leading swap mean donor-margin effect>=.05 each task/direction/panel.
C remainder/full mean effect>=.2 nouns and noun-minus-verb fraction>=.1 eachpanel.
D margin nonadditivity<=.1meanabswhole eachtask/panel; CE nonadditivity<=.05 eachfamily/panel.
E whole removal attenuates task contrast>=10%eachcell and every zero-arm
  meanabsCE<=.05 on unrelated C. P retains grammar and is not removal collateral.
Native complement remains; original ordinary replacement failure is not repaired.
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
STEM='NATIVE_RELATION_COMPOSITION_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(path)==sha for path,sha in binding.items())
    receipt=json.loads((P/'NATIVE_RELATION_SPLIT_V1.json').read_text());assert receipt['pred_a']
    assert digest(P/'NATIVE_RELATION_SPLIT_V1.pt')==receipt['artifact_sha256']
    program=torch.load(P/'NATIVE_RELATION_SPLIT_V1.pt',weights_only=True,map_location='cpu')
    prepared=[]
    for label,stem,prior in [('original','FROZEN_BRANCH_MORPHOLOGY_V1','NATIVE_RELATION_TRACE_PHYSICAL_V1'),
                             ('holdout','NATIVE_RELATION_HOLDOUT_V1','NATIVE_RELATION_HOLDOUT_V1')]:
        cache=torch.load(P/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu')
        rows=json.loads((P/(stem+'_ROWS.json')).read_text())['rows']
        previous=json.loads((P/(prior+'_RESULT.json')).read_text());assert previous['pred_a']
        assert cache['rows_sha256']==digest(P/(stem+'_ROWS.json'))
        leading,remainder=evaluate(program,cache['ports']['input'].double())
        prepared.append((label,rows,cache,previous,leading@program['writers'].T,remainder@program['writers'].T))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,tail_rows=1408,variable_products=45,fitting=False)));return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    started=time.perf_counter();torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=next(path for path in binding if path.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    unembedding=weights['lm_head.weight'].float().cuda()
    panels={};errors=[];tail_rows=0
    for label,rows,cache,previous,common,rest in prepared:
        h=(cache['ports']['pre']+cache['ports']['native_output']).cuda();records=[]
        for i,row in enumerate(rows):
            b,d=2*i,2*i+1;cb,cd=common[b],common[d];rb,rd=rest[b],rest[d]
            dc,dr=cd-cb,rd-rb
            def add(index,write):return h[index]+write.float().cuda()
            states=torch.stack([h[b],h[d],add(b,dc),add(b,dr),add(b,dc+dr),
                add(b,-cb),add(d,-cd),add(b,-rb),add(d,-rd),add(b,-cb-rb),add(d,-cd-rd)])
            logits=30*torch.tanh(F.linear(F.rms_norm(states,(1152,)),unembedding)/30);tail_rows+=len(states)
            margin=(logits[:,row['donor_answer_id']]-logits[:,row['donor_foil_id']]).double()
            # CE at donor states uses donor's own answer, not base's answer.
            targets=torch.tensor([row['donor_answer_id'] if j in (1,6,8,10) else row['base_answer_id'] for j in range(11)],device='cuda')
            ce=F.cross_entropy(logits.double(),targets,reduction='none')
            gap=float(margin[1]-margin[0]);effects=(margin[2:5]-margin[0]).cpu().tolist()
            old=previous['records'][i];assert old['row_id']==row['row_id']
            old_effect=old['margin_shifts']['approx3'] if label=='original' else old['effects'][1]
            errors += [abs(gap-old['native_gap']),abs(effects[2]-old_effect)]
            removals={}
            for name,base_index in [('leading',5),('remainder',7),('whole',9)]:
                removals[name]=dict(contrast_attenuation=gap-float(margin[base_index+1]-margin[base_index]),
                    base_ce_change=float(ce[base_index]-ce[0]),donor_ce_change=float(ce[base_index+1]-ce[1]))
            swap_ce=(ce[2:5]-ce[0]).cpu().tolist()
            records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],
                native_gap=gap,swap_margin_effects=effects,swap_ce_changes=swap_ce,
                margin_nonadditivity=effects[2]-effects[0]-effects[1],
                ce_nonadditivity=swap_ce[2]-swap_ce[0]-swap_ce[1],removals=removals))
        cells=[];families={}
        for family in ('A1','A2','P','C'):
            local=[row for row in records if row['family']==family]
            means=[sum(row['swap_margin_effects'][j] for row in local)/len(local) for j in range(3)]
            whole_abs=sum(abs(row['swap_margin_effects'][2]) for row in local)/len(local)
            mn=sum(abs(row['margin_nonadditivity']) for row in local)/len(local)
            cn=sum(abs(row['ce_nonadditivity']) for row in local)/len(local)
            zero_ce={name:sum(abs(row['removals'][name][key]) for row in local for key in ('base_ce_change','donor_ce_change'))/(2*len(local))
                     for name in ('leading','remainder','whole')}
            families[family]=dict(swap_mean_effects=means,remainder_fraction=means[1]/means[2] if means[2]!=0 else None,
                margin_mean_abs_nonadditivity=mn,margin_relative_nonadditivity=mn/whole_abs if whole_abs else None,
                ce_mean_abs_nonadditivity=cn,removal_mean_abs_ce=zero_ce)
            if family not in ('A1','A2'):continue
            for direction in ('base_to_suffix','suffix_to_base'):
                subset=[row for row in local if row['direction']==direction]
                gap=sum(row['native_gap'] for row in subset)/len(subset)
                leading_mean=sum(row['swap_margin_effects'][0] for row in subset)/len(subset)
                attenuation=sum(row['removals']['whole']['contrast_attenuation'] for row in subset)/len(subset)
                cells.append(dict(family=family,direction=direction,n=len(subset),native_gap=gap,
                    leading_mean=leading_mean,whole_removal_contrast_attenuation=attenuation,whole_removal_fraction=attenuation/gap))
        verb,noun=families['A1'],families['A2']
        panels[label]=dict(cells=cells,families=families,records=records,
            leading_held=all(c['leading_mean']>=.05 for c in cells),
            specificity_held=noun['swap_mean_effects'][2]>0 and verb['swap_mean_effects'][2]>0 and
                noun['remainder_fraction']>=.2 and noun['remainder_fraction']-verb['remainder_fraction']>=.1,
            composition_held=all(families[f]['margin_relative_nonadditivity']<=.1 for f in ('A1','A2')) and
                all(v['ce_mean_abs_nonadditivity']<=.05 for v in families.values()),
            removal_held=all(c['whole_removal_fraction']>=.1 for c in cells) and
                all(v<=.05 for v in families['C']['removal_mean_abs_ce'].values()))
    assert tail_rows==1408
    a=max(errors)<=1e-4 and all(torch.isfinite(torch.tensor(row['swap_margin_effects'])).all() for p in panels.values() for row in p['records'])
    a=bool(a)
    result={'pred_a':a,'pred_b':a and all(p['leading_held'] for p in panels.values()),
        'pred_c':a and all(p['specificity_held'] for p in panels.values()),
        'pred_d':a and all(p['composition_held'] for p in panels.values()),
        'pred_e':a and all(p['removal_held'] for p in panels.values()),'panels':panels,
        'maximum_previous_margin_replay_error':max(errors),'effect_order':['leading','remainder','whole'],
        'price':{'body_forwards':0,'tail_rows':tail_rows,'variable_products':45,'native_background_retained':True},
        'seconds':time.perf_counter()-started,'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'scope':'Exact split of the previously frozen suffix program; native complement/background retained. '
        'Leading includes original radial/bias; remainder is trace-corrected remaining eigen terms. '
        'P has the same grammatical behavior and is not removal collateral. Existing panels, no new fitting, '
        'no arbitrary functional-offset adjustment, full native replacement or automatic circuit promotion.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='panels'},indent=2))
    print(json.dumps({label:{k:v for k,v in p.items() if k!='records'} for label,p in panels.items()},indent=2))


if __name__=='__main__':main()
