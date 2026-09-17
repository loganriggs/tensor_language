#!/usr/bin/env python3
# BQGATE:40bodyforwards;8prefixes;300seconds;no fitting.
"""pred_a carried/native anchors <=1e-5 abs and1e-6 relative; live writes.
pred_b both-frozen target-effect error <=.10 in each cue/endpoint.
pred_c full two-normalizer factorial closure <=1e-12. No fresh adoption.
"""
import json,hashlib,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import typed_face_key_norm_v1 as norm
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v2 import measure
from run_even_value_factorial_native_v1 import setup
STEM='TYPED_FACE_KEY_NORM_V1'
D=P/'extracted_circuits/odd_attention8h2_typed_face_v1'
ARMS=['native','carried','freeze_k1','freeze_k2','freeze_both']
MASKS={'carried':0,'freeze_k1':1,'freeze_k2':2,'freeze_both':3}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(k)==v for k,v in binding.items())
    rows=json.loads((P/'ODD_ATTENTION8H2_TYPED_FACE_REMOVAL_V1_ROWS.json').read_text())['rows']
    groups,mapping=group_rows(rows);assert len(groups)==8
    assert json.loads((P/(STEM+'_CPU_RESULT.json')).read_text())['pred_a']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        assert sorted(MASKS.values())==[0,1,2,3]
        print('40bodyforwards;8prefixes;two-key-normalizer full factorial;CPU native replay passes');return
    out=P/(STEM+'_RESULT.json');assert not out.exists()
    start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();graph,_,_,_=setup('cuda')
    program={k:v.to('cuda') for k,v in torch.load(D/'program.pt',weights_only=True).items()}
    def write(arm,row,donor_row,current,donor,mask):
        city=row['city_position']
        return norm.execute(program,current,donor[:,city],row['ids'][city],donor_row['ids'][city],city,mask,freeze_mask=MASKS[arm])
    measured=measure(model,graph,groups,ARMS,write);v=expand(measured['values'],mapping,6)
    parent=torch.load(P/'REGIONAL_ENDPOINT_BATCHING_V1_ARTIFACT.pt',map_location='cpu',weights_only=True)['values']
    anchor=v[:2]-parent[[0,6]];anchor_abs=float(anchor.abs().max());anchor_rel=float(anchor.norm()/parent[[0,6]].norm())
    effect=v-v[:1];reference=effect[1,:,0]
    cells={'all':list(range(48))}
    cells.update({cue:[i for i,r in enumerate(rows) if r['cue']==cue] for cue in ['British','American']})
    cells.update({f'endpoint_{e}':[i for i,r in enumerate(rows) if r['endpoint']==e] for e in range(6)})
    metrics={}
    for name,ids in cells.items():
        target=reference[ids]
        metrics[name]={arm:float((effect[ai,ids,0]-target).norm()/target.norm().clamp_min(1e-8)) for ai,arm in enumerate(ARMS) if ai>=2}
    interaction=v[4]-v[2]-v[3]+v[1]
    closure=float((v[2]+v[3]-v[1]+interaction-v[4]).abs().max())
    readout_errors=((effect[4]-effect[1]).norm(dim=0)/effect[1].norm(dim=0).clamp_min(1e-8)).tolist()
    pred_a=anchor_abs<=1e-5 and anchor_rel<=1e-6 and max(measured['outside'])==0 and min(min(x) for x in measured['write_norms'].values())>=1e-8
    pred_b=all(x['freeze_both']<=.1 for x in metrics.values())
    pred_c=closure<=1e-12
    result={'pred_a':pred_a,'pred_b':pred_b,'pred_c':pred_c,'terminal':'normalizers_omissible_screen' if all((pred_a,pred_b,pred_c)) else 'normalizer_omission_failed_gate',
      'arms':ARMS,'target_effect_errors':metrics,'both_frozen_effect_error_by_readout':readout_errors,
      'interaction_over_carried_effect':float(interaction[:,0].norm()/reference.norm()),
      'anchor_max_abs':anchor_abs,'anchor_relative':anchor_rel,'factorial_closure':closure,
      'body_forwards':measured['body_forwards'],'seconds':time.perf_counter()-start,
      'panel_status':'opened four-context response diagnosis','source_shas':binding,
      'scope':'Freeze donor-city head-key RMS factors only. Block8 input RMS and native suffix remain. No fresh omission adoption, independent composition or complete circuit claim.'}
    torch.save({'values':v,'effects':effect,'interaction':interaction},P/(STEM+'_ARTIFACT.pt'))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'},indent=2));signal.alarm(0)
if __name__=='__main__':main()
