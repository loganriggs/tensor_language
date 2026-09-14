#!/usr/bin/env python3
"""Correct V1's cyclic gate reference while retaining the recipient reader."""
import copy,hashlib,json,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=POLY/'GERUND_DEPTH_INTERVAL_V2_RESULT.json'
BIND=POLY/'GERUND_DEPTH_INTERVAL_V2_BINDING.json'
V1=POLY/'GERUND_DEPTH_INTERVAL_V1_RESULT.json'
STATES=POLY/'GERUND_DEPTH_INTERVAL_V1_STATES.pt'
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def serial(x):return x.detach().cpu().tolist()
def metric(change,reference):
    return {'transfer':float((change*reference).sum()/reference.square().sum()),'relative_error':float((change-reference).norm()/reference.norm()),'change_per_row':serial(change),'reference_per_row':serial(reference)}
def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    source=json.loads(V1.read_text());saved=torch.load(STATES,map_location='cpu',weights_only=True);result=copy.deepcopy(source)
    for name in ['A1','A2','G']:
        s=saved[name];kp=s['context_reader'].double();base=s['base_u17'].double();tau=(kp*base).sum(-1);reference=(kp*base.roll(-1,0)).sum(-1)-tau
        result['checks'][name+'_corrected_gate_reference_norm']=float(reference.norm())
        for label,state in s['arms'].items():
            change=(kp*state['u17'].double()).sum(-1)-tau
            result['reports'][name]['arms'][label]['gate']=metric(change,reference)
    valid=source['predictions']['pred_a_instrument']
    valid=valid and all(abs(result['reports'][n]['arms']['pre_mlp17']['gate']['transfer']-1)<=1e-12 and result['reports'][n]['arms']['pre_mlp17']['gate']['relative_error']<=1e-12 for n in ['A1','A2','G'])
    def both(label,field,tmin=None,emax=None):
        vals=[result['reports'][n]['arms'][label][field] for n in ['A1','A2']]
        return all((tmin is None or v['transfer']>=tmin) and (emax is None or v['relative_error']<=emax) for v in vals)
    onset=all(result['reports'][n]['arms']['prefix7']['gate']['transfer']<=.5 and result['reports'][n]['arms']['prefix11']['gate']['transfer']>=.5 and result['reports'][n]['arms']['prefix13']['gate']['transfer']>=.65 for n in ['A1','A2'])
    result['schema']='gerund.depth_interval.v2'
    result['predictions']={'pred_a_instrument':bool(valid),'pred_b_distributed_target_band':bool(valid and both('band7_pre17','gate',.75,.5)),'pred_c_cumulative_onset':bool(valid and onset),'pred_d_norm_state_sufficiency':bool(valid and both('band7_pre17','norm_moment',.5,.75)),'pred_e_collateral_control':bool(valid and result['reports']['G']['arms']['band7_pre17']['mean_absolute_ce']<=.15)}
    result['v1_result_sha256']=digest(V1);result['v1_states_sha256']=digest(STATES);result['runner_sha256']=digest(__file__);result['binding_sha256']=digest(BIND)
    result['price']['new_body_forwards']=0;result['price']['new_sequences']=0
    result['scope']='Corrected fixed-recipient-reader audit of immutable V1 native states. V1 gate metrics are withdrawn; norm moments and CE are unchanged. Native weights, donor generation, positions, MLP17 and suffix remain external.'
    assert not OUT.exists();tmp=OUT.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,OUT)
    print(json.dumps({'predictions':result['predictions'],'target':{n:{a:result['reports'][n]['arms'][a]['gate'] for a in ['prefix7','prefix11','prefix13','band7_pre17','band11_pre17','pre_mlp17']} for n in ['A1','A2']},'g_control':result['reports']['G']['arms']['band7_pre17']['mean_absolute_ce']}))
if __name__=='__main__':main()
