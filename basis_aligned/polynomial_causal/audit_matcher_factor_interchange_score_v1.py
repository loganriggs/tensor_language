"""Fixed factor swaps through the other native factor; all source-role cells kept."""
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent;INPUT=BASE/'MATCHER_FACTOR_TRUTH_TABLE_V1_ROWS.pt'
OUT=BASE/'MATCHER_FACTOR_INTERCHANGE_SCORE_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()=='d0dd9dd724a054e9a7915b0331d8b5df354004f1ad159b5dfe408edb884a62bf'
    data=torch.load(INPUT,map_location='cpu',weights_only=True);groups={};mixed_error=joint_error=0.
    for pop,block in data.items():
        meta=block['metadata'];f1=block['factors']['factor1'].reshape(-1,4,4);f2=block['factors']['factor2'].reshape(-1,4,4)
        assert all([m['case'] for m in meta[i:i+4]]==['base','keys','values','both'] for i in range(0,len(meta),4))
        scores={'00':f1[:,:1]*f2[:,:1],'10':f1*f2[:,:1],'01':f1[:,:1]*f2,'11':f1*f2}
        mixed=scores['11']-scores['10']-scores['01']+scores['00']
        mixed_error=max(mixed_error,float((mixed-(f1-f1[:,:1])*(f2-f2[:,:1])).abs().max()))
        joint_error=max(joint_error,float((scores['11'].flatten(0,1)-block['factors']['joint']).abs().max()))
        groups[pop]={}
        for orientation in ('B2_later','B3_later'):
            sel=torch.tensor([meta[i]['orientation']==orientation for i in range(0,len(meta),4)])
            baseline=float(scores['00'][sel].square().mean().sqrt());arms={}
            for name in ('10','01','11'):
                ratios={case:float(scores[name][sel,j].square().mean().sqrt())/max(baseline,1e-8) for j,case in enumerate(('base','keys','values','both'))}
                arms[name]={'rms_over_base':ratios,'consumer_conditioned_suppression_passed':
                    baseline>1e-8 and ratios['keys']<=.10 and ratios['values']<=.10 and ratios['both']>=.50}
            groups[pop][orientation]={'base_joint_rms':baseline,'arms':arms}
    result={'scope':'opened score-level factor interchange through a fixed native multiplicative reader; not full-output causal identification',
        'mixed_identity_max_abs':mixed_error,'joint_replay_max_abs':joint_error,'groups':groups,
        'native_coefficients_removed':0,'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert mixed_error<1e-12 and joint_error<1e-12;print(json.dumps(result,indent=2))


if __name__=='__main__':main()
