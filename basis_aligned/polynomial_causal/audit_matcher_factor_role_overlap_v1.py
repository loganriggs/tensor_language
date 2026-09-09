"""All role cells and exact role-permutation energy baseline; no circuit replacement."""
import hashlib
import itertools
import json
from pathlib import Path
import torch
import suffix_join_middle_match_reference as R

BASE=Path(__file__).resolve().parent;INPUT=BASE/'MATCHER_FACTOR_TRUTH_TABLE_V1_ROWS.pt'
OUT=BASE/'MATCHER_FACTOR_ROLE_OVERLAP_V1.json'
ROLES=('key_from_key','key_from_value','value_from_key','value_from_value')


def energy(a,b):
    a2=a.square();b2=b.square();joint=float((a2*b2).mean())
    independent=float(a2.mean()*b2.mean());permuted=float((a2.mean(-1)*b2.mean(-1)).mean())
    return {'joint_mean_square':joint,'independent_global_energy':independent,
        'global_squared_factor_covariance':joint-independent,'exact_role_permutation_mean_square':permuted,
        'within_row_role_covariance':joint-permuted,'native_to_role_permuted_rms':(joint/max(permuted,1e-30))**.5}


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest()=='d0dd9dd724a054e9a7915b0331d8b5df354004f1ad159b5dfe408edb884a62bf'
    a=torch.tensor([[1.,2.,3.,4.],[4.,2.,1.,3.]],dtype=torch.float64);b=a.flip(-1)
    enumerated=sum(float((a*b[:,p]).square().mean()) for p in itertools.permutations(range(4)))/24
    assert abs(enumerated-energy(a,b)['exact_role_permutation_mean_square'])<1e-12
    data=torch.load(INPUT,map_location='cpu',weights_only=True);inputs=R.populations();groups={};maximum=0.
    for pop,block in data.items():
        tokens,masks,meta=inputs[pop];assert torch.equal(tokens,block['tokens']) and meta==block['metadata']
        for mask in masks:
            parity=mask.nonzero()%2
            assert torch.equal(parity,torch.tensor([[0,0],[0,1],[1,0],[1,1]]))
        factors=block['factors'];maximum=max(maximum,float((factors['factor1']*factors['factor2']-factors['joint']).abs().max()))
        groups[pop]={}
        for orientation in ('B2_later','B3_later'):
            case_results={}
            for case in R.CASES:
                sel=torch.tensor([m['orientation']==orientation and m['case']==case for m in meta]);x=factors['factor1'][sel];y=factors['factor2'][sel]
                cells={role:{name:{'rms':float(value[sel,k].square().mean().sqrt()),'mean':float(value[sel,k].mean())}
                    for name,value in factors.items()} for k,role in enumerate(ROLES)}
                case_results[case]={'rows':int(sel.sum()),'energy':energy(x,y),'roles':cells}
            den_native=min(case_results[c]['energy']['joint_mean_square']**.5 for c in ('base','both'))
            den_permuted=min(case_results[c]['energy']['exact_role_permutation_mean_square']**.5 for c in ('base','both'))
            ratios={c:{'native_mismatch_ratio':case_results[c]['energy']['joint_mean_square']**.5/den_native,
                'role_permuted_energy_mismatch_ratio':case_results[c]['energy']['exact_role_permutation_mean_square']**.5/den_permuted} for c in ('keys','values')}
            groups[pop][orientation]={'cases':case_results,'mismatch_ratios':ratios}
    result={'scope':'opened complete role-cell diagnostic; exact permutation-averaged energy baseline is not a valid native intervention or adopted circuit',
        'controls_passed':True,'role_parity_valid':True,'product_replay_max_abs':maximum,'groups':groups,
        'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),'native_coefficients_removed':0}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert maximum<1e-9
    print(json.dumps({'product_replay_max_abs':maximum,'ratios':{p:{o:g['mismatch_ratios'] for o,g in v.items()} for p,v in groups.items()},
        'overlap_gain':{p:{o:{c:r['energy']['native_to_role_permuted_rms'] for c,r in g['cases'].items()} for o,g in v.items()} for p,v in groups.items()}},indent=2))


if __name__=='__main__':main()
