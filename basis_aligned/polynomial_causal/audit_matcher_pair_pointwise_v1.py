"""All matched source pairs: sign/gain changes versus actual gate-transfer outputs."""
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent
SCORES=BASE/'MATCHER_FACTOR_TRUTH_TABLE_V1_ROWS.pt';OUTPUTS=BASE/'MATCHER_PAIR_TRANSPORT_V1_ROWS.pt'
OUT=BASE/'MATCHER_PAIR_POINTWISE_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists()
    assert hashlib.sha256(SCORES.read_bytes()).hexdigest()=='d0dd9dd724a054e9a7915b0331d8b5df354004f1ad159b5dfe408edb884a62bf'
    assert hashlib.sha256(OUTPUTS.read_bytes()).hexdigest()=='45e9ee20372a4d9cf350964b37d71b6c94cd317270c28d38dbb3cdec9f7a8304'
    scores=torch.load(SCORES,map_location='cpu',weights_only=True);outputs=torch.load(OUTPUTS,map_location='cpu',weights_only=True)
    groups={};cases=[]
    for pop,block in scores.items():
        cells=block['factors']['joint'].reshape(-1,4,4);out=outputs[pop];groups[pop]={}
        for orientation in ('B2_later','B3_later'):
            indices=[i for i in range(len(cells)) if block['metadata'][i*4]['orientation']==orientation]
            old=cells[indices,0];new=cells[indices,3];flip=old[:,3]*new[:,3]<0;kl=out['token_kl']['11'][torch.tensor(indices)*4+3,-1]
            categories={}
            for name,sel in (('value_value_sign_flip',flip),('value_value_same_sign',~flip)):
                categories[name]={'pairs':int(sel.sum()),'hop3_mean_query_kl':float(kl[sel].mean()) if sel.any() else None,
                    'contribution_to_all_pair_hop3_mean_kl':float(kl[sel].sum())/len(indices)}
            groups[pop][orientation]={'pairs':len(indices),'relative_joint_score_change':float((new-old).norm()/old.norm()),
                'joint_score_cosine':float(torch.nn.functional.cosine_similarity(old.flatten(),new.flatten(),dim=0)),
                'value_value_sign_flips':int(flip.sum()),'categories':categories}
            for j,i in enumerate(indices):
                meta=block['metadata'][i*4];other=out['metadata'][i*4+3]
                assert all(meta[k]==other[k] for k in ('world','order','orientation')) and other['hop']==3
                cases.append({'population':pop,**meta,'base_joint_cells':old[j].tolist(),'donor_joint_cells':new[j].tolist(),
                    'relative_joint_score_change':float((new[j]-old[j]).norm()/old[j].norm().clamp_min(1e-8)),
                    'value_value_sign_flip':bool(flip[j]),'hop3_joint_query_kl':float(kl[j])})
    result={'scope':'opened pointwise sign and amplitude diagnostic; no fitted sign/gain adapter or corrected candidate',
        'groups':groups,'cases':cases,'pairs':len(cases),'value_value_sign_flips':sum(c['value_value_sign_flip'] for c in cases),
        'native_coefficients_removed':0,'limits':'Sign correlation with damage does not establish a causal sign-only explanation; all changes and native values remain coupled in the model'}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
