"""Parseval audit of the opened controller grid, not a held-out fitted predictor."""
import hashlib,json
from pathlib import Path
import numpy as np
import controller_selector_v1 as S


def main():
    root=Path(__file__).resolve().parent;path=root/'SUBJECT_OBJECT_CONTROLLER_V1_RESULT.json'
    source=json.loads(path.read_text());assert source['predictions']['pred_a_instrument']
    reports={}
    h=S.design()
    for split,report in source['reports'].items():
        coef=np.array(report['margin_coefficients']);full=coef@h.T
        additive=coef[:,:4]@h[:,:4].T
        observed=np.mean((full-additive)**2,axis=1)
        parseval=np.sum(coef[:,4:]**2,axis=1)
        assert np.max(abs(observed-parseval))<1e-12
        nonconstant=np.sum(coef[:,1:]**2,axis=1)
        ratios=np.sqrt(parseval/nonconstant)
        object_index=S.TERMS.index('o')
        rest=np.sum(abs(coef),axis=1)-abs(coef[:,object_index])
        reports[split]={'world_ids':report['world_ids'],
            'mean_coefficients':dict(zip(S.TERMS,coef.mean(0).tolist())),
            'interaction_to_nonconstant_rms_ratios':ratios.tolist(),
            'object_coefficient_dominance_margins':(coef[:,object_index]-rest).tolist(),
            'dominance_certificate_worlds':int(np.sum(coef[:,object_index]>rest)),
            'additive_grid_sign_matches_object':int(np.sum(np.sign(additive)==np.array([v[2] for v in S.CORNERS])[None,:])),
            'grid_values':int(additive.size),'parseval_max_error':float(np.max(abs(observed-parseval)))}
    result={'schema':1,'reports':reports,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_forwards':0,
            'scope':'Per-world projection of already observed eight-corner tables. No unseen-input or independent-execution claim.'}
    with (root/'CONTROLLER_ADDITIVITY_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
