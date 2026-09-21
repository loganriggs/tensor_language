"""Redteam isotropic reader-rank conclusions under fixed activation-shaped metric.
M=L L^T is the saved second-moment floor01 geometry. Perturbations are L z;
local functional derivative Gram is L^T G L. This is not a Gaussian closure
of the nonlinear target and does not assume natural interventions have this law.
"""
import json,time,math
from pathlib import Path
import torch
from quartic_reader_rank_bound import analyze
from quartic_reader_span import reader_basis
P=Path(__file__).resolve().parent


def main():
    start=time.monotonic();torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(5100)
    j=torch.randn(23,11,dtype=torch.float64);L=torch.randn(11,11,dtype=torch.float64);g=j.T@j;direct=(j@L).T@(j@L);indirect=L.T@g@L;control=float((direct-indirect).norm()/direct.norm());assert control<1e-12
    saved=torch.load(P/'QUARTIC_READER_GRAMS_V1.pt',weights_only=True);grams=saved['grams'];transform=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['second_floor01'].double();weighted=[transform.T@g@transform for g in grams]
    result,bases=analyze(weighted,[128,256,512,1024]);pooled,_=analyze([sum(weighted)],[128,256,512,1024]);rawpooled,_=analyze([sum(grams)],[128,256,512,1024]);student=torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True);q=reader_basis(student['U'].double()@transform,student['V'].double()@transform)
    learned=[float(((g.trace()-(q*(g@q)).sum())/g.trace()).clamp_min(0).sqrt()) for g in weighted]
    result.update(pooled_weighted=pooled['panels'][0],pooled_isotropic=rawpooled['panels'][0],learned_reader_weighted_errors=learned,control_error=control,prediction_weighted256_below10percent_both=all(r['floors']['256']<=.1 for r in result['panels']),seconds=time.monotonic()-start,scope='Fixed saved second-moment geometry applied to exact local Jacobian Grams. Two opened16-anchor panels. Weightedrank necessary, not sufficient; no natural-value, OOD or selectiveintervention claim.')
    result['necessary_products_per_feature_at32features']={name:{t:math.ceil(r/64) for t,r in row['necessary_reader_counts'].items()} for name,row in [('isotropic_pooled',result['pooled_isotropic']),('weighted_pooled',result['pooled_weighted'])]}
    (P/'QUARTIC_WEIGHTED_READER_RANK_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
