"""Same-target output SVD baselines for shared quadratic factor hypotheses."""
import json
from pathlib import Path
import torch
from core import metric,inner,coefficients_from_dense
P=Path(__file__).resolve().parent;torch.set_num_threads(1);M=metric(5,4);chol=torch.linalg.cholesky(M)
cases=torch.load(P.parent.parent/'bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1.pt',weights_only=True);rows=[]
for i,c in enumerate(cases):
 t=coefficients_from_dense(c['hessian_not_applicable_quartic'][0].double());t/=inner(t,t,M).sqrt();s=torch.linalg.svdvals(t@chol)
 rows.append(dict(case_index=i,singular_values=s.tolist(),rank1_gaussian_error=float(s[1:].norm()/s.norm()),rank2_gaussian_error=float(s[2:].norm()/s.norm()),rank1_parameter_values=74,rank2_parameter_values=148,scope='Output sharing only: dense canonical scalar quartic plus output writer;74values, no shared quadratic factor assumption.'))
(P/'COMMON_FACTOR_OUTPUT_BASELINES_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print([(r['case_index'],round(r['rank1_gaussian_error'],4)) for r in rows])
