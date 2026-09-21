"""Independent-start dictionary identity against the frozen pre-refit parent."""
from pathlib import Path
import torch,json
from scipy.optimize import linear_sum_assignment
from audit_profiled_subspaces import gram,compare
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);oldmeta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());old=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[oldmeta['winner']];meta=json.loads((P/'SOURCE_RANDOM_START_FIT_V1.json').read_text());ps=torch.load(P/'SOURCE_RANDOM_START_PROGRAMS_V1.pt',weights_only=True)
def factors(p):
 s=p['shared_mixed'];return root@s['left_reader'],root@s['right_reader'],s['product_weights'].T/d['scales'][:4,None]
X=factors(old);ref=materialize_mixed(*X);records=[]
for lam,key in meta['winners'].items():
 Y=factors(ps[key]);K=gram(X,Y);e=gram(X,X).diag();f=gram(Y,Y).diag();cos=K/(e[:,None]*f[None,:]).sqrt();i,j=linear_sum_assignment(-cos.abs().numpy());c=cos[i,j]
 record=dict(lam=lam,key=key,median_matched_atom_abs_cosine=float(c.abs().median()),matched_atom_energy_fraction_above_090=float((e[i]*(c.abs()>=.9)).sum()/e.sum()),function_disagreement=float((materialize_mixed(*Y)-ref).norm()/ref.norm()),**compare(X,Y));records.append(record);print({k:v for k,v in record.items() if k!='principal_cosines'})
out=dict(records=records,predictions=dict(pred_a_atom_identity=all(r['matched_atom_energy_fraction_above_090']>=.8 for r in records)),scope='Selected winners from truly random shared-direction starts versus inherited parent; lambda1 changes objective. Same trained teacher and covariance, not independent data discovery. No global uniqueness theorem or causal identification.')
(P/'SOURCE_RANDOM_IDENTITY_V1.json').write_text(json.dumps(out,indent=2)+'\n')
