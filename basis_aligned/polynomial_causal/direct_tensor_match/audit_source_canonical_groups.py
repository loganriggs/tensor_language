"""Canonical output-shared quadratic features, retaining all four modes."""
from pathlib import Path
import json,torch
from scipy.optimize import linear_sum_assignment
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);metadata=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());warm=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True);random=torch.load(P/'SOURCE_RANDOM_START_PROGRAMS_V1.pt',weights_only=True)
def tensor(p):
 s=p['shared_mixed'];return materialize_mixed(root@s['left_reader'],root@s['right_reader'],s['product_weights'].T/d['scales'][:4,None])
def canonical(T):
 F=T.flatten(1);eig,U=torch.linalg.eigh(F@F.T);eig=eig.flip(0);U=U.flip(1);G=U.T@F
 replay=float((U@G-F).norm()/F.norm());assert replay<1e-12
 return U,G,eig,replay
ref=canonical(tensor(warm[metadata['winner']]));teacher=canonical(d['teacher'][:4]);records=[]
for name,T in [('original',d['teacher'][:4])]+[('warm_'+k,tensor(v)) for k,v in warm.items()]+[('random_'+k,tensor(v)) for k,v in random.items() if k.startswith('0_')]:
 U,G,eig,replay=canonical(T);V,H,_,_=ref;sim=V.T@U;i,j=linear_sum_assignment(-sim.abs().numpy());core=H@G.T/(H.norm(dim=1)[:,None]*G.norm(dim=1)[None,:]);r=dict(candidate=name,output_cosines=sim[i,j].abs().tolist(),quadratic_feature_cosines=core[i,j].abs().tolist(),eigenvalues=eig.tolist(),relative_successive_gaps=((eig[:-1]-eig[1:])/eig[:-1]).tolist(),replay=replay);records.append(r);print(r,flush=True)
fitrecords=[r for r in records if r['candidate']!='original']
out=dict(records=records,reference=metadata['winner'],predictions=dict(pred_a_replay=max(r['replay'] for r in records)<1e-12,pred_b_all_groups_stable=all(min(r['output_cosines']+r['quadratic_feature_cosines'])>=.99 for r in fitrecords)),scope='Canonicalization is exact for each fitted4outputquadratic tensor and changes no product cost. Groups sum existing products; covariance metric and output spectral gaps determine basis. Same teacher/data; random lambda0 only for like-objective identity. Original comparison is fidelity, not restart variability. No semantic identification.')
(P/'SOURCE_CANONICAL_GROUPS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
