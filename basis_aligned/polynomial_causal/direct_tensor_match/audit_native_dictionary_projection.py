"""Best native-isotropic coefficient readout in the existing fixed product dictionary."""
from pathlib import Path
import json,torch
from shared_private_metric import gram
from global_mixed_source_graph import export,score
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text());parent=torch.load(P/'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['1']]
m=parent['left_reader'].shape[1];L=torch.cat([parent['left_reader'],parent['square_reader']],1);R=torch.cat([parent['right_reader'],parent['square_reader']],1);Q=torch.stack([q for p in d['pairs'] for q in p['Qs']]);G=gram(L,R,L,R);C=torch.einsum('ir,oij,jr->ro',L,Q,R);W=torch.zeros_like(C);conditions=[];normal=[]
for o in range(6):
 n=len(G) if o==5 else m;g=G[:n,:n];e=torch.linalg.eigvalsh(g);assert e[0]>1e-12*e[-1]
 W[:n,o]=torch.linalg.solve(g,C[:n,o]);normal.append(float((g@W[:n,o]-C[:n,o]).norm()/C[:n,o].norm()));conditions.append(float(e[-1]/e[0]))
raw=torch.einsum('ir,ro,jr->oij',L,W,R);hat=(raw+raw.transpose(-1,-2))/2;errors=((hat-Q).square().sum((-1,-2)).reshape(3,2).sum(1)/Q.square().sum((-1,-2)).reshape(3,2).sum(1)).sqrt()
root=torch.linalg.inv(d['inverse_root']);white=torch.einsum('ai,oij,jb->oab',root,hat,root)/d['scales'][:,None,None]
big=export(root@L,root@R,(W/d['scales']).T,d)
out=dict(native_isotropic_pair_errors=errors.tolist(),native_isotropic_equal_pair_error=float(errors.square().mean().sqrt()),calibration_shaped_error=float((white-d['teacher']).norm()/d['teacher'].norm()),normal_equation_relative_residuals=normal,gram_condition_numbers=conditions,**score(big,d),scope='Unregularized exact native-coefficient least-squares projection within the fixed current dictionary. Only output weights change; no new directions or adopted candidate. Independent pair normalization does not change these separable minimizers. Numerical projection optimum, not a lower bound over other dictionaries.')
assert max(normal)<1e-10
(P/'NATIVE_DICTIONARY_PROJECTION_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
