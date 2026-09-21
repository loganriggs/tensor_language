"""Report coefficient error in native and calibration-shaped input coordinates."""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);true=torch.stack([q for p in d['pairs'] for q in p['Qs']]);root=torch.linalg.inv(d['inverse_root']);rows=[]
for family in ['EMPIRICAL_SOURCE_DIRECTIONS','COMPONENT_PAIR_READOUT']:
 meta=json.loads((P/(family+'_V1.json')).read_text());programs=torch.load(P/(family+'_PROGRAMS_V1.pt'),weights_only=True)
 for key in meta['winners'].values():
  p=programs[key];raw=torch.einsum('ir,ro,jr->oij',p['left_reader'],p['product_weights'],p['right_reader']);hat=(raw+raw.transpose(-1,-2))/2;hat[5]+=(p['square_reader']*p['square_weights'])@p['square_reader'].T
  E=hat-true;white=torch.einsum('ai,oij,jb->oab',root,E,root)/d['scales'][:,None,None]
  pair=(E.square().sum((-1,-2)).reshape(3,2).sum(1)/true.square().sum((-1,-2)).reshape(3,2).sum(1)).sqrt()
  rows.append(dict(family=family,key=key,native_isotropic_pair_errors=pair.tolist(),native_isotropic_equal_pair_error=float(pair.square().mean().sqrt()),calibration_shaped_error=float(white.norm()/d['teacher'].norm())))
(P/'NATIVE_COEFFICIENT_GEOMETRY_V1.json').write_text(json.dumps(dict(rows=rows,scope='Native means untransformed residual-input coordinates. Calibration-shaped is the historical local coefficient objective; original referred to its unchanged target/output metric, not isotropic native inputs. No retuning or behavioral adoption.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
