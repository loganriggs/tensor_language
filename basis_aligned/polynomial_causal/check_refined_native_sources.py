import json
from pathlib import Path
import torch
from refined_native_sources import refine,collapse,expand,PARENTS

torch.manual_seed(812);torch.set_num_threads(2)
explicit=torch.randn(3,23,7,dtype=torch.float64);parents=collapse(explicit)+1e-7*torch.randn(3,6,7,dtype=torch.float64)
children,check=refine(parents,explicit)
a=torch.randn(3,6,dtype=torch.float64)
old=torch.einsum('bi,bid->bd',a,parents);new=torch.einsum('bi,bid->bd',expand(a),children)
error=float((old-new).abs().max());assert error<1e-12 and check['collapse_error']<1e-12
# Independent shared-amplitude Jacobian identity.
reader=torch.randn(3,4,7,dtype=torch.float64)
g=torch.einsum('bod,bid->boi',reader,children)
fold=torch.stack([g[:,:,[i for i,p in enumerate(PARENTS) if p==parent]].sum(-1) for parent in range(6)],dim=-1)
expected=torch.einsum('bod,bid->boi',reader,parents)
gradient_error=float((fold-expected).abs().max());assert gradient_error<1e-12
out=dict(collapse=check['collapse_error'],tied_prediction_error=error,gradient_fold_error=gradient_error,scope='Planted refinement identity; native hook and numerical-gauge validation pending.')
(Path(__file__).parent/'REFINED_NATIVE_SOURCES_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
