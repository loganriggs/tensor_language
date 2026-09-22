"""Algebraic stress tests: conditional Gaussian approximations need not preserve
the even parity or homogeneity of the original selected quartic target."""
import json
from pathlib import Path
import torch
from conditional_quartic_cp import evaluate as full
from lean_conditional_cp import evaluate as lean
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 a=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);ix=torch.arange(256)*64+31;x=a['rows'][ix].double();y=a['target'][ix].double();rows=[]
 for seed in [1001,1002]:
  for rank in [64,128,256]:
   for kind,prefix,fn in [('full','CONDITIONAL_CP',full),('lean','LEAN_CONDITIONAL_CP',lean)]:
    p=torch.load(P/f'{prefix}_SEED{seed}_RANK{rank}_V1.pt',weights_only=True);p={k:[v.double() for v in value] if isinstance(value,list) else value.double() if isinstance(value,torch.Tensor) else value for k,value in p.items()}
    pred=fn(p,x);negative=fn(p,-x);even=(pred+negative)/2;odd=(pred-negative)/2
    scales={}
    for scale in [.5,2.]:
     value=fn(p,scale*x);scales[str(scale)]=float((value-scale**4*pred).norm()/(scale**4*pred).norm())
    rows.append(dict(seed=seed,input_rank=rank,kind=kind,positive_value_error=float((pred-y).norm()/y.norm()),negative_input_value_error=float((negative-y).norm()/y.norm()),odd_component_relative_to_native=float(odd.norm()/y.norm()),even_projection_value_error=float((even-y).norm()/y.norm()),homogeneity_defects=scales))
 result=dict(rows=rows,scope='256openedstatesposition31. Nativepurequartic F(-x)=F(x), F(ax)=a^4F(x). Negation preservesnormbutisnotshown tobeareachabletextstate; scaling exitsnormalizedinputinterface. Diagnosticofalgebraicgeneralization, not textOODorbehavior. Gaussianconditionalmean neednotpreserveinvariants undernonzero mean; new lowerdegree terms explicitlyallowed/priced. Evenprojectionis oracleconstructionevaluatedbothsigns, notadopted/exported.')
 (P/'CONDITIONAL_SYMMETRY_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:print(r)
if __name__=='__main__':main()
