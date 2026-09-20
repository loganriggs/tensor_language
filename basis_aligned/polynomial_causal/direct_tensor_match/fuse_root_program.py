"""Compose adjacent linear maps, accepting only literal graph cost reductions."""
import json
from pathlib import Path
import torch
from arithmetic_dag import DAG
from root_product_refactor import evaluate as reference,build as reference_build
P=Path(__file__).resolve().parent

def evaluate(s,x):
 p=(x@s['A'].T)*(x@s['B'].T);q=p@s['bank_writer'].T if 'bank_writer' in s else p
 return ((q@s['root_left'].T)*(q@s['root_right'].T))@s['output_writer'].T+s['constant']

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(s['A'].shape[1])];left=[d.linear(zip(inputs,r.tolist())) for r in s['A']];right=[d.linear(zip(inputs,r.tolist())) for r in s['B']];p=[d.product(a,b) for a,b in zip(left,right)];q=[d.linear(zip(p,r.tolist())) for r in s['bank_writer']] if 'bank_writer' in s else p;rl=[d.linear(zip(q,r.tolist())) for r in s['root_left']];rr=[d.linear(zip(q,r.tolist())) for r in s['root_right']];roots=[d.product(a,b) for a,b in zip(rl,rr)];one=d.constant();out=[d.linear(list(zip(roots,r.tolist()))+[(one,float(c))]) for r,c in zip(s['output_writer'],s['constant'])];return d,out

def main():
 torch.set_num_threads(1);source=torch.load(P/'ROOT_PRODUCT_REFACTOR_V1.pt',weights_only=True);panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];x=panels[0]['rows'][:32].double();rows=[];exports={}
 for width,program in source['programs'].items():
  s={k:v.double() for k,v in program.items()};old_d,old_out=reference_build(s);oldcost=old_d.cost(old_out);base={k:v for k,v in s.items() if k not in ['W','root_writer']};base['output_writer']=s['W']@s['root_writer'];candidates=[]
  for fold_bank in [False,True]:
   model={k:v.clone() for k,v in base.items()}
   if fold_bank:model['root_left']=model['root_left']@model['bank_writer'];model['root_right']=model['root_right']@model['bank_writer'];del model['bank_writer']
   d,out=build(model);cost=d.cost(out);exact=float((evaluate(model,x)-reference(s,x)).norm()/reference(s,x).norm());assert exact<1e-12
   candidates.append((cost,fold_bank,model,exact))
  # Storage, products, additions and edges must not regress from output-only fusion.
  baseline=candidates[0][0];admissible=[c for c in candidates if all(c[0][k]<=baseline[k] for k in ['stored_coefficients','products','additions','edges'])];cost,fold_bank,model,exact=min(admissible,key=lambda c:(c[0]['stored_coefficients'],c[0]['additions']));assert all(cost[k]<=oldcost[k] for k in ['stored_coefficients','products','additions','edges'])
  archive={k:v.float() for k,v in model.items()};loaded={k:v.double() for k,v in archive.items()};d,out=build(loaded);cost=d.cost(out);replay=float((d.evaluate(out,x)-evaluate(loaded,x)).norm()/evaluate(loaded,x).norm());rounding=float((evaluate(loaded,x)-reference(s,x)).norm()/reference(s,x).norm());assert replay<1e-10 and rounding<1e-5 and cost['stored_coefficients']==sum(v.numel() for v in archive.values());errors=[float((evaluate(loaded,p['rows'].double())-p['targets'].double()).norm()/p['targets'].double().norm()) for p in panels];rows.append(dict(root_width=width,bank_fused=fold_bank,original_cost=oldcost,cost=cost,exact_arithmetic_fusion_error=exact,archive_rounding_error=rounding,graph_replay=replay,fresh_errors=errors,rejected_or_alternative_costs=[dict(bank_fused=f,cost=c) for c,f,_,_ in candidates]));exports[width]=archive
 torch.save(dict(programs=exports,teacher_scale=source['teacher_scale']),P/'FUSED_ROOT_PROGRAM_V1.pt');(P/'FUSED_ROOT_PROGRAM_V1.json').write_text(json.dumps(dict(records=rows,scope='Algebraic linear composition chosen by literal reachable graph costs; FP64 equivalence and FP32 rounding audited. Output fusion accepted for all widths, bank fusion only when no counted cost regresses. Same reused text diagnostics.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
