"""Registered root-product sweep with exact centered function metric."""
import itertools,json,time
from pathlib import Path
import torch
from root_product_fit import fit,coefficients
from arithmetic_dag import DAG
P=Path(__file__).resolve().parent

def evaluate(s,x):
 q=((x@s['A'].T)*(x@s['B'].T))@s['bank_writer'].T
 return (((q@s['root_left'].T)*(q@s['root_right'].T))@s['root_writer'].T)@s['W'].T+s['constant']

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(s['A'].shape[1])];left=[d.linear(zip(inputs,row.tolist())) for row in s['A']];right=[d.linear(zip(inputs,row.tolist())) for row in s['B']];products=[d.product(a,b) for a,b in zip(left,right)];bank=[d.linear(zip(products,row.tolist())) for row in s['bank_writer']];rl=[d.linear(zip(bank,row.tolist())) for row in s['root_left']];rr=[d.linear(zip(bank,row.tolist())) for row in s['root_right']];rp=[d.product(a,b) for a,b in zip(rl,rr)];shared=[d.linear(zip(rp,row.tolist())) for row in s['root_writer']];one=d.constant();out=[d.linear(list(zip(shared,row.tolist()))+[(one,float(c))]) for row,c in zip(s['W'],s['constant'])];return d,out

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();metric=torch.load(P/'ROOT_ARCHIVE_METRIC_V1.pt',weights_only=True);source=torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True);s={k:v.double() for k,v in source['programs'][6].items()};Z=metric['weighted_root_writer'];G=metric['covariance'];best={};records=[];allfits={}
 for width,optimizer,lr,seed in itertools.product([4,6,8],['adam','muon'],[.005,.03],[0,1]):
  row,(a,b,c)=fit(Z,G,width,optimizer,lr,seed);row.update(width=width,optimizer=optimizer,lr=lr,seed=seed,retained_centered_energy=1-row['relative_error']**2);H=coefficients(a,b);row['component_energy_ratio']=float(c.square().sum()/(((c@H@G)*(c@H)).sum()));records.append(row);allfits[(width,optimizer,lr,seed)]=dict(a=a,b=b,c=c)
  if width not in best or row['penalized_objective']<best[width][0]['penalized_objective']:best[width]=(row,a,b,c)
  print(row,flush=True)
 panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];fresh=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];exports={};winners=[]
 for width,(row,a,b,c) in best.items():
  writer=torch.linalg.solve(metric['output_factor'],c);constant=s['constant']+s['W']@(s['Z']-writer@coefficients(a,b))@metric['mean'];model={k:v for k,v in s.items() if k not in ['Z','constant']};model.update(root_left=a,root_right=b,root_writer=writer,constant=constant);archive={k:v.float() for k,v in model.items()};loaded={k:v.double() for k,v in archive.items()};d,out=build(loaded);x=panels[0]['rows'][:16].double();replay=float((d.evaluate(out,x)-evaluate(loaded,x)).norm()/evaluate(loaded,x).norm());cost=d.cost(out);assert cost['stored_coefficients']==sum(v.numel() for v in archive.values()) and cost['products']==6+width and replay<1e-10
  errors=[float((evaluate(loaded,p['rows'].double())-y).norm()/y.norm()) for p,y in zip(panels,targets)];fresherrors=[float((evaluate(loaded,p['rows'].double())-p['targets'].double()).norm()/p['targets'].double().norm()) for p in fresh];winners.append(dict(row,diagnostic_errors=errors,fresh_errors=fresherrors,cost=cost,graph_replay=replay));exports[width]=archive;print('WINNER',width,errors,fresherrors,cost,flush=True)
 four=next(w for w in winners if w['width']==4);pred=dict(pred_a_energy=four['retained_centered_energy']>=.99,pred_b_composed=four['fresh_errors'][0]<=.19152331,pred_c_export=four['cost']['stored_coefficients']==24280 and four['cost']['products']==10 and four['graph_replay']<1e-10);torch.save(dict(programs=exports,allfits=allfits,teacher_scale=source['teacher_scale']),P/'ROOT_PRODUCT_REFACTOR_V1.pt');(P/'ROOT_PRODUCT_REFACTOR_V1.json').write_text(json.dumps(dict(records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start,scope='Exact covariance-weighted root fitting; fixed archived6product bank and final output frame. Mean corrected analytically. Text panels now reused diagnostics, no empirical selection or semantic identity claim.'),indent=2)+'\n');print(pred)
if __name__=='__main__':main()
