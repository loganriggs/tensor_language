"""Small cross-reader additions with literal reuse of existing linear forms."""
import json
from pathlib import Path
import torch
from arithmetic_dag import DAG
P=Path(__file__).resolve().parent

def evaluate(s,x):
 left=x@s['A'].T;right=x@s['B'].T;p=left*right;h=(p@s['root_left'].T)*(p@s['root_right'].T)
 y=h@s['quartic_readout'].T+p@s['quadratic_readout'].T+s['constant']
 extra=(x@s['extra_directions'].T)*(torch.cat([left,right],1)@s['extra_partners'].T)
 y=y.clone();y[:,1]+=extra.sum(1)
 return y

def expand(s):
 k=len(s['extra_directions']);base={n:s[n].clone() for n in ['A','B','root_left','root_right','quartic_readout','quadratic_readout','constant']}
 base['A']=torch.cat([s['A'],s['extra_directions']]);base['B']=torch.cat([s['B'],s['extra_partners']@torch.cat([s['A'],s['B']])]);base['root_left']=torch.nn.functional.pad(s['root_left'],(0,k));base['root_right']=torch.nn.functional.pad(s['root_right'],(0,k));extra=torch.zeros(4,k,dtype=s['A'].dtype,device=s['A'].device);extra[1]=1;base['quadratic_readout']=torch.cat([s['quadratic_readout'],extra],1)
 return base

def build(s):
 d=DAG(degree_limit=4);inputs=[d.input(i) for i in range(s['A'].shape[1])];left=[d.linear(zip(inputs,row.tolist())) for row in s['A']];right=[d.linear(zip(inputs,row.tolist())) for row in s['B']];p=[d.product(a,b) for a,b in zip(left,right)];ll=[d.linear(zip(p,row.tolist())) for row in s['root_left']];rr=[d.linear(zip(p,row.tolist())) for row in s['root_right']];h=[d.product(a,b) for a,b in zip(ll,rr)];extra=[d.product(d.linear(zip(inputs,v.tolist())),d.linear(zip(left+right,w.tolist()))) for v,w in zip(s['extra_directions'],s['extra_partners'])];one=d.constant();out=[]
 for g,(a,b,c) in enumerate(zip(s['quartic_readout'],s['quadratic_readout'],s['constant'])):
  terms=list(zip(h,a.tolist()))+list(zip(p,b.tolist()))+[(one,float(c))]
  if g==1:terms+=list(zip(extra,[1.]*len(extra)))
  out.append(d.linear(terms))
 return d,out

def main():
 from extract_scalar_modes import evaluate as old_evaluate
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 base={k:v.double() for k,v in torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True)['program'].items()};H=torch.cat([base['A'],base['B']]);Q=torch.linalg.qr(H.T).Q;directions=torch.load(P/'READER_SPHERE_DIRECTIONS_V1.pt',weights_only=True)['directions'].double();gen=torch.Generator().manual_seed(260957);rand=torch.randn(1152,8,dtype=torch.float64,generator=gen);rand=rand-Q@(Q.T@rand);random=torch.linalg.qr(rand).Q.T
 view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);u=view['output_directions'].double()[:,1];mu=view['output_mean'].double();cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);held=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True);assert cal['token_sha256']==targets['token_sha256'][0];assert abs(float(held['teacher_scale'])/float(targets['teacher_scale'])-1)<1e-6
 panels=[('calibration',cal['rows'].double(),(targets['targets'][0].double()-mu)@u)]+[(f'diagnostic_{p["context"]}',p['rows'].double(),(p['targets'].double()-mu)@u) for p in held['panels']];x,y=panels[0][1:];residual=y-old_evaluate(base,x)[:,1];records=[];exports={};prices=[];checks=[]
 for name,xx,yy in panels:records.append(dict(panel=name,model='baseline',mse=float((old_evaluate(base,xx)[:,1]-yy).square().mean()),relative_error=float((old_evaluate(base,xx)[:,1]-yy).norm()/yy.norm())))
 for family,V in [('gradient',directions),('random',random)]:
  for k in [1,4,8]:
   v=V[:k];cross=((x@v.T)[:,:,None]*(x@H.T)[:,None,:]).flatten(1);mean=cross.mean(0);scale=cross.std(0,unbiased=False);z=(cross-mean)/scale;coef=torch.linalg.solve(z.T@z/len(x)+1e-3*torch.eye(z.shape[1]),z.T@residual/len(x))/scale;model={n:t.clone() for n,t in base.items()};model.update(extra_directions=v,extra_partners=coef.reshape(k,12));model['constant'][1]-=mean@coef;key=f'{family}_{k}';archive={n:t.float() for n,t in model.items()};loaded={n:t.double() for n,t in archive.items()};exports[key]=archive;expanded=expand(loaded);dag,out=build(loaded);prices.append(dict(model=key,cost=dag.cost(out)))
   for name,xx,yy in panels:
    prediction=evaluate(loaded,xx);ref=old_evaluate(expanded,xx);checks.append(float((prediction-ref).norm()/ref.norm()));assert torch.equal(prediction[:,[0,2,3]],old_evaluate(base,xx)[:,[0,2,3]])
    records.append(dict(panel=name,model=key,mse=float((prediction[:,1]-yy).square().mean()),relative_error=float((prediction[:,1]-yy).norm()/yy.norm())))
   checks.append(float((dag.evaluate(out,x[:8])-evaluate(loaded,x[:8])).norm()/evaluate(loaded,x[:8]).norm()))
 def mse(panel,name):return next(r['mse'] for r in records if r['panel']==panel and r['model']==name)
 pred=dict(pred_a_instrument=max(checks)<1e-5,pred_b_gain=all(mse(n,'gradient_4')<=.9*mse(n,'baseline') for n in ['diagnostic_64','diagnostic_256']),pred_c_directions=all(mse(n,'gradient_4')<=.9*mse(n,'random_4') for n in ['diagnostic_64','diagnostic_256']))
 result=dict(records=records,prices=prices,predictions=pred,replay_max=max(checks),primary='gradient_4',scope='Calibration-fitted quadratic cross-reader additions. Shared old scalar readers; no native intervention claim. Gradient proposals data-informed via calibration states and teacher weight derivatives. Reused diagnostic panels.')
 torch.save(dict(programs=exports,teacher_scale=targets['teacher_scale'],primary='gradient_4'),P/'GRADIENT_READER_ADDITIONS_V1.pt');(P/'GRADIENT_READER_ADDITIONS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
