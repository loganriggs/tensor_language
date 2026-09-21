"""Fixed-product cross-component projection sharing, CPU opened-data screen.

Primary320, controls256/384. Pred_a dense executable replay<1e-8;
pred_b each native component error<=.15 and <=1.10 separate baseline;
pred_c total stored floats<.90 separate. Preserve256products per component.
Null: common projections damage individual source functions despite lower cost.
"""
from pathlib import Path
import json,time,torch
import screen_multimode_shared_bank as s
from quadratic_pair_blocks import products
P=Path(__file__).resolve().parent
start=time.perf_counter()
baselines=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True)

def core_matrices(program):
 n=program['shared_reader'].shape[1];matrices=[]
 for out in range(2):
  G=torch.zeros(n,n,dtype=torch.float64)
  for position,(i,j,kind) in enumerate(program['product_indices'].T.tolist()):
   w=program['product_weights'][position,out]
   if kind==0:G[i,i]+=w
   elif kind==1:G[i,i]+=w;G[j,j]-=w
   else:G[i,j]+=w/2;G[j,i]+=w/2
  matrices.append(G)
 return matrices

def scalar(pair,qa,qb):
 return ((s.h@pair['a']-.5*qa)/s.scale-pair['alpha'])*(qb/s.scale-pair['beta'])

def err(value,truth):
 x=truth[s.indices]
 return float((value[s.indices]-x).norm()/(x-x.mean()).norm())

cores=[];grams=[];base_errors=[]
for key,pair in zip(['0','1','2'],s.pairs):
 program=baselines[key];Qs=core_matrices(program);cores.append(Qs)
 reader=program['shared_reader'];metric_reader=s.root@reader
 Ms=[metric_reader@G@metric_reader.T for G in Qs]
 gram=sum(M@M for M in Ms);grams.append(gram/gram.trace())
 values=products(s.z@reader,program['product_indices'])@program['product_weights']
 reads=[values[:,j]+s.z@program[k+'_linear']+program[k+'_bias'] for j,k in enumerate(['a','b'])]
 base_errors.append(err(scalar(pair,*reads),pair['truth']))
base_floats=sum(v.numel() for p in baselines.values() for v in p.values() if v.is_floating_point())
# Independent forms define the common space, not arbitrary compiler eigenvector scales.
joint=sum(grams);records=[];exports={}
for width in [256,320,384]:
 V=s.eigenbasis(joint,width);bank=s.inv@V;shared_activations=s.z@bank;programs={};errors=[];replays=[]
 for idx,(key,pair,Gs) in enumerate(zip(['0','1','2'],s.pairs,cores)):
  old=baselines[key];inner=V.T@s.root@old['shared_reader'];reader=bank@inner
  program={k:v for k,v in old.items() if k!='shared_reader'}
  program['inner_reader']=inner
  for k,Q,G in zip(['a','b'],pair['Qs'],Gs):
   Qnew=reader@G@reader.T
   linear=2*Q@s.mu
   constant=s.mu@Q@s.mu+torch.trace(s.oldcov@Q)
   program[k+'_linear']=linear-2*Qnew@s.mu
   program[k+'_bias']=constant-s.mu@linear+s.mu@Qnew@s.mu-torch.trace(s.oldcov@Qnew)
  values=products(shared_activations@inner,program['product_indices'])@program['product_weights']
  reads=[values[:,j]+s.z@program[k+'_linear']+program[k+'_bias'] for j,k in enumerate(['a','b'])]
  phi=scalar(pair,*reads);errors.append(err(phi,pair['truth']))
  dense=[]
  for k,Q,G in zip(['a','b'],pair['Qs'],Gs):
   Qnew=reader@G@reader.T
   dense.append(s.mu@Q@s.mu+torch.trace(s.oldcov@Q)+s.delta@(2*Q@s.mu)+((s.delta@Qnew)*s.delta).sum(1)-torch.trace(s.oldcov@Qnew))
  direct=scalar(pair,*dense)
  replays.append(float((phi-direct).norm()/direct.norm()))
  programs[key]=program
 stored=bank.numel()+sum(v.numel() for p in programs.values() for v in p.values() if v.is_floating_point())
 records.append(dict(width=width,per_mode_errors=errors,ratios_to_baseline=[a/b for a,b in zip(errors,base_errors)],source_products=sum(len(p['product_weights']) for p in programs.values()),stored_floats=stored,storage_ratio=stored/base_floats,stored_indices=sum(p['product_indices'].numel() for p in programs.values()),dense_replay=replays))
 exports[str(width)]=dict(shared_reader=bank,programs=programs)
 print(records[-1],flush=True)
primary=records[1]
out=dict(scope='Opened seven distinct cached prefixes; no fresh native confirmation or semantic claim.',primary_width=320,baseline_errors=base_errors,baseline_stored_floats=base_floats,records=records,predictions=dict(pred_a_replay=max(max(r['dense_replay']) for r in records)<1e-8,pred_b_fidelity=max(primary['per_mode_errors'])<=.15 and max(primary['ratios_to_baseline'])<=1.10,pred_c_storage=primary['storage_ratio']<.90),seconds=time.perf_counter()-start)
(P/'MULTIMODE_PROJECTION_SHARING_V1.json').write_text(json.dumps(out,indent=2)+'\n')
torch.save(exports,P/'MULTIMODE_PROJECTION_SHARING_V1.pt')
print(out['predictions'],flush=True)
