"""Partial-sharing topology, CPU opened-data screen.

Primary modes1+2 width256; controls other pairings and width320. Pred_a dense executable replay<1e-8;
pred_b each native component error<=.15 and <=1.10 separate baseline;
pred_c total stored floats<.95 separate. Preserve256products per component.
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

records=[];exports={}
for members in [(0,1),(0,2),(1,2)]:
 for width in [256,320]:
  V=s.eigenbasis(sum(grams[j] for j in members),width);bank=s.inv@V;activations=s.z@bank
  programs={};errors=list(base_errors);replays=[]
  for idx in members:
   key=str(idx);pair=s.pairs[idx];Gs=cores[idx];old=baselines[key]
   inner=V.T@s.root@old['shared_reader'];reader=bank@inner
   program={k:v for k,v in old.items() if k!='shared_reader'};program['inner_reader']=inner
   for k,Q,G in zip(['a','b'],pair['Qs'],Gs):
    Qnew=reader@G@reader.T;linear=2*Q@s.mu
    constant=s.mu@Q@s.mu+torch.trace(s.oldcov@Q)
    program[k+'_linear']=linear-2*Qnew@s.mu
    program[k+'_bias']=constant-s.mu@linear+s.mu@Qnew@s.mu-torch.trace(s.oldcov@Qnew)
   values=products(activations@inner,program['product_indices'])@program['product_weights']
   reads=[values[:,j]+s.z@program[k+'_linear']+program[k+'_bias'] for j,k in enumerate(['a','b'])]
   phi=scalar(pair,*reads);errors[idx]=err(phi,pair['truth'])
   direct_reads=[]
   for k,Q,G in zip(['a','b'],pair['Qs'],Gs):
    Qnew=reader@G@reader.T
    direct_reads.append(s.mu@Q@s.mu+torch.trace(s.oldcov@Q)+s.delta@(2*Q@s.mu)+((s.delta@Qnew)*s.delta).sum(1)-torch.trace(s.oldcov@Qnew))
   direct=scalar(pair,*direct_reads);replays.append(float((phi-direct).norm()/direct.norm()))
   programs[key]=program
  private={str(idx):baselines[str(idx)] for idx in range(3) if idx not in members}
  floats=bank.numel()+sum(v.numel() for p in list(programs.values())+list(private.values()) for v in p.values() if v.is_floating_point())
  record=dict(shared_modes=[j+1 for j in members],width=width,per_mode_errors=errors,ratios_to_baseline=[a/b for a,b in zip(errors,base_errors)],stored_floats=floats,storage_ratio=floats/base_floats,source_products=768,dense_replay=replays)
  record['fidelity_pass']=max(errors)<=.15 and max(record['ratios_to_baseline'])<=1.1
  records.append(record);exports[f'{members[0]}{members[1]}_{width}']=dict(shared_reader=bank,programs=programs,private_programs=private)
  print(record,flush=True)
primary=records[0]
out=dict(scope='Opened seven cached prefixes. Partial sharing topology; primary modes1+2 width256 fixed before evaluation; controls not confirmations.',baseline_errors=base_errors,baseline_stored_floats=base_floats,records=records,predictions=dict(pred_a_replay=max(max(r['dense_replay']) for r in records)<1e-8,pred_b_fidelity=primary['fidelity_pass'],pred_c_storage=primary['storage_ratio']<.95),seconds=time.perf_counter()-start)
(P/'MULTIMODE_PARTIAL_SHARING_V1.json').write_text(json.dumps(out,indent=2)+'\n')
torch.save(exports,P/'MULTIMODE_PARTIAL_SHARING_V1.pt')
print(out['predictions'],flush=True)
