"""Common input feature bank across three native modes; opened CPU screen.
Project originalsourceforms, preserve exactold centeredaffine branches.
Compare one bank with three separately chosen banks, each256wide.
"""
from pathlib import Path
import torch,json,time
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True);c=data['common'];mu=c['mu']
geom=torch.load(P/'EXPANDED_SOURCE_METRIC_GEOMETRY_V1.pt',weights_only=True);root=geom['root'];inv=geom['inverse_root'];oldroot=torch.linalg.inv(c['source_inverse_root']);oldcov=oldroot@oldroot
cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);oldrows=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);mode=torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
z=cal['z'].flatten(0,1).double();delta=z-mu;scale=c['scale'];h=((oldrows['n']+.5*oldrows['m']).flatten(0,1).double())*scale[:,None]
indices=torch.tensor([j for i in [24,25,26,27,29,30,31] for j in range(i*64,(i+1)*64)])
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();D=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double()
pairs=[];grams=[]
for index in range(3):
 a=mode['A'][:,index];b=mode['B'][:,index];Qs=[];Ms=[]
 for reader in [a,b]:
  raw=L.T@((lam*(D.T@reader))[:,None]*R);Q=(raw+raw.T)/2;Qs.append(Q);Ms.append(root@Q@root)
 gram=sum(M@M for M in Ms);gram=(gram+gram.T)/2;grams.append(gram)
 pair=dict(a=a,b=b,alpha=mode['mean_n']@a,beta=mode['mean_m']@b,Qs=Qs,Ms=Ms)
 qa,qb=[((z@Q)*z).sum(1) for Q in Qs];pair['truth']=((h@a-.5*qa)/scale-pair['alpha'])*(qb/scale-pair['beta'])
 pairs.append(pair)
def eigenbasis(gram,width):
 ev,V=torch.linalg.eigh(gram);return V[:,ev.argsort(descending=True)[:width]]
def evaluate(pair,V):
 shared=inv@V;coords=delta@shared;smallcov=shared.T@oldcov@shared;reads=[]
 for Q,M in zip(pair['Qs'],pair['Ms']):
  core=V.T@M@V
  reads.append(mu@Q@mu+torch.trace(oldcov@Q)+delta@(2*Q@mu)+((coords@core)*coords).sum(1)-torch.trace(smallcov@core))
 qa,qb=reads;phi=((h@pair['a']-.5*qa)/scale-pair['alpha'])*(qb/scale-pair['beta']);truth=pair['truth'][indices]
 return float((phi[indices]-truth).norm()/(truth-truth.mean()).norm()),phi
separate=[evaluate(pair,eigenbasis(gram,256)) for pair,gram in zip(pairs,grams)]
assert abs(separate[2][0]-.11941807480512867)<1e-7
joint=sum(G/G.trace() for G in grams);records=[]
for width in [256,320]:
 V=eigenbasis(joint,width);results=[evaluate(pair,V) for pair in pairs]
 truth=sum(p['truth'] for p in pairs)[indices];pred=sum(r[1] for r in results)[indices]
 record=dict(shared_width=width,per_mode_errors=[r[0] for r in results],ratios_to_separate=[r[0]/s[0] for r,s in zip(results,separate)],aggregate_error=float((pred-truth).norm()/(truth-truth.mean()).norm()),
  projected_program_price=dict(shared_input_coefficients=1152*width,three_inner_transforms_coefficients=3*width*width,source_products_if_regular_pair_compilers_succeed=3*width),per_mode_absolute_pass=all(r[0]<=.15 for r in results))
 records.append(record);print(record,flush=True)
primary=records[0];out=dict(predictions=dict(pred_a_baseline_replay=abs(separate[2][0]-.11941807480512867)<1e-7,pred_b_common256=primary['per_mode_absolute_pass'] and max(primary['ratios_to_separate'])<=1.1),separate_per_mode_errors=[r[0] for r in separate],records=records,seconds=time.perf_counter()-start,scope='Originalnative modes1/2/3, commoninputprojection candidates with pair-normalizedGrams; no corecompiler or nativepromotion. Per-modeopenedscalarerrors preventleadingmodeaggregate concealment. Prices are prospective corecompiler costs, not executable savings.')
(P/'MULTIMODE_SHARED_BANK_SCREEN_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(out['predictions'])
