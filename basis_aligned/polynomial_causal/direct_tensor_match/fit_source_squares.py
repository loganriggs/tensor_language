"""Fixed spectral source corrections, exact finite allocation, and executable replay."""
import copy,itertools,json,time
from pathlib import Path
import torch
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
plan=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text());base=plan['baseline'];d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);audit=Assessment(d);parents=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)
ids=d['indices'];z=d['z'][ids];h=d['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();budget=plan['maximum_added'];records=[]
def evaluate(Q,true,j):
 delta=true-Q;lin=2*delta@d['mu'];bias=torch.einsum('ij,oij->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu'])
 def native(forms,linear,constant):
  reads=torch.einsum('ni,oij,nj->no',z,forms,z)+z@linear.T+constant
  grad=2*torch.einsum('oij,nj->noi',forms,z)+linear[None]
  pair=d['pairs'][j];a=(h@pair['a']-.5*reads[:,0])/scale-pair['alpha'];b=reads[:,1]/scale-pair['beta']
  return a*b,-.5*(b/scale)[:,None]*grad[:,0]+(a/scale)[:,None]*grad[:,1]
 value,jac=native(Q,lin,bias);_,J=native(true,torch.zeros_like(lin),torch.zeros_like(bias));truth=d['pairs'][j]['truth'][ids]
 coeff=[]
 for A in (audit.I,audit.S):
  coeff.append(float((A@(Q-true)@A).square().sum()/(A@true@A).square().sum()))
 return dict(pair_squared_errors=coeff,value_error=float((value-truth).norm()/(truth-truth.mean()).norm()),jacobian_error=float((jac-J).norm()/J.norm()))
for geometry in plan['geometries']:
 parent=parents[geometry+'_inherited'];bundle=expand(parent);A,inv=(audit.S,d['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);profiles=[];banks=[]
 for j in range(3):
  true=audit.Q[2*j:2*j+2];H=decode(bundle[str(j)]);eig,U=torch.linalg.eigh(A@(true[1]-H[1])@A);order=eig.abs().argsort(descending=True)[:budget];values=eig[order];V=inv@U[:,order];banks.append((V,values));profile=[]
  for n in range(budget+1):
   Q=H.clone();Q[1]+=(V[:,:n]*values[:n])@V[:,:n].T;profile.append(evaluate(Q,true,j))
  profiles.append(profile)
 allocations=[v for v in itertools.product(range(budget+1),repeat=3) if sum(v)<=budget]
 def metrics(v):
  selected=[profiles[j][n] for j,n in enumerate(v)];coef=[(sum(x['pair_squared_errors'][k] for x in selected)/3)**.5 for k in (0,1)]
  ratios=[coef[0]/(1.1*base['native_error']),coef[1]/(1.1*base['covariance_error'])]
  ratios.extend(x['value_error']/min(.15,1.1*base['per_mode_errors'][j]) for j,x in enumerate(selected));ratios.extend(x['jacobian_error']/(1.1*base['euclidean_jacobian_errors'][j]) for j,x in enumerate(selected))
  return coef,max(ratios)
 k=1 if geometry=='calibration_shaped' else 0
 choices={'coefficient':min(allocations,key=lambda v:(metrics(v)[0][k],sum(v))),'opened_fidelity':min(allocations,key=lambda v:(metrics(v)[1],sum(v)))}
 for selection,allocation in choices.items():
  graph=copy.deepcopy(parent)
  for j,n in enumerate(allocation):
   if not n:continue
   V,values=banks[j];part=graph['pairs'][str(j)];offset=len(part['shared_indices'])+len(part['private_indices']);indices=torch.arange(offset,offset+n)
   part['private_reader']=torch.cat([part['private_reader'],V[:,:n]],1);part['private_indices']=torch.cat([part['private_indices'],indices]);part['product_indices']=torch.cat([part['product_indices'],torch.stack([indices,indices,torch.zeros_like(indices)])],1);part['product_weights']=torch.cat([part['product_weights'],torch.stack([torch.zeros_like(values[:n]),values[:n]],1)])
  graph=audit.correct(graph);result=audit.assess(graph);expected=metrics(allocation);assert abs(result['native_error']-expected[0][0])<1e-8;assert abs(result['covariance_error']-expected[0][1])<1e-8
  for j,n in enumerate(allocation):
   assert abs(result['per_mode_errors'][j]-profiles[j][n]['value_error'])<1e-8
   assert abs(result['euclidean_jacobian_errors'][j]-profiles[j][n]['jacobian_error'])<1e-8
  assert result['source_total_multiplications']==1047648+1155*sum(allocation)<=plan['source_ceiling'];assert result['stored_floats']==result['physical_storage_floats']
  records.append(dict(geometry=geometry,selection=selection,allocation=list(allocation),worst_fidelity_ratio=expected[1],fidelity_pass=expected[1]<=1,**result));print(json.dumps(records[-1]),flush=True)
 outprofiles=dict(geometry=geometry,profiles=profiles)
 (P/('SOURCE_SQUARE_PROFILES_'+geometry+'_V1.json')).write_text(json.dumps(outprofiles,indent=2)+'\n')
primary=next(r for r in records if r['geometry']=='calibration_shaped' and r['selection']=='coefficient')
out=dict(plan=plan,records=records,predictions=dict(instrument=True,fidelity=primary['fidelity_pass'],cost=True),seconds=time.monotonic()-start)
(P/'SOURCE_SQUARE_V1.json').write_text(json.dumps(out,indent=2)+'\n')
