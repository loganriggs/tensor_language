"""Freeze original six folded quadratic forms and spectral shared-CP start."""
from pathlib import Path
import torch,json,hashlib
import screen_multimode_shared_bank as s
P=Path(__file__).resolve().parent
teachers=[];scales=[];directions=[];weights=[]
for pair in s.pairs:
 scale=sum(M.square().sum() for M in pair['Ms']).sqrt();scales.extend([scale,scale]);teachers.extend([M/scale for M in pair['Ms']])
teacher=torch.stack(teachers)
for output,M in enumerate(teacher):
 ev,V=torch.linalg.eigh(M)
 for j in ev.abs().argsort(descending=True)[:512]:
  directions.append(V[:,j]*ev[j].abs().sqrt())
  w=torch.zeros(6,dtype=torch.float64);w[output]=ev[j].sign();weights.append(w)
values=torch.stack(directions,1);coefficients=torch.stack(weights,1)
order=values.square().sum(0).argsort(descending=True)[:512]
artifact=dict(teacher=teacher,scales=torch.stack(scales),initial_reader=values[:,order],initial_weights=coefficients[:,order],inverse_root=s.inv,mu=s.mu,old_covariance=s.oldcov,z=s.z,h=s.h,scale=s.scale,indices=s.indices,
 pairs=[{k:v for k,v in p.items() if k!='Ms'} for p in s.pairs],residual_writer=torch.linalg.solve(s.mode['R_U'],s.mode['writer']))
path=P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt';torch.save(artifact,path)
plan=dict(source_products=512,outputs=6,input_width=1152,learning_rates=[.01,.05],starts=['spectral','spectral_1pct_perturbed'],steps=4000,schedule='cosine',optimizer='Adam',dtype='float64',native_forwards=0,selection='Lowest coefficient objective only; opened native errors never select the winner.',baseline='MULTIMODE_PAIR_BASELINES_V1.json',input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),predictions=dict(pred_a_instrument='Implicit vs dense coefficient loss difference<1e-8; finite factors.',pred_b_fidelity='Every selected component normalized error<=.15 and <=1.10 independent pair baseline.',pred_c_simplicity='512 products versus768; total stored floats<.90 independent baseline.'),scope='Original folded quadratic forms in expanded source metric; old exact centered affine terms retained. Fits shared product dictionary, not fixed projected inputs. Seven opened cached prefixes diagnostic only; no native promotion without fresh tests.')
(P/'SHARED_PRODUCT_NATIVE_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n');print(plan)
