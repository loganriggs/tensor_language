"""Pair opposite-signed spectral terms into exact rank-two product atoms."""
from pathlib import Path
import torch,json,hashlib
from shared_quadratic_products import mixed_coefficient_loss,materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
base=P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt';d=torch.load(base,weights_only=True);left=[];right=[];weights=[];energies=[]
for output,M in enumerate(d['teacher']):
 ev,V=torch.linalg.eigh(M);pos=(ev>0).nonzero().flatten();pos=pos[ev[pos].argsort(descending=True)];neg=(ev<0).nonzero().flatten();neg=neg[ev[neg].abs().argsort(descending=True)]
 paired=min(len(pos),len(neg))
 for a,b in zip(pos[:paired],neg[:paired]):
  u=ev[a].sqrt()*V[:,a];v=(-ev[b]).sqrt()*V[:,b];left.append(u+v);right.append(u-v);w=torch.zeros(6,dtype=torch.float64);w[output]=1;weights.append(w);energies.append(ev[a].square()+ev[b].square())
 for j in torch.cat([pos[paired:],neg[paired:]]):
  u=ev[j].abs().sqrt()*V[:,j];left.append(u);right.append(u);w=torch.zeros(6,dtype=torch.float64);w[output]=ev[j].sign();weights.append(w);energies.append(ev[j].square())
order=torch.stack(energies).argsort(descending=True)[:384];L=torch.stack(left,1)[:,order];R=torch.stack(right,1)[:,order];W=torch.stack(weights,1)[:,order]
implicit=float(mixed_coefficient_loss(d['teacher'],L,R,W));explicit=float((materialize_mixed(L,R,W)-d['teacher']).square().sum()/d['teacher'].square().sum());assert abs(implicit-explicit)<1e-10
init=P/'MIXED_PRODUCT_NATIVE_INITIALIZATION_V1.pt';torch.save(dict(left=L,right=R,weights=W),init)
plan=dict(source_products=384,outputs=6,input_width=1152,learning_rates=[.01,.05],starts=['spectral','spectral_1pct_perturbed'],steps=4000,schedule='cosine',optimizer='Adam',dtype='float64',native_forwards=0,selection='Lowest coefficient objective only; no native outcome selection.',input_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),initialization_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),predictions=dict(pred_a_instrument='Implicit/dense dimensionless coefficient loss discrepancy<1e-8; finite factors.',pred_b_fidelity='Each selected component opened error<=.15 and<=1.10 independent pair baseline.',pred_c_simplicity='At most384 products versus768, floats<=1.01 writer-deduplicated baseline897804.'),scope='General mixed-product CP of six original folded quadratic forms; expanded metric and fixed exact old affine branches. Compare half products at approximately matched storage, not equal product budget.')
(P/'MIXED_PRODUCT_NATIVE_PLAN_V1.json').write_text(json.dumps(plan,indent=2)+'\n')
(P/'MIXED_PRODUCT_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(initial_relative_squared_coefficient_loss=implicit,dense_loss_replay=abs(implicit-explicit),products_per_initial_output=(W!=0).sum(1).tolist(),left_right_relative_difference=float((L-R).norm()/L.norm())),indent=2)+'\n');print(plan);print('Initial mixed coefficient error',implicit**.5)
