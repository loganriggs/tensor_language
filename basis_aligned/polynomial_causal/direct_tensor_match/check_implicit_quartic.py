"""Independent dense permutation/gradient validation and sampling-variance control."""
import itertools,json,math
from pathlib import Path
import torch
from implicit_quartic import entries
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1609)
d=6;params=[torch.randn(*shape,requires_grad=True) for shape in [(3,4),(4,5),(4,5),(5,7),(7,d),(7,d)]];C,L2,R2,D1,L1,R1=params
S=torch.einsum('ak,ki,kj->aij',D1,L1,R1);raw=torch.einsum('vk,ka,kb,aij,bmn->vijmn',C,L2,R2,S,S);symmetric=sum(raw.permute((0,)+tuple(i+1 for i in p)) for p in itertools.permutations(range(4)))/24
idx=torch.tensor(list(itertools.product(range(d),repeat=4)));got=entries(*params,idx);ref=symmetric.reshape(3,-1).T;relative=float(((got-ref).norm()/ref.norm()).detach());assert relative<1e-12
weights=torch.randn_like(got);g1=torch.autograd.grad((got*weights).sum(),params,retain_graph=True);g2=torch.autograd.grad((ref*weights).sum(),params);gradient=max(float((a-b).norm()/b.norm()) for a,b in zip(g1,g2));assert gradient<1e-11
values=got.detach().square().sum(-1);dense_energy=float(symmetric.detach().square().sum());enumerated=float(values.mean()*d**4);assert abs(enumerated-dense_energy)/dense_energy<1e-12
# Uniform entry sampling estimates coefficient Frobenius error, not Gaussian function error.
variance=[]
for name,v in [('dense_two_layer',values),('single_coordinate_quartic',torch.nn.functional.one_hot(torch.tensor(0),d**4).double())]:
 cv=float(v.std(unbiased=False)/v.mean())
 for batch in [512,4096,32768]:
  torch.manual_seed(batch);est=v[torch.randint(len(v),(100,batch))].mean(1)/v.mean()
  variance.append(dict(target=name,samples=batch,theoretical_relative_standard_error=cv/math.sqrt(batch),observed_relative_rmse=float((est-1).square().mean().sqrt()),zero_estimates=int((est==0).sum()),repetitions=100))
out=dict(entry_relative_error=relative,gradient_relative_error=gradient,enumerated_frobenius_relative_error=abs(enumerated-dense_energy)/dense_energy,sampling=variance,scope='Exact fully symmetrized coefficient-entry oracle and its gradients; small dense teacher only. Uniform stochastic loss is unbiased for squared Frobenius numerator, can miss concentrated structure; ratio of estimated losses need not be unbiased. No native fit yet.')
(P/'IMPLICIT_QUARTIC_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
