"""Native cross-boundary interaction energy of fitted input subspaces."""
from pathlib import Path
import torch,json
from sparse_orthogonal_quadratic_core_v1 import input_marginal,orthogonal_core
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
torch.set_num_threads(2);dt=torch.float64;torch.manual_seed(815)
t=torch.randn(4,6,6,dtype=dt);t=(t+t.transpose(-1,-2))/2;e,_=torch.linalg.qr(torch.randn(6,2,dtype=dt));proj=e@e.T;s=torch.einsum('oij,okj->ik',t,t);inside=torch.einsum('ir,oij,js->ors',e,t,e).square().sum();one=torch.trace(e.T@s@e);comm=(t@proj-proj@t).square().sum();control=float(abs(comm-2*(one-inside))/t.square().sum());assert control<=1e-10
sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u;root=torch.linalg.cholesky(metric).T
l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']];z=root@d;marginal=input_marginal(l,r,z.T@z);total=marginal.trace()
initial=torch.load(P/'SPARSE_ORTHOGONAL_CORE_V1_CENTERED_COMPACT.pt',weights_only=False,map_location='cpu')['basis'].T
learned=torch.load(P/'SPARSE_CORE_RCG_V1_CHECKPOINT.pt',weights_only=False,map_location='cpu');q=learned['q'];edges=learned['edges'];active=torch.unique(edges)
parent=list(range(128))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
for i,j in edges.T.tolist():parent[find(i)]=find(j)
groups={}
for i in active.tolist():groups.setdefault(find(i),[]).append(i)
rows=[]
for name,basis in [('initial128',initial),('learned128',q),('learned_active',q[:,active])]:
    w,_=orthogonal_core(l,r,z,basis.T);within=w.square().sum();one=torch.trace(basis.T@marginal@basis);mixed=2*(one-within);outside=total-2*one+within
    assert float(min(within,mixed,outside))>=-1e-10*float(total)
    rows.append(dict(name=name,input_dimensions=basis.shape[1],within_energy_fraction=float(within/total),mixed_energy_fraction=float(mixed/total),outside_energy_fraction=float(outside/total),sum_replay_error=float(abs((within+mixed+outside)/total-1)),cross_port_fraction_of_projected_read_energy=float((one-within)/one),commutator_squared_over_total=float(mixed/total)))
result=dict(control_relative_error=control,subspaces=rows,learned_graph_active_nodes=len(active),learned_graph_component_sizes=sorted([len(x) for x in groups.values()],reverse=True),scope='Exactnativecenteredquadratic coefficients, notjustsurrogategraph. Mixedinteractionenergy quantifies requiredports; nonzeroports reject closedorthogonalblockinterpretation, notopenreusablecircuits. No activationdata or nativeforward.')
(P/'SPARSE_CORE_NATIVE_PORTS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
