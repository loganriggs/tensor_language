"""Keep all three components visible in the output-contrast diagnosis."""
from pathlib import Path
import json,torch
from source_graph_metrics import matrices
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);m=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[m['winner']];T=d['teacher'];root=torch.linalg.inv(d['inverse_root']);shared=matrices(p['shared_mixed']);b=p['private_pair'];i,j,k=b['product_indices'].long();u=b['shared_reader'][:,i];v=b['shared_reader'][:,j];L=torch.where((k==1)[None],u+v,u);R=torch.where((k==1)[None],u-v,v);raw=torch.einsum('ir,ro,jr->oij',L,b['product_weights'],R);Qhat=torch.cat([shared,(raw+raw.transpose(-1,-2))/2]);Mhat=torch.stack([root@Q@root for Q in Qhat])/d['scales'][:,None,None]
def spectrum(A,B):
 F=A.flatten(1);e,U=torch.linalg.eigh(F@F.T);e=e.flip(0);U=U.flip(1);G=U.T@F;H=U.T@B.flatten(1)
 return dict(eigenvalues=e.tolist(),energy_fractions=(e/e.sum()).tolist(),teacher_anchored_feature_cosines=((G*H).sum(1)/(G.norm(dim=1)*H.norm(dim=1))).tolist(),relative_feature_errors=((G-H).norm(dim=1)/G.norm(dim=1)).tolist()),U
joint,U=spectrum(T,Mhat);pairwise=[spectrum(T[2*j:2*j+2],Mhat[2*j:2*j+2])[0] for j in range(3)]
u=U[:,-1];weak=torch.einsum('o,oij->ij',u,T);removed=u[:,None,None]*weak;inv=d['inverse_root'];D=torch.stack([inv@M@inv for M in removed])*d['scales'][:,None,None];Qs=torch.stack([Q for pair in d['pairs'] for Q in pair['Qs']]);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();arms=[]
for centered in [False,True]:
 reads=torch.einsum('ni,oij,nj->no',z,Qs-D,z)
 if centered:
  linear=2*torch.einsum('oij,j->oi',D,d['mu']);bias=torch.einsum('ij,oji->o',d['old_covariance'],D)-torch.einsum('i,oij,j->o',d['mu'],D,d['mu']);reads+=z@linear.T+bias
 errors=[]
 for j,pair in enumerate(d['pairs']):
  phi=((h@pair['a']-.5*reads[:,2*j])/s-pair['alpha'])*(reads[:,2*j+1]/s-pair['beta']);truth=pair['truth'][ids];errors.append(float((phi-truth).norm()/(truth-truth.mean()).norm()))
 arms.append(dict(centered_remainder_only=centered,component_errors=errors))
out=dict(joint=joint,per_pair=pairwise,weak_joint_contrast_removal=arms,scope='All six original source quadratics and parentshared/private approximations. Teacher-anchored directions; exploratory opened-state removal, not fresh/causal adoption. Fitting firstfour cannotrepairprivate3.')
(P/'ALL_SOURCE_CONTRASTS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
