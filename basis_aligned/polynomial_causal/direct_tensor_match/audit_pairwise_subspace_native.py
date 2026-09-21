"""Independent dense projection and conditional component audit of native relaxation."""
from pathlib import Path
import json,time,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);states=torch.load(P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);meta=json.loads((P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').read_text());Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=Q.dtype);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
def evaluate(H,rms=scale):
 delta=Q-H;linear=2*torch.einsum('oij,j->oi',delta,d['mu']).T;bias=torch.einsum('ij,oji->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu'])
 reads=torch.einsum('ni,oij,nj->no',z,H,z)+z@linear+bias;grads=2*torch.einsum('oij,nj->noi',H,z)+linear.T[None];values=[];jac=[]
 for j,pair in enumerate(d['pairs']):
  a=(h@pair['a']-.5*reads[:,2*j])/rms-pair['alpha'];b=reads[:,2*j+1]/rms-pair['beta'];values.append(a*b);jac.append(-.5*(b/rms)[:,None]*grads[:,2*j]+(a/rms)[:,None]*grads[:,2*j+1])
 return torch.stack(values,1),torch.stack(jac,1)
true_values,true_jac=evaluate(Q);truth=torch.stack([p['truth'][ids] for p in d['pairs']],1);truth_replay=float((true_values-truth).norm()/truth.norm());cached_values,_=evaluate(Q,d['scale'][ids]);cached_replay=float((cached_values-truth).norm()/truth.norm());assert cached_replay<1e-10
rows=[]
for key,params in states.items():
 A,inv=(S,d['inverse_root']) if key.startswith('calibration') else (I,I)
 # SVD replaces fit-time QR; independently reconstruct the same spans.
 hats=[];ranks=[]
 for j in range(3):
  groups=((0,1),(0,2),(1,2));a,b=groups[j];columns=[params[a],params[b],params[3+j]] if '_pairwise_' in key else [params[0],params[j+1]];U,s,_=torch.linalg.svd(torch.cat(columns,1),full_matrices=False);rank=int((s>1e-10*s[0]).sum());assert rank==352;U=U[:,:rank];T=A@Q[2*j:2*j+2]@A;hats.append(inv@U@(U.T@T@U)@U.T@inv);ranks.append(rank)
 H=torch.cat(hats);scores=[]
 for B in (I,S):
  T=B@Q@B;E=B@(H-Q)@B;scores.append(float((E.square().sum((-1,-2)).reshape(3,2).sum(1)/T.square().sum((-1,-2)).reshape(3,2).sum(1)).mean().sqrt()))
 saved=next(r for r in meta['records'] if r['key']==key);replay=max(abs(scores[0]-saved['native_error']),abs(scores[1]-saved['covariance_error']));assert replay<1e-8
 values,jac=evaluate(H);err=((values-truth).norm(dim=0)/(truth-truth.mean(0)).norm(dim=0)).tolist();jerr=[float((jac[:,j]-true_jac[:,j]).norm()/true_jac[:,j].norm()) for j in range(3)]
 rows.append(dict(key=key,span_ranks=ranks,native_error=scores[0],covariance_error=scores[1],independent_metric_replay=replay,per_mode_errors=err,euclidean_jacobian_errors=jerr))
# Literal generic dense-core cost using three64pairwise dictionaries and private224bases.
# Count each dictionary-internal product once across its two consumers.
r=64;p=224;n=2*r+p;projection=1152*(3*r+3*p);products=3*n*(n+1)//2-3*r*(r+1)//2;readout=3*n*(n+1)
price=dict(shared_width=r,private_width=p,source_projection_multiplications=projection,distinct_nonlinear_products=products,source_readout_multiplications=readout,source_total_multiplications=projection+products+readout,stored_floats=projection+readout+11532,scope='Pairwise dense symmetric cores, within-dictionary products computed once across their two consumers; full coefficients generic. Source arithmetic excludes fixed affine and downstream component work, as in original comparisons. Not an exported/adopted graph.')
base=json.loads((P/'JOINT_OVERLAP_PLAN_V1.json').read_text())['baseline'];primary=next(r for r in rows if r['key']==meta['primary'])
checks=dict(primary_component_guard=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])),primary_jacobian_guard=all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),dense_core_arithmetic_guard=price['source_total_multiplications']<=.8*base['source_total_multiplications'])
(P/'PAIRWISE_SUBSPACE_NATIVE_AUDIT_V1.json').write_text(json.dumps(dict(records=rows,primary=meta['primary'],truth_replay_recomputed_rms=truth_replay,truth_replay_original_cached_scale=cached_replay,truth_replay_correction='Initial audit assumed cached truth used recomputed RMS. Its producer uses saved scale; replay is checked with that scale. Candidate metrics retain recomputed RMS as in all prior comparisons.',dense_core_price=price,diagnostic_comparisons=checks,seconds=time.monotonic()-start,scope='Opened448states; coefficient metrics independently reconstructed via SVD. Components and fixed-h z derivatives retain native states/normalization. No fresh model intervention or stable circuit claim.'),indent=2)+'\n');print(json.dumps(dict(records=rows,price=price,checks=checks),indent=2))
