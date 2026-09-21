"""Rank relaxation for two quadratic reads sharing one private feature dictionary.
For sharedS and orthogonal complementV, M=[[sqrt(2)B1,sqrt(2)B2],[C1,C2]].
Any privateD with pcolumns yields rank(M_hat)<=p; the common branch changes onlyA.
The best rank-p tail is thus a coefficient-error lower bound, even without
requiring reconstructed privateC blocks symmetric. Does not bound movingS.
"""
from pathlib import Path
import json,torch,time
from pairwise_reader_graph import GROUPS
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);states=torch.load(P/'PAIRWISE_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);meta=json.loads((P/'PAIRWISE_SUBSPACE_NATIVE_V1.json').read_text());key=meta['primary'];params=states[key];A=torch.linalg.inv(d['inverse_root']);rows=[]
for scope in ('fixed_pair_span','free_private_directions'):
 for j,(a,b) in enumerate(GROUPS):
  Q=A@torch.stack(d['pairs'][j]['Qs'])@A;shared=torch.cat([params[a],params[b]],1);U=torch.linalg.qr(shared,mode='complete').Q;S=U[:,:128]
  V=U[:,128:] if scope=='free_private_directions' else torch.linalg.qr(params[3+j]-S@(S.T@params[3+j]),mode='reduced').Q
  Z=torch.cat([S,V],1);T=Z.T@Q@Z;B=T[:,:128,128:];C=T[:,128:,128:];M=torch.cat([2**.5*torch.cat(list(B),1),torch.cat(list(C),1)],0)
  spectrum=torch.linalg.eigvalsh(M@M.T).clamp_min(0);tail=spectrum[:-224].sum();outside=(Q-Z@T@Z.T).square().sum();lower=float((tail+outside)/Q.square().sum());rows.append(dict(scope=scope,pair=j+1,private_width=224,ambient_complement_width=V.shape[1],relative_squared_error_lower_bound=lower,outside_span_squared_error=float(outside/Q.square().sum()),rank_tail_squared_error=float(tail/Q.square().sum())))
aggregate={scope:(sum(r['relative_squared_error_lower_bound'] for r in rows if r['scope']==scope)/3)**.5 for scope in ('fixed_pair_span','free_private_directions')}
out=dict(primary=key,records=rows,equal_pair_root_bounds=aggregate,covariance_guard=meta['plan']['covariance_guard'],seconds=time.monotonic()-start,scope='Fixed learned shared128space perpair; common quadratic branch arbitrary, private224features arbitrary with two symmetric cores. Fullcomplementbound permits private directions anywhere in1152inputs. Bounds do not constrain changed sharedspaces, directshared-private products, or generalDAGs. Floating-point spectral calculation, not an interval certificate.')
(P/'PRIVATE_BRANCH_RANK_BOUND_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
