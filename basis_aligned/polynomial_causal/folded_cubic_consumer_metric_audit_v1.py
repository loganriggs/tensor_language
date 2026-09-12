"""Re-score frozen fitted functions in the previously derived consumer metric.
No source-factor refit. Projection onto a fixed source span commutes with an
invertible linear output transform, so these are the same physical functions.
"""
from pathlib import Path
import json,time,torch
from cubic_secant_coordinates_v1 import components
from cubic_secant_block_v1 import gram_kernel
from cubic_projected_function_compare_v1 import solve
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);tic=time.perf_counter();out=P/'FOLDED_CUBIC_CONSUMER_METRIC_V1_AUDIT.json';assert not out.exists()
 old=torch.load(P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_ARTIFACT.pt',weights_only=True)['atoms'];new=torch.load(P/'FOLDED_CUBIC_TRUST_PILOT_V1_ARTIFACT.pt',weights_only=True)['programs'];banks={f'old{i}':(a,torch.eye(len(a))) for i,a in enumerate(old)}
 for z in new:
  a=z['theta'];banks[f"{z['coordinates']}{z['arm']}"]=(a,torch.eye(len(a))) if z['coordinates']=='raw' else components(a,z['separation'])
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,C,'cpu')
 H=torch.load(P/'CONSUMER_PULLBACK_V1_ARTIFACT.pt',weights_only=True)['metric'];ev,U=torch.linalg.eigh(H);assert ev.min()>0;H=H/(H.trace()/4);ev,U=torch.linalg.eigh(H);root=(U*ev.sqrt()[None])@U.T
 records=[]
 with torch.no_grad():
  for pos in [7,0]:
   r=rotary(8,128).T@rotary(pos,128);ka=torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1);kb=torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1)
   for metric,output in [('reading_euclidean',o),('consumer_pullback',torch.einsum('ab,bhk->ahk',root,o))]:
    energies={}
    for name,(c,m) in banks.items():
     g,k=gram_kernel(c,m,(q1,ka,q2,kb,v,output));energies[name]=torch.stack([torch.trace(solve(g,x)) for x in k])
    for arm in [0,1]:
     before=energies[f'old{arm}'];after=energies[f'secant{arm}'];delta=after-before;total=float(delta.sum());idx=18
     records.append(dict(position=pos,metric=metric,arm=arm,old_capture=float(before.sum()),new_capture=float(after.sum()),gain_ratio=float(after.sum()/before.sum()),head13_0_fraction_of_net_gain=float(delta[idx]/total) if abs(total)>1e-12 else None,head8_2_gain=float(delta[2]),head9_8_gain=float(delta[17]),per_head_gain=delta.tolist()))
 result=dict(rows=records,metric_normalized_eigenvalues=ev.tolist(),seconds=time.perf_counter()-tic,scope='Frozen coefficient functions reweighted by existing weights-only downstream polynomial Jacobian metric, with trace normalization. No source refit, no language data, no actual normalized consumer-effect approximation guarantee, and no target total-energy denominator estimated.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='per_head_gain'} for r in records]},indent=2))
if __name__=='__main__':main()
