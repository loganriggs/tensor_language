"""Preservation and conditioning of the exact three-product chart on native fits."""
from pathlib import Path
import time,json,torch
from cubic_cluster_coordinates_v1 import encode,components as cluster_components
from cubic_secant_coordinates_v1 import raw_atoms,components as secant_components
from cubic_secant_block_v1 import gram_kernel
from cubic_projected_function_compare_v1 import compare,solve
from shared_cubic_source_projection_v1 import cross_factors,atom_gram
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent

def execute(bank,q,s,w):
 c,m=bank;a,b,v=cross_factors(c,*w)
 qa=torch.einsum('nd,hrtd->nhrt',q,a);qb=torch.einsum('nd,hrtd->nhrt',q,b)
 beta=torch.einsum('nhrt,hrto->nhro',qa*qb,v);beta=torch.einsum('ra,nhao->nhro',m,beta)
 phi=torch.einsum('nd,rid->nri',s,c).prod(-1)@m.T;g=m@atom_gram(c)@m.T;dual=solve(g,phi.T).T
 return torch.einsum('nr,nhro->nho',dual,beta)

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);tic=time.perf_counter();out=P/'CUBIC_CLUSTER_NATIVE_V1_AUDIT.json';assert not out.exists()
 saved=torch.load(P/'FOLDED_CUBIC_TRUST_PILOT_V1_ARTIFACT.pt',weights_only=True)['programs'];old=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in old if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,C,'cpu')
 gen=torch.Generator().manual_seed(9122037);q=torch.randn(128,1152,generator=gen);s=torch.randn(128,2304,generator=gen);rows=[];charts=[]
 with torch.no_grad():
  for z in saved:
   if z['coordinates']!='secant':continue
   arm=z['arm'];raw=raw_atoms(z['theta'],z['separation']);indices=[10,14,15] if arm==0 else [3,14,15];theta,chart=encode(raw,indices);charts.append(dict(arm=arm,theta=theta,chart=chart));before=secant_components(z['theta'],z['separation']);after=cluster_components(theta,chart)
   for pos in [7,0]:
    r=rotary(8,128).T@rotary(pos,128);w=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
    c=compare(before,after,w);oldg,_=gram_kernel(*before,w);newg,_=gram_kernel(*after,w)
    def cond(g):d=g.diagonal().sqrt();return float(torch.linalg.cond(g/d[:,None]/d[None,:]))
    x,y=execute(before,q,s,w),execute(after,q,s,w);error=float((x-y).norm()/x.norm());energyerror=float(abs(c['first_energy'].sum()-c['second_energy'].sum())/c['first_energy'].sum())
    row=dict(arm=arm,position=pos,old_normalized_condition=cond(oldg),new_normalized_condition=cond(newg),relative_energy_error=energyerror,coefficient_difference_squared_relative=float(abs(c['squared_difference'].sum())/c['first_energy'].sum()),sampled_execution_relative_error=error,reader_singular_values=chart['singular_values'].tolist());rows.append(row);print(json.dumps(row),flush=True)
 result=dict(pred_a=all(r['relative_energy_error']<=1e-6 and r['sampled_execution_relative_error']<=1e-5 for r in rows),pred_b=all(r['new_normalized_condition']<=100 and r['new_normalized_condition']<=r['old_normalized_condition']/10 for r in rows),rows=rows,seconds=time.perf_counter()-tic,scope='Same frozen full coefficient functions at fit/held positions;128synthetic diagonal polynomial execution controls. No refit, language behavior, normalizer closure or globally stable chart claim.')
 torch.save(charts,P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt');out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
