"""Complete-function and private-head comparisons after matched continuation."""
from pathlib import Path
import json,time,torch
from cubic_projected_function_compare_v1 import compare
from cubic_secant_coordinates_v1 import components as pair
from cubic_cluster_coordinates_v1 import components as cluster
from folded_producer_cubic_weights_v1 import weights
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);tic=time.perf_counter();out=P/'FOLDED_CUBIC_CLUSTER_CONTINUE_V1_ANALYSIS.json';assert not out.exists()
 banks={}
 for z in torch.load(P/'FOLDED_CUBIC_CLUSTER_CONTINUE_V1_ARTIFACT.pt',weights_only=True)['programs']:banks[z['coordinates']+str(z['arm'])]=(pair if z['coordinates']=='secant' else cluster)(z['theta'],z['chart'])
 for z in torch.load(P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt',weights_only=True):banks['prior'+str(z['arm'])]=cluster(z['theta'],z['chart'])
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(qa,ka,qb,kb,v,o),_,_=weights(sd,C,'cpu');records=[]
 pairs=[('cluster0','cluster1'),('secant0','cluster0'),('secant1','cluster1'),('prior0','cluster0'),('prior1','cluster1'),('prior0','prior1')]
 for pos in [7,0]:
  r=rotary(8,128).T@rotary(pos,128);w=(qa,torch.cat((torch.einsum('ab,hbd->had',r,ka),torch.zeros_like(ka)),-1),qb,torch.cat((torch.einsum('ab,hbd->had',r,kb),torch.zeros_like(kb)),-1),v,o)
  for first,second in pairs:
   z=compare(banks[first],banks[second],w);a=z['first_energy'];b=z['second_energy'];cross=z['cross_inner_product'];relative=((a+b-2*cross).clamp_min(0)/((a+b)/2).clamp_min(1e-30)).sqrt()
   row=dict(position=pos,first=first,second=second,cosine=float(z['cosine']),relative_error=float(z['symmetric_relative_error']),head13_0_relative_error=float(relative[18]),head13_0_cosine=float(cross[18]/(a[18]*b[18]).sqrt()),first_energy=a.tolist(),second_energy=b.tolist(),per_head_relative_error=relative.tolist());records.append(row);print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
 result=json.loads((P/'FOLDED_CUBIC_CLUSTER_CONTINUE_V1_RESULT.json').read_text());prices=[dict(arm=r['arm'],coordinates=r['coordinates'],capture_gain=r['final_capture']-r['initial_capture'],gain_per_second=(r['final_capture']-r['initial_capture'])/r['seconds'],hvp_per_evaluation=r['hessian_products']/r['evaluations']) for r in result['rows']]
 out.write_text(json.dumps(dict(records=records,prices=prices,seconds=time.perf_counter()-tic,scope='Exact coefficient comparison with original head labels and re-solved private query/output maps. Head13 result is separated from whole27bank disagreement. Native behavior is a separate frozen-bank screen.'),indent=2)+'\n')
if __name__=='__main__':main()
