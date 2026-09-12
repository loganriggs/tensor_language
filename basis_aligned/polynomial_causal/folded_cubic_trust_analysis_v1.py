"""Interpret terminal warm solver pilot with complete coefficient functions."""
from pathlib import Path
import json,time
import torch
from cubic_projected_function_compare_v1 import compare
from cubic_secant_coordinates_v1 import components
from folded_producer_cubic_weights_v1 import weights as producer_weights
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);tic=time.perf_counter()
 out=P/'FOLDED_CUBIC_TRUST_PILOT_V1_ANALYSIS.json';assert not out.exists()
 result=json.loads((P/'FOLDED_CUBIC_TRUST_PILOT_V1_RESULT.json').read_text());artifact=torch.load(P/'FOLDED_CUBIC_TRUST_PILOT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 old=torch.load(P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms']
 banks={f'old{i}':(a,torch.eye(len(a))) for i,a in enumerate(old)}
 for a in artifact['programs']:
  name=f"{a['coordinates']}{a['arm']}";x=a['theta'];banks[name]=(x,torch.eye(len(x))) if a['coordinates']=='raw' else components(x,a['separation'])
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')['current_readers'];(q1,k1,q2,k2,v,o),_,_=producer_weights(sd,C,'cpu')
 pairs=[(f'old{i}',f'{mode}{i}') for i in [0,1] for mode in ['raw','secant']]+[('raw0','raw1'),('secant0','secant1'),('raw0','secant0'),('raw1','secant1')];records=[]
 with torch.no_grad():
  for pos in [7,0]:
   r=rotary(8,128).T@rotary(pos,128);w=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
   for first,second in pairs:
    z=compare(banks[first],banks[second],w)
    row=dict(position=pos,first=first,second=second,cosine=float(z['cosine']),symmetric_relative_error=float(z['symmetric_relative_error']),energies=[float(z[k].sum()) for k in ['first_energy','second_energy']],head_energies=[z[k].tolist() for k in ['first_energy','second_energy']],gram_conditions=[float(t) for t in z['gram_conditions']]);records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='head_energies'}),flush=True)
 report=dict(records=records,seconds=time.perf_counter()-tic,all_stationary=all(r['stationary'] for r in result['rows']),scope='Exact complete projected coefficient-function comparisons with private query/output maps re-solved at each position; no native input normalizers or text behavior. Warm-starts are not independent discovery starts. No cross-head context identification implied.')
 out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
