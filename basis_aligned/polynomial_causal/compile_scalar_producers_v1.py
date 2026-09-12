"""Compile weights and all vocabulary first-value readings, without text fitting."""
from pathlib import Path
import torch,json,time
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_PRODUCERS_COMPILE_V1_RESULT.json';assert not out.exists()
 merged=torch.load(P/'STRUCTURED_PRODUCER_SHARED_OUTPUT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');bank=torch.load(P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 binding=json.loads((P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 p={}
 for short,original in [('q1','c_q'),('k1','c_k'),('q2','c_q2'),('k2','c_k2')]:p[short]=torch.stack([state[f'transformer.h.{layer}.attn.{original}.weight'].reshape(9,128,1152)[head].float().clone() for layer,head in ((8,2),(9,8))])
 readers=bank['source_readers'][[2,17]];embedding=state['transformer.wte.weight'];lam=state['transformer.h.0.lambdas'].float();table=torch.empty(len(embedding),2,dtype=torch.float64)
 for start in range(0,len(table),1024):
  end=min(start+1024,len(table));x=F.rms_norm(embedding[start:end].float(),(1152,));first=F.rms_norm(lam[0]*x+lam[1]*x,(1152,));table[start:end]=first.double()@readers[:,1152:].T
 p.update(current_value_readers=readers[:,:1152].clone(),first_token_values=table,shared_output=merged['shared_output'].clone(),output_coefficients=merged['coefficients'].clone())
 count=sum(x.numel() for x in p.values());nbytes=sum(x.numel()*x.element_size() for x in p.values());assert count==1282566
 torch.save(p,P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt')
 result=dict(program_scalars=count,program_tensor_bytes=nbytes,vocabulary_rows=len(table),full_qk_scalars=4*2*128*1152,seconds=time.perf_counter()-tic,scope='Two frozen scalar-value producers into shared four-reading output. Token lookup built from native embedding and block0 re-entry/RMS. Full layer8/9 normalized context inputs and exact QK remain external/retained. Native validation pending.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
