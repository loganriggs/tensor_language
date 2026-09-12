"""Exact supports implied by native joint QK/value equations, no factor fitting."""
from pathlib import Path
import torch,json,time
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(824);tic=time.perf_counter();out=P/'CUBIC_EXACT_SUPPORT_V1_AUDIT.json';assert not out.exists()
 binding=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'].double();groups=[];records=[]
 for layer,head in [(8,2),(9,8),(13,0)]:
  prefix=f'transformer.h.{layer}.attn.';qa,ka,qb,kb=[sd[prefix+n+'.weight'].double().reshape(9,128,1152)[head] for n in ['c_q','c_k','c_q2','c_k2']];mu=float(sd[prefix+'lamb']);v=torch.cat(((1-mu)*sd[prefix+'c_v.weight'].double(),mu*sd['transformer.h.0.attn.c_v.weight'].double()),1).reshape(9,128,2304)[head];o=sd[prefix+'c_proj.weight'].double().reshape(1152,9,128)[:,head]
  full=torch.cat((torch.cat((ka,torch.zeros_like(ka)),1),torch.cat((kb,torch.zeros_like(kb)),1),v));folded=torch.cat((full[:256],C@o@v));query=torch.cat((qa,qb));groups.append((full,folded,query))
  for label,matrix in [('full_source',full),('four_read_source',folded),('query',query)]:
   basis=torch.linalg.qr(matrix.T,mode='reduced').Q;error=float((matrix-(matrix@basis)@basis.T).norm()/matrix.norm());records.append(dict(head=f'{layer}.{head}',object=label,ambient=matrix.shape[1],support_columns=basis.shape[1],coordinate_reduction=matrix.shape[1]/basis.shape[1],relative_reconstruction_error=error))
  basis=torch.linalg.qr(full.T,mode='reduced').Q;qbase=torch.linalg.qr(query.T,mode='reduced').Q;x=torch.randn(64,2304);q=torch.randn(64,1152);xp=(x@basis)@basis.T;qp=(q@qbase)@qbase.T
  def fn(q,x):
   a=q@qa.T;b=q@qb.T;c=x[:,:1152]@ka.T;d=x[:,:1152]@kb.T;gate=1/(128**2*((a.square().mean(-1)+1.1920928955078125e-7)*(b.square().mean(-1)+1.1920928955078125e-7)*(c.square().mean(-1)+1.1920928955078125e-7)*(d.square().mean(-1)+1.1920928955078125e-7)).sqrt());return gate[:,None]*(a*c).sum(-1)[:,None]*(b*d).sum(-1)[:,None]*(x@v.T@o.T)
  y=fn(q,x);z=fn(qp,xp);records.append(dict(head=f'{layer}.{head}',object='normalized_pair_replay',relative_reconstruction_error=float((y-z).norm()/y.norm())))
 for index,label in enumerate(['full_source','four_read_source','query']):
  matrix=torch.cat([g[index] for g in groups[:2]]);basis=torch.linalg.qr(matrix.T,mode='reduced').Q;records.append(dict(head='8.2+9.8',object=label,ambient=matrix.shape[1],support_columns=basis.shape[1],coordinate_reduction=matrix.shape[1]/basis.shape[1],relative_reconstruction_error=float((matrix-(matrix@basis)@basis.T).norm()/matrix.norm())))
 report=dict(pred_a=all(z['relative_reconstruction_error']<=1e-10 for z in records),records=records,seconds=time.perf_counter()-tic,scope='Untruncated QR spans; column counts are safe dimension upper bounds, not estimated minimal ranks. Source and query support of supplied native states; no prefix closure. Four-read source bound preserves folded C O V only, not the entire physical value output. Random FP64 pair replay at identity relative rotation; matrix identities imply all rotations. No fit or GPU speedup measured.')
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
