"""Exact fixed-writer/attention10 mixed composition; CPU weights-only screen."""
from pathlib import Path
import torch,json,time
from head17_source_interface_v1 import CHECKPOINT
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();torch.manual_seed(9130958)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
 program=torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
 w=program['direction'].double()*float(sd['transformer.h.10.lambdas'][0])
 l,r,d=[sd['transformer.h.10.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
 o=sd['transformer.h.10.attn.c_proj.weight'].double()
 j=(d*(r@w)[None,:])@l+(d*(l@w)[None,:])@r;m=j@o
 u=torch.randn(8,1152,dtype=torch.float64);x=u@o.T
 direct=((l@w)*(x@r.T)+(r@w)*(x@l.T))@d.T
 replay=float((u@m.T-direct).norm()/direct.norm());assert replay<1e-10
 rows=[]
 for name,mat in [('O',o),('Jw_O',m)]:
  energy=mat.reshape(1152,9,128).square().sum((0,2));shares=energy/energy.sum();order=shares.argsort(descending=True)
  s=torch.linalg.svdvals(mat);tail=(s.square().sum()-s.square().cumsum(0)).clamp_min(0)/s.square().sum()
  rank10=int(torch.nonzero(tail<=.01)[0])+1
  rows.append(dict(name=name,head_energy_shares=shares.tolist(),head_order=order.tolist(),top3_energy=float(shares[order[:3]].sum()),rank_for_10pct_error=rank10,
    rank_errors={str(rank):float(tail[rank-1].sqrt()) for rank in [32,64,128,256,512,768,1024]}))
 result=dict(pred_a=rows[1]['top3_energy']>=.9,pred_b=rows[1]['rank_for_10pct_error']<=128,rows=rows,direct_algebra_relative_error=replay,seconds=time.perf_counter()-tic,
  scope='Exact interaction K(lambda*w,O*u), fixed writer and arbitrary head-write input; O comparator. Omits other residual modes, attention producer, normalization, background and suffix. Weight energy not causal head attribution.')
 (P/'FIXED_WRITER_ATTENTION_OPERATOR_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
