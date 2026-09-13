"""CPU reconstruction/rank screen, with signed cancellation and scale changes."""
from pathlib import Path
import json,torch
from writer_span_coordinates_v1 import prepare,transform,edit_error_bound
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(7131111)
 w=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)['writers'].double()
 x=torch.randn(1152,dtype=torch.float64);y=torch.randn_like(x);y-=x*(y@x)/(x@x)
 pairs={'stored':w,'near':torch.stack([x,x+1e-7*y]),'duplicate':torch.stack([x,x]),'rescaled':w*torch.tensor([1e-4,1e4])[:,None]}
 rows=[]
 for name,pair in pairs.items():
  c=prepare(pair);a=torch.randn(20,2,dtype=torch.float64);a[:2]=torch.tensor([[1.,-1.],[2.,-2.]])
  actual=a@pair;pred=transform(a,c)@c['basis'];err=(pred-actual).norm(dim=-1)
  scale=a.norm(dim=-1)*pair.norm();bound=edit_error_bound(a,c)
  relative=float((err/scale).max());excess=float(((err-bound)/scale).max())
  rows.append(dict(name=name,rank=c['rank'],relative_error=relative,bound_excess=excess,basis_scalars=c['basis'].numel(),map_scalars=c['map'].numel(),original_scalars=pair.numel()))
 out={'pred_a':all(r['relative_error']<1e-12 for r in rows),'pred_b':all(r['bound_excess']<1e-14 for r in rows),'pred_c':[r['rank'] for r in rows]==[2,2,1,2],'rows':rows,'scope':'Writer edit equivalence, not behavioral circuit identification. Full-rank basis plus map costs four extra scalars for two writers; rank-one duplicates save representation.'}
 (P/'WRITER_SPAN_COORDINATES_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
