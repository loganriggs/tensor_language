from pathlib import Path
import json,torch
from joint_attention_mixed_ports_v1 import contract,decompose
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(913229);shapes=[(5,13),(5,13),(5,13,128)];n=tuple(torch.randn(s,dtype=torch.float64) for s in shapes);c=tuple(x+.1*torch.randn_like(x) for x in n);r=tuple(x+.2*torch.randn_like(x) for x in n);a=tuple(x+y-z+.03*torch.randn_like(z) for x,y,z in zip(c,r,n));d=decompose(n,c,r,a);ref=contract(a)-contract(c)-contract(r)+contract(n);rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30));replay=rel(d['total'],ref);swap=rel(decompose(n,r,c,a)['total'],d['total']);qkswap=rel(decompose(*[tuple([x[1],x[0],x[2]]) for x in [n,c,r,a]])['total'],d['total'])
 # Pure cross-edit witness: each port has zero own mixed defect, yet joint response is nonzero.
 c2=(n[0]+.1,n[1],n[2]);r2=(n[0],n[1]-.2,n[2]);a2=(c2[0],r2[1],n[2]);w=decompose(n,c2,r2,a2);expected=-.02*n[2].sum(-2)
 witness=rel(w['total'],expected);defectnorm=float(w['defect'].norm());assert max(replay,swap,qkswap,witness)<1e-11 and defectnorm<1e-12
 result=dict(replay_error=replay,child_remainder_swap_error=swap,QK_factor_swap_error=qkswap,cross_monomials=len(d['cross_terms']),defect_subset_groups=len(d['defect_terms']),pure_cross_witness_error=witness,witness_total_norm=float(w['total'].norm()),witness_defect_norm=defectnorm,scope='FP64 synthetic source-summed joint QK1*QK2*value contraction; exact finite corner decomposition and falsifier of independent-port-mixed-only approximation. No native text validation or semantic port identification.')
 (P/'JOINT_ATTENTION_MIXED_PORTS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
