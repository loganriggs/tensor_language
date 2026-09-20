"""Exact output Gram of the mixed midpoint tensor; no tensor materialization."""
import torch

def output_gram(teacher):
 C,L,R,D=teacher
 M=D@D.T
 cross=(L@R.T)*(R@M@L.T)
 H=(L@L.T)*(R@M@R.T)+(R@R.T)*(L@M@L.T)+cross+cross.T
 gram=C@H@C.T
 return (gram+gram.T)/2

def spectrum(gram):
 e=torch.linalg.eigvalsh(gram).flip(0)
 negative=float(e.min()/e.abs().sum())
 assert negative>-1e-10
 e=e.clamp_min(0)
 return e,negative

def toy_check():
 gen=torch.Generator().manual_seed(261036)
 rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
 C,L,R,D=rand(5,9),rand(9,7),rand(9,7),rand(7,11)
 T=torch.einsum('vk,ki,kj->vij',C,L,R@D)+torch.einsum('vk,ki,kj->vij',C,R,L@D)
 flat=T.flatten(1);g=output_gram((C,L,R,D));reference=flat@flat.T
 replay=float((g-reference).norm()/reference.norm());assert replay<1e-12
 e,_=spectrum(g);sv=torch.linalg.svdvals(flat).square()
 eigen=float((e-sv).norm()/sv.norm());assert eigen<1e-12
 # Known rank-two output: no optimizer should need more output directions.
 C2=rand(5,2)@rand(2,9);e2,_=spectrum(output_gram((C2,L,R,D)))
 tail=float(e2[2:].sum()/e2.sum());assert tail<1e-12
 return dict(gram_replay=replay,singular_energy_replay=eigen,planted_rank2_tail=tail)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('MIDPOINT_SPECTRUM_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
