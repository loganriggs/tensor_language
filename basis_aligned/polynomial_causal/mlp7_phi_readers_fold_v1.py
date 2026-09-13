"""Exact upstream fold for four frozen MLP8 input readers; no fitted factors."""
from pathlib import Path
import json,time,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(24001);tic=time.perf_counter()
 gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);u=gen['eigenvectors'][:,:4];eig=gen['eigenvalues'][:4]
 files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in files if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True)
 base='transformer.h.7.mlp.';L=sd[base+'Left.weight'].double();R=sd[base+'Right.weight'].double();D=sd[base+'Down.weight'].double();bias=sd[base+'Down_bias'].double();C=u.T@D
 matrices=torch.stack([(L.T*C[i])@R for i in range(4)]);matrices=(matrices+matrices.transpose(-1,-2))/2
 x=torch.randn(32,1152,dtype=torch.float64);x=x/x.square().mean(-1,keepdim=True).sqrt();direct=((x@L.T)*(x@R.T))@C.T;fold=torch.einsum('bi,mij,bj->bm',x,matrices,x)
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));scale=direct.std();b=torch.randn_like(direct)*scale;h=torch.randn_like(direct)*scale;q=direct
 paths=torch.stack([(b.square()*eig).sum(-1),(q.square()*eig).sum(-1),(h.square()*eig).sum(-1),2*(b*q*eig).sum(-1),2*(b*h*eig).sum(-1),2*(q*h*eig).sum(-1)],-1);target=((b+q+h).square()*eig).sum(-1)
 program=dict(readers=u,eigenvalues=eig,product_coefficients=C,bias_reads=bias@u,lambda8=sd['transformer.h.8.lambdas'].double())
 out=dict(fold_error=rel(fold,direct),six_path_error=rel(paths.sum(-1),target),reader_shape=list(u.shape),product_coefficients_shape=list(C.shape),symmetric_quadratic_shape=list(matrices.shape),stored_program_scalars=sum(t.numel() for t in program.values()),external_mlp7_input_weight_scalars=L.numel()+R.numel(),seconds=time.perf_counter()-tic,scope='Exact coefficient fold and random-input algebra control. Six paths B2/Q2/H2/2BQ/2BH/2QH; native input/norm/routing generation still external. No native circuit attribution or matrix-rank compression claim.')
 assert out['fold_error']<1e-12 and out['six_path_error']<1e-12
 torch.save(program,P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt');(P/'MLP7_PHI_READERS_FOLD_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
