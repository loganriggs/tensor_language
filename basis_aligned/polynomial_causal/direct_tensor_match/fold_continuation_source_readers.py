"""Exact upstream MLP16 folding for the native rank-one observer's scalar readers.

For normalized MLP16 input z, source m0=lambda17[0]*D16[(L16z)*(R16z)].
a.m0=zTQa z and b.m0=zTQb z. RMS denominator s(h) remains explicit downstream.
Report isotropic coefficient spectral cost only; no behavioral compression claim.
"""
from pathlib import Path
import json,torch,time
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
out=p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.json';assert not out.exists()
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
s=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');e=torch.load(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True);L=s['transformer.h.16.mlp.Left.weight'].double();R=s['transformer.h.16.mlp.Right.weight'].double();D=s['transformer.h.16.mlp.Down.weight'].double();lam=s['transformer.h.17.lambdas'].double()[0]
gen=torch.Generator().manual_seed(261356);z=torch.randn(64,1152,dtype=torch.float64,generator=gen);z/=z.square().mean(1,keepdim=True).sqrt();source=lam*((z@L.T)*(z@R.T))@D.T
records=[];exports={}
for name,reader in [('a',e['A'][:,0]),('b',e['B'][:,0])]:
 channel=lam*(D.T@reader);raw=L.T@(channel[:,None]*R);Q=(raw+raw.T)/2
 direct=source@reader;folded=((z@Q)*z).sum(1);replay=float((folded-direct).norm()/direct.norm());assert replay<1e-10
 eig,V=torch.linalg.eigh(Q);order=eig.abs().argsort(descending=True);eig=eig[order];V=V[:,order];energy=eig.square();cum=energy.cumsum(0)/energy.sum();ranks={str(f):int(torch.searchsorted(cum,torch.tensor(f,dtype=cum.dtype)))+1 for f in [.9,.99,.999]}
 records.append(dict(reader=name,replay=replay,rank_for_coefficient_energy=ranks,truncation=[dict(rank=k,coefficient_relative_error=float(energy[k:].sum().sqrt()/energy.sum().sqrt())) for k in [1,4,16,64,256]],positive_eigenvalues=int((eig>0).sum()),negative_eigenvalues=int((eig<0).sum())))
 exports[name]=dict(matrix=Q,eigenvalues=eig,eigenvectors=V,reader=reader)
torch.save(exports,p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt');result=dict(records=records,lambda17=float(lam),seconds=time.perf_counter()-start,scope='Exact fold of two scalar reads through preceding bilinear MLP output, excluding Down_bias consistently. z is normalized MLP16 input; its producing normalization and final recipient RMS remain explicit. Isotropic coefficient Frobenius ranks, not Gaussian functional loss or data-informed/native accuracy. h remains an external native input and itself contains upstream computations; no whole-model extraction claim.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
