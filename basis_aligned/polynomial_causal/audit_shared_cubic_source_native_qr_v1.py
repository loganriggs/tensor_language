"""CPU audit of projected source energy overlap, with no fitting."""
import json,time,sys
from shared_cubic_source_projection_v1 import capture as gram_capture
from shared_cubic_source_qr_v1 import capture as qr_capture
from pathlib import Path
import torch
from folded_normalized_router_v1 import rotary
from sparse_path_stability_atlas_v1 import digest
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram
from source_energy_overlap_v1 import overlap,operators,literal_shared
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);tic=time.perf_counter()
    binding=json.loads((P/'SHARED_CUBIC_SOURCE_NATIVE_V1_BINDING.json').read_text())['files']
    sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def w(k):return sd[k].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mix=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-mix)*w('transformer.h.17.attn.c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1))
    o=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128)*g0[None,:,None];qr=rotary(8,128);weights={}
    for pos in (7,0):
        rotation=qr.T@rotary(pos,128);ka=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kb=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1)
        weights[pos]=(q1,ka,q2,kb,v,o)
    artifact=P/'SHARED_CUBIC_SOURCE_CONTINUE_V1_ARTIFACT.pt'
    atoms=torch.load(artifact,weights_only=True,map_location='cpu')['atoms'][1]
    evaluations={};gradients={}
    for name,fn in [('gram',gram_capture),('qr',qr_capture)]:
        start=time.perf_counter();x=atoms.clone().requires_grad_(True);v=fn(x,*weights[7]);g=torch.autograd.grad(v,x)[0];g=g-(g*atoms).sum(-1,keepdim=True)*atoms
        gradients[name]=g; evaluations[name]=dict(capture=float(v.detach()),tangent_gradient=float(g.norm()),seconds=time.perf_counter()-start)
    direction=gradients['qr']/gradients['qr'].norm();fd=[]
    for eps in (1e-3,1e-4,1e-5):
        row=dict(epsilon=eps)
        for name,fn in [('gram',gram_capture),('qr',qr_capture)]:
            with torch.no_grad():numeric=float((fn(atoms+eps*direction,*weights[7])-fn(atoms-eps*direction,*weights[7]))/(2*eps))
            analytic=float((gradients[name]*direction).sum());row[name]=dict(numeric=numeric,analytic=analytic,relative_error=abs(numeric-analytic)/max(abs(analytic),1e-30))
        fd.append(row)
    result=dict(artifact_sha=digest(artifact),evaluations=evaluations,gradient_relative_difference=float((gradients['gram']-gradients['qr']).norm()/gradients['qr'].norm()),finite_differences=fd,seconds=time.perf_counter()-tic)
    out=P/'SHARED_CUBIC_SOURCE_NATIVE_QR_V1_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
