"""Independent objective/gradient and exact secant audit of failed producer fits."""
from pathlib import Path
import torch,json,time
from folded_producer_cubic_weights_v1 import weights as load_weights
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import atom_gram,capture
from shared_cubic_source_qr_v1 import capture as qr_capture
from cubic_secant_block_v1 import encode,gram_kernel,align_pair
P=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'FOLDED_PRODUCER_CUBIC_V1_AUDIT.json';assert not out.exists()
 bind=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in bind if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 readers=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=load_weights(sd,readers,'cpu');r=rotary(8,128).T@rotary(7,128)
 weights=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
 saved=torch.load(P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_ARTIFACT.pt',weights_only=True);records=[];packed=[]
 for arm,atoms in enumerate(saved['atoms']):
  g=atom_gram(atoms);corr=g/(g.diagonal()[:,None]*g.diagonal()[None,:]).sqrt();ii,jj=torch.triu_indices(16,16,offset=1);order=corr[ii,jj].abs().argsort(descending=True);pairs=[(int(ii[t]),int(jj[t])) for t in order[:5]];pair=pairs[0]
  evaluations={};gradients={}
  for name,fn in [('gram',capture),('qr',qr_capture)]:
   t=time.perf_counter();x=atoms.clone().requires_grad_(True);cap=fn(x,*weights);grad=torch.autograd.grad(cap,x)[0];grad=grad-(grad*atoms).sum(-1,keepdim=True)*atoms;gradients[name]=grad;evaluations[name]=dict(capture=float(cap.detach()),gradient_norm=float(grad.norm()),seconds=time.perf_counter()-t)
  aligned=align_pair(atoms[pair[0]],atoms[pair[1]]);components,mix,info=encode(atoms,pair);packed.append(dict(components=components,mix=mix,pair=pair,**info))
  with torch.no_grad():
   gg,kk=gram_kernel(components,mix,weights);newcap=float(torch.linalg.solve(gg,kk.sum(0)).trace());inv=torch.linalg.inv(gg);energies=(inv[None]@kk@inv[None]).diagonal(dim1=-2,dim2=-1)*gg.diagonal()[None]
  records.append(dict(arm=arm,top_pairs=[dict(pair=z,coefficient_cosine=float(corr[z[0],z[1]])) for z in pairs],reader_cosines=(atoms[pair[0]]*aligned).sum(-1).tolist(),evaluations=evaluations,gradient_relative_difference=float((gradients['gram']-gradients['qr']).norm()/gradients['qr'].norm()),secant=dict(separation=info['separation'],capture=newcap,relative_capture_change=(newcap-evaluations['qr']['capture'])/evaluations['qr']['capture'],gram_condition=float(torch.linalg.cond(gg)),cancellation_ratio=float(energies.sum()/newcap))))
 result=dict(records=records,seconds=time.perf_counter()-tic,scope='Diagnostic of frozen failed endpoints. QR independent source solve; exact secant changes coordinates, not function span or circuit behavior. No native fit continuation or convergence repair claimed.')
 torch.save(packed,P/'FOLDED_PRODUCER_CUBIC_SECANT_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
