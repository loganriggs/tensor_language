"""Compare complete source-block functions, retaining private query/output maps."""
from pathlib import Path
import torch,json
from folded_producer_cubic_weights_v1 import weights as load_weights
from folded_normalized_router_v1 import rotary
from cubic_secant_block_v1 import gram_kernel
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'FOLDED_PRODUCER_BLOCK_RECURRENCE_V1_RESULT.json';assert not out.exists()
 bind=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in bind if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=load_weights(sd,C,'cpu')
 saved=[a for a in torch.load(P/'FOLDED_PRODUCER_SHARED_PARENT_V1_ARTIFACT.pt',weights_only=True) if a['kind']=='shared_linear_parent'];comps=torch.cat([a['components'] for a in saved]);mix=torch.block_diag(*[a['mix'] for a in saved]);records=[]
 for pos in (7,0):
  r=rotary(8,128).T@rotary(pos,128);weights=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o);g,k=gram_kernel(comps,mix,weights);g0,g1,gh=g[:16,:16],g[16:,16:],g[:16,16:];i0,i1=torch.linalg.inv(g0),torch.linalg.inv(g1);k0,k1c,kh=k[:,:16,:16],k[:,16:,16:],k[:,:16,16:]
  e0=(g0[-2:,-2:]*(i0[None]@k0@i0[None])[:,-2:,-2:]).sum((1,2));e1=(g1[-2:,-2:]*(i1[None]@k1c@i1[None])[:,-2:,-2:]).sum((1,2));cross=(gh[-2:,-2:]*(i0[None]@kh@i1[None])[:,-2:,-2:]).sum((1,2));cos=float(cross.sum()/(e0.sum()*e1.sum()).sqrt());err=float(((e0.sum()+e1.sum()-2*cross.sum()).clamp_min(0)/((e0.sum()+e1.sum())/2)).sqrt())
  l0=torch.linalg.cholesky(g0[-2:,-2:]);l1=torch.linalg.cholesky(g1[-2:,-2:]);wh=torch.linalg.solve_triangular(l0,gh[-2:,-2:],upper=False);wh=torch.linalg.solve_triangular(l1,wh.T,upper=False).T
  records.append(dict(position=pos,block_function_cosine=cos,symmetric_relative_function_error=err,source_subspace_cosines=torch.linalg.svdvals(wh).tolist(),block_layer_energies=[e.reshape(3,9).sum(1).tolist() for e in (e0,e1)],head_energies=[e.tolist() for e in (e0,e1)]))
 result=dict(records=records,scope='Complete two-source-function blocks with original 16-feature private query solves. Head labels/reference gates retained. Frozen coefficient comparison, not native behavior or same activation states across layers.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
