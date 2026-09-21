import torch
from quadratic_pair_blocks import compile_pair
from pairwise_reader_graph import GROUPS

def export(states,scales,template,transform,inverse):
 out=dict(input_bases=template['input_bases'],pairs={});diagnostics=[]
 for j,(a,b) in enumerate(GROUPS):
  P,V,E,C,A=[x.detach().cpu() for x in states[j]];scale=scales[j].detach().cpu();common=(A-E@C@E.T)*scale;private=C*scale
  cs=compile_pair(common[0],common[1]);cp=compile_pair(private[0],private[1]);shared=torch.cat([template['input_bases'][str(a)],template['input_bases'][str(b)]],1);R=P.T@transform@shared;r=P.shape[1];p=V.shape[1]
  block=dict(template['pairs'][str(j)]);block.update(shared_map=torch.linalg.solve(R,cs['input_transform']),private_reader=inverse@(V+P@E)@cp['input_transform'],product_indices=torch.cat([cs['product_indices'],cp['product_indices']+torch.tensor([[r],[r],[0]],dtype=torch.int64)],1),product_weights=torch.cat([cs['product_weights'],cp['product_weights']]),shared_indices=torch.arange(r),private_indices=torch.arange(r,r+p));out['pairs'][str(j)]=block;diagnostics.append(dict(common=cs['diagnostics'],private=cp['diagnostics'],coupling_norm=float(E.norm())))
 return out,diagnostics
