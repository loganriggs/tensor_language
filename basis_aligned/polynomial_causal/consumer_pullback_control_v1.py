"""Gauge and independent sample controls of the frozen analytic consumer metric."""
from pathlib import Path
import json,time,torch
from consumer_pullback_metric_v1 import query_gram,fourth_moment,metric_from,weighted_component
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'CONSUMER_PULLBACK_V1_CONTROL.json';assert not out.exists()
 p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');C=p['current_readers'];tokens=p['token_reads'];T=fourth_moment(C,tokens)
 Bsum=torch.zeros(2,2,dtype=torch.float64)
 for pos in range(32):Bsum+=query_gram(p,rotary(31,128).T@rotary(pos,128))[0]/32
 H=metric_from(Bsum,T);H=H/H.trace()
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 scale=1.
 for j in range(14,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
 mix=float(state['transformer.h.13.attn.lamb']);fold=(scale*C@state['transformer.h.13.attn.c_proj.weight'].double()).reshape(4,9,128)[:,0]
 values=torch.cat([(1-mix)*state['transformer.h.13.attn.c_v.weight'].double().reshape(9,128,1152)[0],mix*state['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)[0]],1);M=fold@values
 selected,projector,reader,singular=weighted_component(M,H)
 gauges=[]
 for axis in range(4):
  d=torch.ones(4,dtype=torch.float64);d[axis]=100.
  factor=d[0]*d[1]*d[2:];gB=Bsum/factor[:,None]/factor[None,:]
  gH=metric_from(gB,fourth_moment(C*d[:,None],tokens*d));originalH=metric_from(Bsum,T)
  gselected,_,gr,_=weighted_component(M*d[:,None],gH)
  gauges.append(dict(axis=axis,metric_covariance_error=float((gH*d[:,None]*d[None,:]-originalH).norm()/originalH.norm()),source_direction_cosine=float((gr@reader).abs()),selected_map_error=float((gselected/d[:,None]-selected).norm()/selected.norm())))
 # Independent Gaussian + discrete-token samples validate exact fourth moments.
 gen=torch.Generator().manual_seed(92341);cov=C@C.T;root=torch.linalg.cholesky(cov)
 sample=torch.randn(131072,4,dtype=torch.float64,generator=gen)@root.T+tokens[torch.randint(len(tokens),(131072,),generator=gen)]
 pairs=torch.einsum('ni,nj->nij',sample,sample).flatten(1);estimate=(pairs.T@pairs/len(sample)).reshape(4,4,4,4)
 moment_error=float((estimate-T).norm()/T.norm())
 # Independently sample native query numerator consumers for one fixed position.
 B,(ell,right,writers,coeff)=query_gram(p,rotary(8,128).T@rotary(7,128));mc=torch.zeros(2,2,dtype=torch.float64)
 for batch in range(32):
  q=torch.randn(256,1152,dtype=torch.float64,generator=gen);products=(q@ell.T)*(q@right.T)
  beta=torch.einsum('nr,ar,ro->nao',products,coeff,writers);mc+=torch.einsum('nao,nbo->ab',beta,beta)/8192
 query_error=float((mc-B).norm()/B.norm())
 controls=max(g['metric_covariance_error'] for g in gauges)<=1e-10 and max(g['selected_map_error'] for g in gauges)<=1e-9 and moment_error<=.05 and query_error<=.1
 torch.save(dict(metric=H,projector=projector,selected_value=selected,source_reader=reader,value_matrix=M,singular_values=singular),P/'CONSUMER_PULLBACK_V1_ARTIFACT.pt')
 result=dict(pred_a=controls,gauges=gauges,fourth_moment_monte_carlo_error=moment_error,query_gram_monte_carlo_error=query_error,metric_eigenvalues=torch.linalg.eigvalsh(H).tolist(),metric=H.tolist(),weighted_rank1_fraction=float(singular[0].square()/singular.square().sum()),euclidean_source_cosine=float((reader@torch.linalg.svd(M,full_matrices=False)[2][0]).abs()),projector_idempotence=float((projector@projector-projector).norm()/projector.norm()),seconds=time.perf_counter()-tic,scope='Exact polynomial consumer derivative metric under independent isotropic query/current and uniform full-vocabulary initialization. Average32relativepositions. Native QK/RMS gates and correlations omitted in discovery metric; native validation retains them. Weighted rank1 solve is global for this fixed one-sided metric only.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
