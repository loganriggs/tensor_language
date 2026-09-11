"""Signed full-function product Gram of a frozen projected fit, CPU only.

Reuse product_cross and prior native-product Gram logic. Not a CP border-rank proof.
pred_a energy replays<=1e-8normalized and cosinebounds; pred_b mincos<=-.95;
pred_c top16component energy share>=.5. Other concentration measures descriptive.
"""
import hashlib,io,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P
from folded_sparse_dictionary_v1 import decode
from joint_quadratic_fit_v1 import product_cross


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'PROJECTED_PRODUCT_CANCELLATION_V1_AUDIT.json';assert not out.exists()
    payload=Path('/dev/shm/bilin18_projected_sparse_dictionary_fit_v1_s0.pt').read_bytes()
    sha=hashlib.sha256(payload).hexdigest();source=Path('/dev/shm/bilin18_projected_product_cancellation_v1_source.pt')
    with source.open('xb') as f:f.write(payload)
    saved=torch.load(io.BytesIO(payload),weights_only=True,map_location='cpu');del payload
    values=saved['code_values'].double();ids=saved['code_indices'].long()
    readers,_,_,_=decode(saved['analysis_basis'].double(),ids,values,torch.ones(len(ids)))
    a,b=readers.chunk(2)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    w=torch.linalg.cholesky(metric).T@saved['down'].double()
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    energy=w.square().sum(0)*.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())
    started=time.perf_counter();n=len(a);signed_total=0.;negative=0.;positive=0.;maxcos=0.;most_negative=[];largest_negative=[]
    for start in range(0,n,256):
        stop=min(start+256,n);sl=slice(start,stop)
        gram=product_cross(a[sl],b[sl],a,b)*(w[:,sl].T@w)
        cosine=gram/(energy[sl,None]*energy[None,:]).sqrt()
        signed_total+=float(gram.sum());maxcos=max(maxcos,float(cosine.abs().max()))
        upper=torch.arange(n)[None,:]>torch.arange(start,stop)[:,None]
        neg=(-gram).clamp_min(0)*upper;pos=gram.clamp_min(0)*upper
        negative+=float(neg.sum());positive+=float(pos.sum())
        masked=cosine.masked_fill(~upper,torch.inf)
        scores,indices=masked.flatten().topk(20,largest=False)
        for score,index in zip(scores.tolist(),indices.tolist()):
            i,j=start+index//n,index%n
            most_negative.append(dict(i=i,j=j,cosine=score,energy_i=float(energy[i]/total),energy_j=float(energy[j]/total),
                                      pair_inner=float(gram[index//n,j]/total)))
        scores,indices=neg.flatten().topk(128)
        for score,index in zip(scores.tolist(),indices.tolist()):
            if score>0:largest_negative.append(dict(i=start+index//n,j=index%n,negative_inner=score/total,
                                                  cosine=float(cosine[index//n,index%n])))
    most_negative=sorted(most_negative,key=lambda r:r['cosine'])[:20]
    largest_negative=sorted(largest_negative,key=lambda r:r['negative_inner'],reverse=True)[:128]
    diagonal=float(energy.sum()/total);function=signed_total/total
    diagonal_replay=abs(diagonal-saved['details']['component_energy'])
    function_replay=abs(function-saved['details']['fitted_energy'])
    decomposition_replay=abs(function-(diagonal+2*(positive-negative)/total))
    top,indices=energy.topk(16);share=float(top.sum()/energy.sum())
    result=dict(predictions=dict(pred_a_instrument=max(diagonal_replay,function_replay,decomposition_replay)<=1e-8 and maxcos<=1+1e-8,
                 pred_b_near_opposite=most_negative[0]['cosine']<=-.95,pred_c_localized=share>=.5),
        summed_component_energy=diagonal,function_energy=function,component_to_function_ratio=diagonal/function,
        negative_pair_mass=negative/total,positive_pair_mass=positive/total,
        top128_negative_mass_share=sum(r['negative_inner'] for r in largest_negative)/(negative/total),
        top16_component_share=share,top16_products=indices.tolist(),minimum_pairs=most_negative,
        largest_negative_pairs=largest_negative,diagonal_replay=diagonal_replay,function_replay=function_replay,
        decomposition_replay=decomposition_replay,compute_seconds=time.perf_counter()-started,
        source=dict(path=str(source),sha256=sha,iteration=saved['history'][-1]['iteration'],loss=saved['loss']),
        references=['https://www.kolda.net/publication/TensorReview.pdf','https://arxiv.org/abs/math/0607647'],
        scope='Finite candidate product Gram, not native circuit identity, generic arithmetic bound, '
              'proof of diverging factors or nonexistence of a constrained optimum. No text/GPU fit.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('largest_negative_pairs','minimum_pairs')},indent=2))
    print(json.dumps(most_negative[:3],indent=2))


if __name__=='__main__':main()
