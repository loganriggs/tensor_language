"""Exact output-mode spectra of the producer-folded private correction.

No normalization, affine constant or text weighting in the quadratic numerator.
Pred_a rank64 composition error <=.1; pred_b Gram slice oracle <=1e-10.
Output-rank optimum is exact for this coefficient norm, not global graph rank.
"""
import json,time,sys
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from fixed_writer_products_v1 import prepare_global

@torch.no_grad()
def main():
    torch.set_num_threads(2);start=time.perf_counter();p=Path(__file__).resolve().parent
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    def maps(layer):return [sd[f'transformer.h.{layer}.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    l,r,d=maps(9);l10,r10,d10=maps(10)
    program=torch.load(p/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt',weights_only=True)
    w=program['direction'].double()*float(sd['transformer.h.10.lambdas'][0])
    g=prepare_global(w,l10,r10,d10,compile_mixed=True);j=g['mixed_map']
    # Gram of symmetrized hidden quadratic forms; no d*d*m tensor.
    h=torch.empty((4608,4608),dtype=torch.float64)
    for a in range(0,4608,256):
        b=min(a+256,4608)
        h[a:b]=.5*((l[a:b]@l.T)*(r[a:b]@r.T)+(l[a:b]@r.T)*(r[a:b]@l.T))
    # Small independent explicit coefficient oracle uses truncated input axes.
    ls=l[:7,:5];rs=r[:7,:5]
    q=.5*(ls[:,:,None]*rs[:,None,:]+rs[:,:,None]*ls[:,None,:])
    direct=q.flatten(1)@q.flatten(1).T
    implicit=.5*((ls@ls.T)*(rs@rs.T)+(ls@rs.T)*(rs@ls.T))
    oracle=float((direct-implicit).norm()/direct.norm())
    base=d@h@d.T;folded=j@base@j.T
    def spectrum(gram):
        vals=torch.linalg.eigvalsh((gram+gram.T)/2).flip(0)
        negative=float(vals.min());vals=vals.clamp_min(0);total=vals.sum()
        errs={str(k):float((vals[k:].sum()/total).sqrt()) for k in (16,32,64,128,256,512,768,1024)}
        ranks={str(e):next(k for k in range(1153) if vals[k:].sum()<=e*e*total) for e in (.1,.05,.01)}
        return dict(relative_errors=errs,necessary_output_ranks=ranks,min_raw_eigenvalue=negative)
    results={name:spectrum(gram) for name,gram in [('upstream_quadratic',base),('linear_reader',j@j.T),('composed_quadratic',folded)]}
    if '--radial' in sys.argv:
        trace=j@(d@(l*r).sum(-1))
        outer=trace[:,None]*trace[None,:]
        results['gaussian_second_moment']=spectrum(2*folded+outer)
        results['traceless_coefficient']=spectrum(folded-outer/1152)
    result=dict(pred_a=results['composed_quadratic']['relative_errors']['64']<=.1,pred_b=oracle<1e-10,
        spectra=results,oracle_error=oracle,seconds=time.perf_counter()-start,
        scope=__doc__,prices='Native J10 is 1152 squared scalars; a rank-r factored J10 costs 2304r. Output rank of the composed tensor alone does not price input computation or an executor.')
    (p/('COMPOSED_PRIVATE_OUTPUT_RADIAL_V1_RESULT.json' if '--radial' in sys.argv else 'COMPOSED_PRIVATE_OUTPUT_SPECTRUM_V1_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
