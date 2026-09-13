"""Select native scalar products and freely refit all output connections.

Target J10 B9, not the earlier independent-input K(Jz,y) dictionary.
Pred_a <=10% coefficient error and >=10% local scalar saving in some arm.
Pred_b normal equations <=1e-10 and full-dictionary replay <=1e-8.
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
    j=prepare_global(w,l10,r10,d10,compile_mixed=True)['mixed_map'];c=j@d
    h=torch.empty((4608,4608),dtype=torch.float64)
    for a in range(0,4608,256):
        b=min(a+256,4608)
        h[a:b]=.5*((l[a:b]@l.T)*(r[a:b]@r.T)+(l[a:b]@r.T)*(r[a:b]@l.T))
    rhs=h@c.T;energy=(c.T*rhs).sum();baseline=3*1152*4608+1152**2
    rows=[]
    for name,writes in [('upstream',d),('composed',c)]:
        score=writes.square().sum(0)*h.diag();order=score.argsort(descending=True)
        for size in (1024,2048,3072,4096):
            ids=order[:size];hs=h[ids][:,ids];b=rhs[ids]
            fit=torch.cholesky_solve(b,torch.linalg.cholesky(hs))
            residual=(energy-(fit*b).sum()).clamp_min(0)
            rows.append(dict(selection=name,products=size,relative_error=float((residual/energy).sqrt()),
                normal_equation_error=float((hs@fit-b).norm()/b.norm()),
                stored_scalars=3*1152*size,scalar_saving=1-3*1152*size/baseline))
    full=torch.cholesky_solve(rhs,torch.linalg.cholesky(h))
    replay=float((full-c.T).norm()/c.norm())
    result=dict(pred_a=any(x['relative_error']<=.1 and x['scalar_saving']>=.1 for x in rows),
        pred_b=replay<=1e-8 and all(x['normal_equation_error']<=1e-10 for x in rows),
        full_dictionary_writer_replay=replay,baseline_scalars=baseline,rows=rows,
        seconds=time.perf_counter()-start,
        scope='Exact least squares at norm-ranked supports, free output vectors, frozen native input readers. No global support optimum, learned readers, normalization, text fit, behavioral validation or adopted executor. Local price assumes standalone J10 B9; if native B9 is independently needed, joint graph accounting differs.')
    if '--bound' in sys.argv:
        norm=h.diag().sqrt()
        normalized=h/norm[:,None]/norm[None,:]
        smallest=float(torch.linalg.eigvalsh(normalized)[0])
        energies=(c.square().sum(0)*h.diag()).sort(descending=True).values
        result['normalized_dictionary_min_eigenvalue']=smallest
        result['any_support_relative_error_lower_bounds']={str(k):float((smallest*energies[k:].sum()/energy).sqrt()) for k in (1024,2048,3072,4096)}
        result['seconds']=time.perf_counter()-start
    (p/('COMPOSED_PRODUCT_DICTIONARY_V1_BOUND.json' if '--bound' in sys.argv else 'COMPOSED_PRODUCT_DICTIONARY_V1_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
