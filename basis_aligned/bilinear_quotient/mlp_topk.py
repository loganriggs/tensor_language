"""MLP TOP-K -- extend the k-dial to the MLP side. Each bilinear MLP
is a sum of 4608 rank-1 quadratics: h_i = (L_i x)(R_i x), out =
Down h + b. The readable-code analog of 4-read attention: per
position, keep only the k largest-|h| quadratic units. How sparse is
the active set? (This is the census's 'half of each middle layer
suffices' result made per-position and per-unit.)
REGISTERED PREDICTIONS (mid MLPs m4-m9 substituted jointly):
  (a) k=512 of 4608 (11%) costs <= +0.10 on the census grid;
  (b) curve k in {32,128,512,1024} reported; random-unit control at
      matched k costs >= 3x at k=512;
  (c) FRESH leg at the best k within bar: <=1.5x grid cost."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp_topk_results.json'
MIDS=[4,5,6,7,8,9]

@torch.no_grad()
def main():
    t0=time.time()
    def hooks(k,random_units=False):
        hs=[]
        for li in MIDS:
            mlp=m.transformer.h[li].mlp
            def fh(mo,i_,o_,mlp=mlp,k=k,random_units=random_units):
                x=i_[0]
                h=mlp.Left(x)*mlp.Right(x)
                if random_units:
                    g=torch.Generator().manual_seed(4)
                    idx=torch.randperm(h.shape[-1],generator=g)[:k] \
                        .to(DEV)
                    msk=torch.zeros(h.shape[-1],device=DEV)
                    msk[idx]=1.0
                    h=h*msk
                else:
                    _,idx=h.abs().topk(k,dim=-1)
                    msk=torch.zeros_like(h).scatter(-1,idx,1.0)
                    h=h*msk
                return mlp.Down(h)+mlp.Down_bias
            hs.append(mlp.register_forward_hook(fh))
        return hs
    base=cl.ce_sweep([])
    res={}
    for k in (32,128,512,1024):
        d=cl.ce_sweep(hooks(k))
        res[f'k{k}']=round(float((d-base).mean()),4)
        print(f'k={k}: {res[f"k{k}"]:+.4f}',flush=True)
    dr=cl.ce_sweep(hooks(512,random_units=True))
    res['rand512']=round(float((dr-base).mean()),4)
    print(f'random-512: {res["rand512"]:+.4f}',flush=True)
    pa=res['k512']<=0.10
    pb=res['rand512']>=3*max(res['k512'],1e-3)
    bestk=None
    for k in (32,128,512,1024):
        if res[f'k{k}']<=0.10: bestk=k; break
    fr=None
    if bestk:
        FRESH=cl.fresh_rows(120)
        bF=cl.ce_sweep([],tok=FRESH)
        cF=cl.ce_sweep(hooks(bestk),tok=FRESH)
        fr=round(float((cF-bF).mean()),4)
    pc_=fr is not None and fr<=1.5*max(res[f'k{bestk}'],1e-3)
    out=dict(res); out.update({'bestk':bestk,'fresh':fr,
        'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_)})
    print(f'bestk {bestk} fresh {fr}')
    print(f"(a) k=512 <=+0.10: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random >=3x: {'HELD' if pb else 'FAILED'}")
    print(f"(c) fresh <=1.5x: {'HELD' if pc_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
