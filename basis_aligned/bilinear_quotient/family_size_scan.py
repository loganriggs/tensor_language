"""2x2 test of section 203's size-scaling prediction with on-hand models:
rank-0 (constant) licensing across the family. bilin18 (18L) licensed 4
constants at <=0.05; bilin12 (12L) licensed zero. Predictions for the other
two: swiglu18 (18L, gated) should license like the other 18L model; sqrd12
(12L, conventional) should not, like the other 12L model.

REGISTERED PREDICTIONS: (a) swiglu18 licenses >= 2 tail layers (rank-0 cost
<= 0.05, layers 5-15); (b) sqrd12 licenses <= 1 (layers 3-10); jointly the 2x2
supports size-over-architecture; a crossed outcome refutes it."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_size_scan_results.json')

@torch.no_grad()
def scan(name, layers):
    m2,_=load_elriggs(name, device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def ce(assign):
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            idx=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for lj,blk in enumerate(m2.transformer.h):
                x=blk.lambdas[0]*x+blk.lambdas[1]*x0
                x1,v1=blk.attn(F.rms_norm(x,(D,)),v1)
                x=x+x1
                if lj in assign:
                    mo=assign[lj][None,None,:].to(x.dtype).expand_as(x)
                else:
                    mo=blk.mlp(F.rms_norm(x,(D,)))
                x=x+mo
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg)
            tot+=float(c)*tg.numel(); n+=tg.numel()
        return tot/n
    # means from stats rows
    mus={}
    caps={li:[] for li in layers}
    hs=[]
    for li in layers:
        def mk(li=li):
            return lambda mod,i_,o_: caps[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    base=ce({})
    lic=[]
    costs={}
    for li in layers:
        mu=torch.cat(caps[li]).mean(0)
        c=ce({li:mu})-base
        costs[li]=c
        if c<=0.05: lic.append(li)
        print(f'{name} L{li}: rank-0 +{c:.4f}',flush=True)
    del m2; torch.cuda.empty_cache()
    return lic,costs

@torch.no_grad()
def main():
    t0=time.time()
    lic_sw,c_sw=scan('swiglu18', list(range(5,16)))
    lic_sq,c_sq=scan('sqrd12', list(range(3,11)))
    pa=len(lic_sw)>=2
    pb=len(lic_sq)<=1
    out={'swiglu18_licensed':lic_sw,'sqrd12_licensed':lic_sq,
         'swiglu18_costs':{str(k):v for k,v in c_sw.items()},
         'sqrd12_costs':{str(k):v for k,v in c_sq.items()},
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\nswiglu18 licensed {lic_sw} | sqrd12 licensed {lic_sq}")
    print(f"(a) 18L licenses >=2: {'HELD' if pa else 'FAILED'}")
    print(f"(b) 12L licenses <=1: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
