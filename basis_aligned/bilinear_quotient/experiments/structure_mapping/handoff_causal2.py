"""CAUSAL HANDOFF TEST v2 -- v1 subtracted the raw attention output from
the MLP's ALREADY-NORMALIZED input: wrong interface, wrong scale
(costs ~10 nats everywhere, control included -- instrument void, ledger
rule: reconstruction-scale sanity first). The block computes
mlp(rms_norm(x_mixed + attn_out)); the correct no-handoff counterfactual
is mlp(rms_norm(x_mixed)). v2 does exactly that, with a norm-matched
random perturbation control (blocks 2, 8, 14) whose magnitude equals the
real change vector at the same interface.
REGISTERED PREDICTIONS (same as v1):
  (a) causal handoff cost correlates with the wiring-map diagonal
      (Spearman >= 0.5 over 18 blocks);
  (b) front blocks (0-8) mean cost >= 2x tail (9-17);
  (c) random control <= half the real cost at blocks 2 and 8."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'handoff_causal2_results.json'
R0,R1=120,300

@torch.no_grad()
def evalCE(hooks):
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    for h in hooks: h.remove()
    return float(torch.cat(ces).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base=evalCE([])
    mp=json.load(open(PT+'block_motif_results.json'))['map']
    diag=[mp[i][i] for i in range(18)]
    g=torch.Generator(device=DEV).manual_seed(0)
    cur={}
    def mk_pair(li,rand=False):
        blk=m.transformer.h[li]
        def hb(mo,args):
            x,v1,x0=args
            cur['xm']=(blk.lambdas[0]*x+blk.lambdas[1]*x0).detach()
        h1=blk.register_forward_pre_hook(hb)
        def hm(mo,args):
            x=args[0]
            alt=F.rms_norm(cur['xm'].float(),(D,))
            if not rand:
                return (alt.to(x.dtype),)+args[1:]
            d=x.float()-alt
            r=torch.randn(d.shape,device=DEV,generator=g)
            r=r*(d.reshape(-1).norm()/r.reshape(-1).norm())
            return ((x.float()-r).to(x.dtype),)+args[1:]
        h2=blk.mlp.register_forward_pre_hook(hm)
        return [h1,h2]
    costs=[]
    for li in range(18):
        c=evalCE(mk_pair(li))-base
        costs.append(c)
        print(f'block {li:2d}: handoff cut {c:+.4f} (map diag '
              f'{diag[li]:.3f})',flush=True)
    ctl={}
    for li in (2,8,14):
        ctl[li]=evalCE(mk_pair(li,rand=True))-base
        print(f'block {li:2d}: random-subtract control {ctl[li]:+.4f}',
              flush=True)
    def rank(v):
        s=sorted(range(len(v)),key=lambda i:v[i])
        r=[0]*len(v)
        for j,i in enumerate(s): r[i]=j
        return r
    ra,rb=rank(costs),rank(diag)
    n=18
    rho=1-6*sum((ra[i]-rb[i])**2 for i in range(n))/(n*(n*n-1))
    fr=sum(costs[:9])/9; tl=sum(costs[9:])/9
    pa=rho>=0.5; pb=fr>=2*max(tl,1e-4)
    pc=all(ctl[li]<=0.5*max(costs[li],1e-4) for li in (2,8))
    out={'base':round(base,4),'costs':[round(c,4) for c in costs],
         'controls':{li:round(v,4) for li,v in ctl.items()},
         'spearman':round(rho,3),'front_mean':round(fr,4),
         'tail_mean':round(tl,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'spearman {rho:.3f} | front {fr:+.4f} tail {tl:+.4f}')
    print(f"(a) spearman >= 0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) front >= 2x tail: {'HELD' if pb else 'FAILED'}")
    print(f"(c) random control <= half at 2,8: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
