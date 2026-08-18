"""CAUSAL HANDOFF TEST -- 285's wiring map says the attn->own-mlp
handoff is a front-loaded motif (subspace coupling 2x cross-block in
early layers, floor-level in the tail). Causal check: for each block i,
let mlp_i alone see the stream WITHOUT attn_i's fresh contribution
(subtract attn_i's output from mlp_i's input only; the rest of the
network is untouched) and measure CE cost. Control: subtract a
norm-matched random vector instead (3 blocks: 2, 8, 14).
REGISTERED PREDICTIONS:
  (a) causal handoff cost correlates with the map's diagonal coupling
      across the 18 blocks (Spearman >= 0.5);
  (b) front blocks (0-8) mean cost >= 2x tail blocks (9-17);
  (c) random-subtraction control <= half the real cost at blocks 2, 8."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'handoff_causal_results.json'
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
        at=m.transformer.h[li].attn
        def ha(mo,i_,o_):
            y=o_[0] if isinstance(o_,tuple) else o_
            cur['a']=y.detach()
        h1=at.register_forward_hook(ha)
        def hm(mo,args):
            x=args[0]
            a=cur['a']
            if rand:
                r=torch.randn(a.shape,device=DEV,generator=g)
                a=r*(a.norm()/r.norm())
            return (x-a.to(x.dtype),)+args[1:]
        h2=m.transformer.h[li].mlp.register_forward_pre_hook(hm)
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
