"""WEIGHT-ONLY tensor compression of the middle MLPs (mlp4-9).

The bilinear MLP is out = Down((Lx)*(Rx)) + Down_bias -- an exact third-
order tensor with no activation in the way, so compression hypotheses can
be computed FROM WEIGHTS ALONE, no data fitting. Two data-free truncations:
  (a) INPUT-SUBSPACE: project the MLP's input onto the top-r right-singular
      directions of stacked [L;R] (the directions the layer reads), then run
      the exact MLP. Control: a random orthonormal r-dim subspace.
  (b) HIDDEN CP-TRUNCATION: each hidden unit h is the rank-1 quadratic
      (l_h.x)(r_h.x)*down_h; keep the top-k units by
      ||down_h||*||l_h||*||r_h||, drop the rest.
Metric: SOLO CE cost (eval window C rows 120-300) of swapping the truncated
MLP for the real one, per layer 4..9. Zero-output cost printed as ceiling.
REGISTERED PREDICTIONS:
  (a) the weight-chosen subspace beats the random control at r=64 by >=2x
      lower cost for >=5/6 layers (the read-directions are real structure);
  (b) r=128 (11% of D) costs <=25% of r=16 for >=4/6 layers (the spectrum
      decays -- middle MLPs read a low-dimensional subspace);
  (c) CP k=1152 (25% of H) solo cost <=0.10 for >=4/6 layers."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mlp_weight_rank_results.json'
R0,R1=120,300
LAYERS=(4,5,6,7,8,9)
RANKS=(16,32,64,128)
KS=(288,1152,2304)

@torch.no_grad()
def evalCE():
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
    return float(torch.cat(ces).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base=evalCE()
    print(f'base CE {base:.4f}',flush=True)
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'base':round(base,4),'layers':{}}
    for li in LAYERS:
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        _,_,Vh=torch.linalg.svd(torch.cat([L,Rw]),full_matrices=False)
        imp=(Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1))
        order=imp.argsort(descending=True)
        rec={'zero':None,'sub':{},'rand':{},'cp':{}}
        def cost_with(fh):
            h=mlp.register_forward_hook(fh)
            c=evalCE()-base
            h.remove()
            return c
        rec['zero']=cost_with(lambda mo,i_,o_: torch.zeros_like(o_))
        print(f'L{li} zero {rec["zero"]:+.4f}',flush=True)
        for r in RANKS:
            P=Vh[:r].T.contiguous()
            def fh(mo,i_,o_,P=P):
                x=i_[0].float(); xp=(x@P)@P.T
                return (((xp@L.T)*(xp@Rw.T))@Dw.T+db).to(o_.dtype)
            rec['sub'][r]=cost_with(fh)
            Q=orth(torch.randn(D,r,device=DEV,generator=g))
            def fr(mo,i_,o_,Q=Q):
                x=i_[0].float(); xp=(x@Q)@Q.T
                return (((xp@L.T)*(xp@Rw.T))@Dw.T+db).to(o_.dtype)
            rec['rand'][r]=cost_with(fr)
            print(f'L{li} r={r:4d} sub {rec["sub"][r]:+.4f} '
                  f'rand {rec["rand"][r]:+.4f}',flush=True)
        for k in KS:
            keep=order[:k]
            Lk=L[keep].contiguous(); Rk=Rw[keep].contiguous()
            Dk=Dw[:,keep].contiguous()
            def fk(mo,i_,o_,Lk=Lk,Rk=Rk,Dk=Dk):
                x=i_[0].float()
                return (((x@Lk.T)*(x@Rk.T))@Dk.T+db).to(o_.dtype)
            rec['cp'][k]=cost_with(fk)
            print(f'L{li} k={k:4d} cp {rec["cp"][k]:+.4f}',flush=True)
        out['layers'][li]=rec
    wins=sum(1 for li in LAYERS
             if out['layers'][li]['rand'][64]
                >=2*max(out['layers'][li]['sub'][64],1e-4))
    dec=sum(1 for li in LAYERS
            if out['layers'][li]['sub'][128]
               <=0.25*max(out['layers'][li]['sub'][16],1e-4))
    cpk=sum(1 for li in LAYERS if out['layers'][li]['cp'][1152]<=0.10)
    pa=wins>=5; pb=dec>=4; pc=cpk>=4
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) sub beats rand 2x at r=64, {wins}/6: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) r=128 <=25% of r=16, {dec}/6: {'HELD' if pb else 'FAILED'}")
    print(f"(c) CP k=1152 <=0.10, {cpk}/6: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
