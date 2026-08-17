"""Does layer 1's higher-order structure factor into independent subspaces?

User question (2026-08-17): the order-3+ dominance was measured on ONE partition
(PCA bands). If the causal structure factorises as S1 (+) S2 with interactions
confined within factors, cross-factor synergy would vanish for the right split and
the layer compresses as a product of independent pieces regardless of interaction
order. Test: two-way splits (576+576) from several families, measuring the synergy
share syn(A,B) = (full - d(A) - d(B)) / full.

Split families: PCA low/high; PCA interleaved (even/odd indices, mixes scales);
G_lam top/bottom; five random orthogonal splits.

REGISTERED PREDICTIONS:
  (a) NO tested split achieves synergy < 50% of full -- the entanglement is
      basis-robust across these families (true holism, not a wrong-coordinates
      artifact);
  (b) the five random splits cluster within +/-10 percentage points of each other
      (isotropic entanglement).
Sign check rides along: synergy positive everywhere (need-both), never negative
(mutual exclusivity) -- extending the pairwise-band sign result.
HONEST LIMIT stated in advance: this tests LINEAR partitions only; a nonlinear
factorisation (few polynomial invariants) is untested by design here."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_factorization_results.json')

@torch.no_grad()
def collect(li):
    outs=[]
    def hook(mod,inp,o): outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    Y=torch.cat(outs); return Y.mean(0), Y

def main():
    t0=time.time()
    base=held()
    Yb,Y=collect(1)
    _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
    Vp=Vh.T                                     # (D, D) PCA basis columns
    X_in=None
    # G_lam basis
    ins=[]
    def hk(mod,inp,o): ins.append(inp[0].detach().reshape(-1,D).float())
    h=m.transformer.h[1].mlp.register_forward_hook(hk)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    Xi=torch.cat(ins); S=Xi.T@Xi/Xi.shape[0]
    mlp=m.transformer.h[1].mlp
    L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float()
    G=Dw@((L@S@L.T)*(R@S@R.T))@Dw.T
    evG,UG=torch.linalg.eigh(G); UG=UG[:,evG.argsort(descending=True)]
    def val(Q):
        PATCH[1]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally: PATCH.pop(1)
    full=val(orth(Vp))          # full-space deletion via complete basis
    print(f'full-space deletion: {full:+.4f}')
    splits={}
    splits['PCA low/high']=(orth(Vp[:,:576]),orth(Vp[:,576:]))
    idx=torch.arange(D)
    splits['PCA interleaved']=(orth(Vp[:,idx%2==0]),orth(Vp[:,idx%2==1]))
    splits['G_lam top/bot']=(orth(UG[:,:576]),orth(UG[:,576:]))
    for s_ in range(5):
        g=torch.Generator(device=DEV).manual_seed(s_)
        Qr=torch.linalg.qr(torch.randn(D,D,device=DEV,generator=g))[0]
        splits[f'random-{s_}']=(Qr[:,:576],Qr[:,576:])
    out={'full':full,'splits':{}}
    print(f"  {'split':>16} {'d(A)':>8} {'d(B)':>8} {'synergy':>9} {'share':>7}")
    for tag,(A,B) in splits.items():
        dA=val(A); dB=val(B)
        syn=full-dA-dB; share=syn/full
        out['splits'][tag]={'dA':dA,'dB':dB,'synergy':syn,'share':share}
        print(f"  {tag:>16} {dA:>+8.4f} {dB:>+8.4f} {syn:>+9.4f} {100*share:>6.0f}%",
              flush=True)
    shares=[v['share'] for v in out['splits'].values()]
    rnd=[out['splits'][f'random-{s_}']['share'] for s_ in range(5)]
    pa=min(shares)>=0.50
    pb=(max(rnd)-min(rnd))<=0.10
    allpos=min(v['synergy'] for v in out['splits'].values())>0
    out['pred_a_no_factorization']=bool(pa)
    out['pred_b_isotropy']=bool(pb)
    out['all_synergies_positive']=bool(allpos)
    print(f"\n(a) no split below 50% synergy: {'HELD' if pa else 'FAILED'} "
          f"(min {100*min(shares):.0f}%)")
    print(f"(b) random splits within 10pp: {'HELD' if pb else 'FAILED'} "
          f"(spread {100*(max(rnd)-min(rnd)):.0f}pp)")
    print(f"sign: all synergies positive (need-both, no mutual exclusivity): "
          f"{'YES' if allpos else 'NO'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
