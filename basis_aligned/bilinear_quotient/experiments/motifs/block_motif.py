"""CIRCUIT MOTIFS AT BLOCK LEVEL -- user direction: are there repeated
CONNECTED computations, e.g. "attention writes, the adjacent MLP reads
it" as a canonical composed unit repeated down the depth? Method: an
18x18 subspace coupling map. For each attention layer i, the WRITE
subspace W_i = top-64 eigenvectors of attn_i's output covariance on
window A (behavioral). For each MLP j, the READ subspace V_j = top-64
right-singular vectors of stacked [Left;Right] weights (weights-only).
coupling(i,j) = ||projection of V_j onto W_i||^2 / 64 -- the fraction of
mlp_j's read directions that attn_i's writes cover. Random-subspace
floor: 64/1152 = 0.056.
REGISTERED PREDICTIONS:
  (a) SAME-BLOCK MOTIF: coupling(i,i) >= 1.3x the mean cross-block
      coupling for >= 12 of 18 blocks (the attn->own-mlp handoff is a
      repeated wiring motif);
  (b) positive control: attn0's row (the v1 lexical broadcast) has mean
      coupling >= the global cross-block mean (known broadcast);
  (c) NEXT-BLOCK vs SAME-BLOCK: report whether coupling(i,i+1) beats
      coupling(i,i) on average (which handoff is the real motif);
  (d) full 18x18 map reported (this is the model's wiring diagram at
      subspace level)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV, orth
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'block_motif_results.json'
CA=300; NB=10

@torch.no_grad()
def main():
    t0=time.time()
    covs={li:torch.zeros(D,D,device=DEV) for li in range(18)}
    hs=[]
    for li in range(18):
        def mk(li=li):
            def h(mo,i_,o_):
                y=(o_[0] if isinstance(o_,tuple) else o_)
                y=y.detach().float().reshape(-1,D)
                covs[li]+=y.T@y
            return h
        hs.append(m.transformer.h[li].attn.register_forward_hook(mk()))
    for i in range(CA,CA+NB*4,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    W={}
    for li in range(18):
        ev,evec=torch.linalg.eigh(covs[li])
        W[li]=evec[:,-64:].contiguous()
    V={}
    for lj in range(18):
        mlp=m.transformer.h[lj].mlp
        _,_,Vh=torch.linalg.svd(
            torch.cat([mlp.Left.weight.detach().float(),
                       mlp.Right.weight.detach().float()]),
            full_matrices=False)
        V[lj]=Vh[:64].T.contiguous()
    C=torch.zeros(18,18)
    for i in range(18):
        for j in range(18):
            C[i,j]=float((W[i].T@V[j]).pow(2).sum())/64
    diag=[float(C[i,i]) for i in range(18)]
    cross=[float(C[i,j]) for i in range(18) for j in range(18) if i!=j]
    cm=sum(cross)/len(cross)
    nd=sum(1 for i in range(18) if C[i,i]>=1.3*cm)
    a0row=float(C[0,1:].mean())
    nxt=[float(C[i,i+1]) for i in range(17)]
    nm_=sum(nxt)/len(nxt); dm=sum(diag)/len(diag)
    for i in range(18):
        print(f'attn{i:2d}: own {C[i,i]:.3f} next '
              f'{(C[i,i+1] if i<17 else float("nan")):.3f} '
              f'rowmean {float(C[i].mean()):.3f}',flush=True)
    print(f'cross-block mean {cm:.3f} (random floor 0.056) | diag mean '
          f'{dm:.3f} | next mean {nm_:.3f} | attn0 row {a0row:.3f}')
    pa=nd>=12; pb=a0row>=cm
    out={'map':[[round(float(C[i,j]),4) for j in range(18)]
                for i in range(18)],
         'diag_over_cross':nd,'cross_mean':round(cm,4),
         'diag_mean':round(dm,4),'next_mean':round(nm_,4),
         'attn0_row':round(a0row,4),
         'pred_a':bool(pa),'pred_b':bool(pb),
         'next_beats_same':bool(nm_>dm)}
    print(f"(a) same-block motif ({nd}/18 at 1.3x): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) attn0 broadcast row >= cross mean: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) next-block {nm_:.3f} vs same-block {dm:.3f}: "
          f"{'next' if nm_>dm else 'same'} wins (informational)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
