"""Constraint-release, candidate FOUR -- the first with a home-corpus no-finetune
benefit. Section 96: deleting L9's top-8 output-PCA span improves held-out CE by
-0.0156 with no finetune (and L15's by -0.0107). Candidates 1-3 (chosen by
shifted-corpus benefit, vocabulary foreignness, structural implication) all proved
load-bearing (+0.026/+0.033/+0.067 vs finetuned control). This candidate is chosen
by the exact criterion the hypothesis needs. Prune: mean-ablate the L9 and L15
top-8 PCA spans. REGISTERED, still skeptical after three refutations: (b4) C-B >=
+0.005 (the improvement evaporates once the control also finetunes). Alternative:
C-B <= -0.002 is the FIRST CONSTRAINT-RELEASE POSITIVE and partially vindicates
the user's hypothesis. Original arms docstring follows.

User's hypothesis (2026-08-17): the residual stream constrains the model; removing
certain connections plus a SHORT FINETUNE could beat CE of the intact model.

One-sided evidence already on record: tail spans (L9/L12/L15 top-32 output spans)
improve pile CE when deleted with NO finetune (-0.007/-0.003/-0.007, §37/§38 --
though those flips were shift-dependent). The proper test needs the finetune and the
matched control: arms, all evaluated on held-out pile rows 300-452 (never trained):
  A  baseline, no finetune
  B  finetune only              (200 steps, lr 1e-5, pile rows 0-260)
  C  prune (permanently mean-ablate the three negative tail spans) + same finetune
REGISTERED PREDICTIONS: (a) B < A (finetuning on pile helps -- sanity); (b) the
user's hypothesis: C < B by >= 0.004 nats (constraint release: with the harmful
spans gone, the short finetune reorganises to a better optimum than it can reach
with them present). The null outcome C ~= B (within 0.002) would say the negative
spans are shift-artifacts only, not constraints."""
import json, sys, time, torch, copy
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV, PATCH
from tier2_model import load_elriggs
D=1152
SPANS={9:8,15:8}
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_constraint_c4_results.json')

@torch.no_grad()
def eval_ce_local(mdl):
    tot,n=0.0,0
    for i in range(300,452,4):
        b=FW[i:i+4,:257].to(DEV)
        out=mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        loss=out[1] if isinstance(out,tuple) else out
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    return tot/n

def make_hooks(mdl, spans_dirs):
    handles=[]
    for li,(Q,cbar) in spans_dirs.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,inp,outp):
                c=outp.float()@Q
                return (outp-((c-cbar)@Q.T).to(outp.dtype))
            return hook
        handles.append(mdl.transformer.h[li].mlp.register_forward_hook(mk()))
    return handles

def finetune(mdl, steps=200, lr=1e-5):
    for p in mdl.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.0)
    mdl.train()
    g=torch.Generator().manual_seed(0)
    for t in range(steps):
        i=int(torch.randint(0,256,(1,),generator=g))
        b=FW[i:i+4,:257].to(DEV)
        out=mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        loss=out[1] if isinstance(out,tuple) else out
        opt.zero_grad(); loss.backward(); opt.step()
    mdl.eval()

def main():
    t0=time.time()
    out={}
    # spans from the intact model
    spans_dirs={}
    for li,k in SPANS.items():
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        Q=orth(Vh[:k].T)
        spans_dirs[li]=(Q,(Yb@Q))
    mA,cfg=load_elriggs('bilin18', device=DEV)
    ceA=eval_ce_local(mA)
    print(f'A baseline:            {ceA:.4f}',flush=True)
    del mA; torch.cuda.empty_cache()
    mB,_=load_elriggs('bilin18', device=DEV)
    finetune(mB)
    ceB=eval_ce_local(mB)
    print(f'B finetune-only:       {ceB:.4f}',flush=True)
    del mB; torch.cuda.empty_cache()
    mC,_=load_elriggs('bilin18', device=DEV)
    hooks=make_hooks(mC,spans_dirs)
    finetune(mC)
    ceC=eval_ce_local(mC)
    print(f'C prune+finetune:      {ceC:.4f}',flush=True)
    for h in hooks: h.remove()
    del mC; torch.cuda.empty_cache()
    out={'A':ceA,'B':ceB,'C':ceC,'C_minus_B':ceC-ceB}
    pa=ceB<ceA
    pb=(ceC-ceB)>=0.005
    null=abs(ceC-ceB)<0.002
    out['pred_a']=bool(pa); out['pred_b_constraint_release']=bool(pb)
    out['null_shift_artifact']=bool(null)
    print(f"\n(a) finetune helps: {'HELD' if pa else 'FAILED'}")
    print(f"(b4 skeptical) improvement evaporates (C-B >= +0.005): "
          f"{'HELD' if pb else 'FAILED'} (C-B = {ceC-ceB:+.4f})")
    if null: print('null outcome: C ~= B -- negative spans are shift artifacts, '
                   'not constraints')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
