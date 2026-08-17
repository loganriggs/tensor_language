"""Per-token anatomy of section 97's reclaim: WHY does the pruned model (C) lose
to the finetuned control (B) by +0.011 despite the spans' replicated frozen
benefit? Section 98 says the spans buy sharpness on easy tokens at the cost of
overshoot on hard ones; deleting them trades the reverse way.

REGISTERED PREDICTIONS, per-token CE on held-out rows 300-452, quartiles defined
by the BASELINE model's per-token loss: (aP) finetuned-C retains a hard-token
advantage over finetuned-B (mean CE on the hardest quartile: C < B); (bP) the
+0.011 aggregate gap lives on the easy tokens (C - B on the easiest quartile
>= +0.010). Alternative: if C's hard-token advantage is gone, the frozen
redistribution signature says nothing about adapted models."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import load_elriggs
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_reclaim_anatomy_results.json')

@torch.no_grad()
def per_token(mdl):
    ces=[]
    for i in range(300,452,4):
        b=FW[i:i+4,:257].to(DEV)
        idx,tg=b[:,:-1].contiguous(), b[:,1:].contiguous()
        x=F.rms_norm(mdl.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in mdl.transformer.h:
            x,v1=blk(x,v1,x0)
        x=F.rms_norm(x,(D,))
        lg=mdl.lm_head(x); lg=30*torch.tanh(lg/30)
        ces.append(F.cross_entropy(lg.float().view(-1,lg.size(-1)),
                                   tg.reshape(-1),reduction='none'))
    return torch.cat(ces)

def make_hooks(mdl, spans_dirs):
    hs=[]
    for li,(Q,cbar) in spans_dirs.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(mdl.transformer.h[li].mlp.register_forward_hook(mk()))
    return hs

def finetune(mdl, steps=200, lr=1e-5):
    for p in mdl.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.0)
    mdl.train()
    g=torch.Generator().manual_seed(0)
    for t in range(steps):
        i=int(torch.randint(0,256,(1,),generator=g))
        b=FW[i:i+4,:257].to(DEV)
        loss=mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        opt.zero_grad(); loss.backward(); opt.step()
    mdl.eval()

def main():
    t0=time.time()
    spans={}
    for li in (9,15):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    ceA=per_token(m)
    q=torch.quantile(ceA.float(),torch.tensor([0.25,0.75],device=DEV))
    easy=ceA<=q[0]; hard=ceA>=q[1]
    mB,_=load_elriggs('bilin18', device=DEV)
    finetune(mB); ceB=per_token(mB)
    del mB; torch.cuda.empty_cache()
    mC,_=load_elriggs('bilin18', device=DEV)
    hs=make_hooks(mC,spans)
    finetune(mC); ceC=per_token(mC)
    for h in hs: h.remove()
    del mC; torch.cuda.empty_cache()
    d_hard=float((ceC-ceB)[hard].mean()); d_easy=float((ceC-ceB)[easy].mean())
    d_all=float((ceC-ceB).mean())
    pa=d_hard<0; pb=d_easy>=0.010
    out={'gap_all':d_all,'gap_hard':d_hard,'gap_easy':d_easy,
         'pred_aP':bool(pa),'pred_bP':bool(pb)}
    print(f'C-B overall {d_all:+.4f} | hard quartile {d_hard:+.4f} | '
          f'easy quartile {d_easy:+.4f}')
    print(f"(aP) C keeps hard-token advantage: {'HELD' if pa else 'FAILED'}")
    print(f"(bP) gap lives on easy tokens (>= +0.010): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
