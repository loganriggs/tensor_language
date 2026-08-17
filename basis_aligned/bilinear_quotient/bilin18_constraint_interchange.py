"""Constraint-release, candidate three: cut the interchange edge. Candidates one
(tail spans, +0.026) and two (dissident L11, +0.033) were load-bearing. The last
motivated class: the L16->L17 interaction that carries the composition product-law
excess (sections 43-48) -- if any connection acts as a CONSTRAINT the residual
stream imposes, it is the verified cross-layer coupling that three removal attempts
could not close. Prune: project L16's top-8 output-PCA span (variation, mean kept)
out of L17's MLP input, permanently, then finetune 200 steps; compare to
finetune-only.

REGISTERED, skeptical: (b17) C-B >= +0.01 (the edge is load-bearing function, not
constraint). Alternative C-B <= 0 would be the first constraint-release positive,
localized at the interchange. Sanity (a): finetune helps."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import load_elriggs
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_constraint_interchange_results.json')

@torch.no_grad()
def eval_ce_local(mdl):
    tot,n=0.0,0
    for i in range(300,452,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    return tot/n

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
    # L16's top-8 output span, from the intact model
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y-Y.mean(0)).float(), full_matrices=False)
    Q=orth(Vh[:8].T)
    # mean of L17 mlp input along the span, from the intact model
    ins=[]
    h=m.transformer.h[17].mlp.register_forward_pre_hook(
        lambda mod,inp: ins.append(inp[0].detach().reshape(-1,D).float()) or None)
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    cbar=(torch.cat(ins)@Q).mean(0)
    def cut(mdl):
        def pre(mod,inp):
            x=inp[0]
            c=x.float()@Q
            return ((x-((c-cbar)@Q.T).to(x.dtype)),)+inp[1:]
        return mdl.transformer.h[17].mlp.register_forward_pre_hook(pre)
    mA,_=load_elriggs('bilin18', device=DEV)
    ceA=eval_ce_local(mA)
    hA=cut(mA); ceA_cut=eval_ce_local(mA); hA.remove()
    print(f'A baseline:           {ceA:.4f} (cut, no finetune: {ceA_cut:.4f})',flush=True)
    del mA; torch.cuda.empty_cache()
    mB,_=load_elriggs('bilin18', device=DEV)
    finetune(mB); ceB=eval_ce_local(mB)
    print(f'B finetune-only:      {ceB:.4f}',flush=True)
    del mB; torch.cuda.empty_cache()
    mC,_=load_elriggs('bilin18', device=DEV)
    hC=cut(mC); finetune(mC); ceC=eval_ce_local(mC); hC.remove()
    print(f'C cut+finetune:       {ceC:.4f}',flush=True)
    del mC; torch.cuda.empty_cache()
    pa=ceB<ceA; pb=(ceC-ceB)>=0.01
    out={'A':ceA,'A_cut':ceA_cut,'B':ceB,'C':ceC,'C_minus_B':ceC-ceB,
         'pred_a':bool(pa),'pred_b17_loadbearing':bool(pb)}
    print(f"\n(a) finetune helps: {'HELD' if pa else 'FAILED'}")
    print(f"(b17) interchange load-bearing (C-B >= +0.01): "
          f"{'HELD' if pb else 'FAILED'} (C-B = {ceC-ceB:+.4f})")
    if ceC-ceB<=0: print('FIRST CONSTRAINT-RELEASE POSITIVE -- localize and rerun')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
