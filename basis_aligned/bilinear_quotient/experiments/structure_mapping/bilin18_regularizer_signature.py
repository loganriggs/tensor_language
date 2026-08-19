"""WHY do the negative spans help the frozen model? Section 97's resolution claims
they act as noise at the frozen operating point. A regularization story makes a
token-level prediction: the deletion benefit should be CONCENTRATED where the model
is confidently wrong (high loss), trimming overshoot -- not spread uniformly.

Delete L9+L15 PCA-8 spans (the replicated pair), per-token CE deltas on rows
384-452. REGISTERED PREDICTIONS: (a) the benefit is overshoot-trimming -- tokens in
the top loss quartile of the base model capture >= 60% of the total improvement;
(b) on the bottom (easy) quartile the deletion HURTS or is flat (mean delta >=
-0.001); control: the same statistics under a fresh random-8 deletion at the same
sites are flat everywhere (|mean| <= 0.001 per quartile)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_l11_function import run as run_ce
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_regularizer_signature_results.json')

@torch.no_grad()
def per_token(rows, patches):
    hs=[]
    for li,(Q,cbar) in patches.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    ces=[]
    for i in range(0,len(rows),4):
        b=rows[i:i+4].to(DEV)
        idx,tg=b[:,:-1].contiguous(), b[:,1:].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        x=F.rms_norm(x,(D,))
        logits=m.lm_head(x); logits=30*torch.tanh(logits/30)
        ce=F.cross_entropy(logits.float().view(-1,logits.size(-1)),
                           tg.reshape(-1),reduction='none')
        ces.append(ce)
    for h in hs: h.remove()
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    rows=FW[384:452,:257]
    spans={}
    for li in (9,15):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    ce0=per_token(rows,{})
    ce1=per_token(rows,spans)
    g=torch.Generator(device=DEV).manual_seed(3)
    rnd={li:(orth(torch.randn(D,8,device=DEV,generator=g)),None) for li in (9,15)}
    rnd={li:(Q,(spans[li][1]*0+ (torch.zeros(8,device=DEV)))) for li,(Q,_) in rnd.items()}
    # random spans need their own means: recompute cbar for the random Qs
    for li in rnd:
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs)
        rnd[li]=(rnd[li][0], Y.mean(0)@rnd[li][0])
    ce2=per_token(rows,rnd)
    d=(ce1-ce0); dr=(ce2-ce0)
    q=torch.quantile(ce0.float(),torch.tensor([0.25,0.75],device=DEV))
    hard=ce0>=q[1]; easy=ce0<=q[0]
    tot=float(d.sum())
    hard_share=float(d[hard].sum())/tot if tot<0 else float('nan')
    mean_easy=float(d[easy].mean())
    ctrl=[float(dr[hard].mean()),float(dr[easy].mean())]
    pa=(tot<0) and hard_share>=0.6
    pb=mean_easy>=-0.001
    pc=all(abs(c)<=0.001 for c in ctrl)
    out={'total_delta':tot,'hard_quartile_share':hard_share,
         'easy_quartile_mean':mean_easy,'ctrl_hard_easy':ctrl,
         'pred_a':bool(pa),'pred_b':bool(pb),'ctrl_held':bool(pc)}
    print(f'total delta {tot:+.1f} (per-token mean {float(d.mean()):+.5f})')
    print(f'hard-quartile share of improvement: {hard_share:.2f}')
    print(f'easy-quartile mean delta: {mean_easy:+.5f}')
    print(f'random control hard/easy means: {ctrl[0]:+.5f}/{ctrl[1]:+.5f}')
    print(f"(a) overshoot-trimming (>=60% in hard quartile): {'HELD' if pa else 'FAILED'}")
    print(f"(b) easy tokens not helped: {'HELD' if pb else 'FAILED'}")
    print(f"random control flat: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
