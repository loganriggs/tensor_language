"""Replication before interpretation: section 121 found deleting L9's PCA-8 span
(the strongest deletion-improves regularizer span) TOGETHER with L16's span (the
model's biggest content span) is net negative: joint delta -0.024 on rows
300-348. Replicate on disjoint rows 352-448 with a fresh random-pair control.

REGISTERED PREDICTIONS: (a) the joint L9+L16 delta stays negative on disjoint
rows; (b) it is below the sum of the individual deltas by >= 0.01 (genuine
beneficial interaction, not just L9's own negativity); control (c): L9 + a
random-8 span at L16 shows joint ~= sum (|excess| <= 0.005)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l9l16_replicate_results.json')

@torch.no_grad()
def ce_eval(patches):
    hs=[]
    for li,(Q,cbar) in patches.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    tot,n=0.0,0
    for i in range(352,448,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    spans={}
    for li in (9,16):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        spans[li]=(Q,Ybar@Q)
    g=torch.Generator(device=DEV).manual_seed(11)
    Qr=orth(torch.randn(D,8,device=DEV,generator=g))
    accs=[]
    for i in range(0,36,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y16=torch.cat(accs)
    rnd16=(Qr,Y16.mean(0)@Qr)
    base=ce_eval({})
    d9=ce_eval({9:spans[9]})-base
    d16=ce_eval({16:spans[16]})-base
    joint=ce_eval({9:spans[9],16:spans[16]})-base
    dr=ce_eval({16:rnd16})-base
    jr=ce_eval({9:spans[9],16:rnd16})-base
    exc=joint-d9-d16; exc_r=jr-d9-dr
    print(f'base {base:.4f}')
    print(f'd9 {d9:+.4f} | d16 {d16:+.4f} | joint {joint:+.4f} | excess {exc:+.4f}')
    print(f'control: d16rand {dr:+.4f} | joint {jr:+.4f} | excess {exc_r:+.4f}')
    pa=joint<0; pb=exc<=-0.01; pc=abs(exc_r)<=0.005
    out={'base':base,'d9':d9,'d16':d16,'joint':joint,'excess':exc,
         'rand_excess':exc_r,'pred_a':bool(pa),'pred_b':bool(pb),
         'ctrl_c':bool(pc)}
    print(f"(a) joint negative replicates: {'HELD' if pa else 'FAILED'}")
    print(f"(b) beneficial interaction (excess <= -0.01): {'HELD' if pb else 'FAILED'}")
    print(f"(c) random-pair control additive: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
