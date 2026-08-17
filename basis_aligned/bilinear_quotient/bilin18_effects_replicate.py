"""Effects-side replication of the dilution law (section 144). Recompute all
ten adjacent-edge transplant effects with FRESH base rows (330-354) and source
rows (430-454), skip the score machinery. REGISTERED: (a) Spearman(new, old
effects) >= 0.8; (b) dilution law on the all-fresh pair >= 0.7.

Prior context -- the registered-open blind protocol for EDGES: can weights+stats rank the tail's
edge strengths before any intervention? For the ten adjacent pairs (i,i+1),
i=5..14, the coupling score is trace(C_i G2(i+1)) -- layer i's output covariance
contracted with layer i+1's input-mode Lambda-Gram (weights + input second moment).
Ground truth: transplant layer i's full MLP write from different documents and
measure the induced relative change in layer i+1's MLP output.

REGISTERED PREDICTIONS: (a) Spearman(score, measured effect) >= 0.6 over the ten
edges; (b) the top-scored edge's measured effect >= 3x the bottom-scored edge's;
(c) ALIGNMENT is what predicts, not loudness: the score's Spearman exceeds the
loudness-null score trace(C_i)*trace(G2)/1152 (magnitudes without alignment) by
>= 0.2."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
EDGES=[(i,i+1) for i in range(5,15)]
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_effects_replicate_results.json')

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base_rows=FW[330:354,:257].to(DEV); src_rows=FW[430:454,:257].to(DEV)
    # covariances of writes and input second moments, one forward sweep each
    mos={}; ins={}
    hs=[]
    for li in range(5,16):
        def mko(li=li):
            return lambda mod,i_,o_: mos.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        def mki(li=li):
            return lambda mod,inp: ins.setdefault(li,[]).append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
        hs.append(m.transformer.h[li].mlp.register_forward_pre_hook(mki()))
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    scores=[0.0]*10; null_scores=[0.0]*10
    # measured: full-write transplant at i -> relative change in j's output
    effects=[]
    for i,j in EDGES:
        cap={}
        def cap_hook(mod,i_,o_): cap['mo']=o_.detach()
        h=m.transformer.h[i].mlp.register_forward_hook(cap_hook)
        outj={}
        def outj_hook(mod,i_,o_): outj.setdefault('l',[]).append(
            o_.detach().reshape(-1,D).float())
        h2=m.transformer.h[j].mlp.register_forward_hook(outj_hook)
        m(src_rows[:, :-1].contiguous(), src_rows[:,1:].contiguous())
        src_mo=cap['mo']; outj['l']=[]
        m(base_rows[:,:-1].contiguous(), base_rows[:,1:].contiguous())
        base_j=torch.cat(outj['l']); outj['l']=[]
        h.remove()
        def tr_hook(mod,i_,o_): return src_mo.to(o_.dtype)
        h3=m.transformer.h[i].mlp.register_forward_hook(tr_hook)
        m(base_rows[:,:-1].contiguous(), base_rows[:,1:].contiguous())
        h3.remove(); h2.remove()
        pat_j=torch.cat(outj['l'])
        eff=float((pat_j-base_j).pow(2).mean())/float(base_j.var())
        effects.append(eff)
        print(f'edge {i}->{j}: score {scores[len(effects)-1]:.3e} | '
              f'measured {eff:.4f}',flush=True)
    OLD=[0.2768,0.3593,0.3403,0.2401,0.2771,0.1500,0.1620,0.1396,0.1475,0.0406]
    NEWR=json.load(open('/workspace/tensor_language/basis_aligned/'
                        'bilinear_quotient/bilin18_dilution_replicate_results.json'))['ratios']
    sO=spearman(effects,OLD); sD=spearman(effects,NEWR)
    pa=sO>=0.8; pb=sD>=0.7
    out={'effects_new':effects,'effects_old':OLD,'spearman_old':sO,
         'spearman_dilution_fresh':sD,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'\nSpearman(new, old effects): {sO:+.2f} | vs fresh ratios: {sD:+.2f}')
    print(f"(a) effects replicate (>=0.8): {'HELD' if pa else 'FAILED'}")
    print(f"(b) all-fresh dilution law (>=0.7): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
