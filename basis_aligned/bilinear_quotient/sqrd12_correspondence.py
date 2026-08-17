"""Third-model test: sqrd12 (squared attention, CONVENTIONAL MLPs, 12 layers).
If depth-fraction correspondence (sections 163-166) holds here, it transcends
the bilinear-MLP family. Attention components only (L1, L2, L6). REGISTERED:
(a) identity check -- base CE <= 4.7 (user's standing rule); (b) each sqrd12
attention fingerprint's best bilin18 match (vs attn 1,3,5,7,9,11) lands within
+-0.15 of its own depth fraction; (c) analog correlations >= 3x the 0.05 floor.

Prior context -- bilin12 fingerprint counterpart (benchmark model-level test split), plus a new
cross-model question: the eval rows are the SAME TEXT for both models, so
fingerprints are comparable position-by-position. Do analogous components (front
attention, mid MLP, late MLP) produce correlated fingerprints across models?

Components: mlp top-8 spans of L1,L5,L8; full attention of L1,L2,L6 (module
hooks -- bilin12 runs through TT blocks, so blk.attn / blk.mlp are called as
modules). REGISTERED: (a) bilin12 fingerprints mutually distinguishable
(median pairwise |Spearman| <= 0.5); (b) base-loss floor <= 0.25; (c)
CROSS-MODEL: median |Spearman| between analog pairs (bilin12 mlp1~bilin18 mlp1,
mlp5~mlp5, attn1~attn1, attn2~attn2, attn6~attn6) exceeds the median over
non-analog cross-model pairs (the two models distribute function over the same
text similarly); alternative: function placement is model-idiosyncratic."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'sqrd12_fingerprints_results.json')
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'sqrd12_fingerprints.pt')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('sqrd12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def per_token(hooks):
        ces=[]
        for i in range(384,448,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        return torch.cat(ces)
    # stats pass for means/spans
    stats={'attn':{li:[] for li in (1,2,6)}}
    hs=[]
    for li in (1,2,6):
        def mka(li=li):
            def hook(mod,i_,o_):
                y,v1=o_
                stats['attn'][li].append(y.detach().reshape(-1,D).float())
            return hook
        hs.append(m2.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    ce0=per_token([])
    print(f'sqrd12 base CE: {float(ce0.mean()):.4f}',flush=True)
    fps={}
    for li in (1,2,6):
        mu=torch.cat(stats['attn'][li]).mean(0)
        def hook(mod,i_,o_,mu=mu):
            y,v1=o_
            return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
        h=m2.transformer.h[li].attn.register_forward_hook(hook)
        fps[f'attn{li}']=(per_token([])-ce0).cpu()
        h.remove()
        print(f'attn{li}: net {float(fps[f"attn{li}"].mean()):+.4f}',flush=True)
    torch.save({'base':ce0.cpu(),'fingerprints':fps},PT)
    keys=list(fps); pw=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            pw.append(abs(spearman(fps[keys[i]].float(),fps[keys[j]].float())))
    mpw=sorted(pw)[len(pw)//2]
    fb=sorted(abs(spearman(ce0.float().cpu(),v.float())) for v in fps.values())
    mb=fb[len(fb)//2]
    d18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprints.pt')
    f18={k:v.float() for k,v in d18['fingerprints'].items()}
    L18=[k for k in f18 if k.startswith('attn')]
    fr12={'attn1':1/12,'attn2':2/12,'attn6':6/12}
    hits=0; analogs=[]
    for a in ('attn1','attn2','attn6'):
        cur={b:abs(spearman(fps[a].float(),f18[b])) for b in L18}
        best=max(cur,key=cur.get)
        bf=int(best[4:])/18
        if abs(bf-fr12[a])<=0.15: hits+=1
        analogs.append(cur[best])
        print(f'{a}: best {best} ({cur[best]:.2f}) fraction {bf:.2f} vs own '
              f'{fr12[a]:.2f}',flush=True)
    pa=float(ce0.mean())<=4.7
    pb=hits==3
    pc=all(v>=0.15 for v in analogs)
    out={'nets':{k:float(v.mean()) for k,v in fps.items()},
         'base_ce':float(ce0.mean()),'hits':hits,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) identity CE <= 4.7: {'HELD' if pa else 'FAILED'} "
          f"({float(ce0.mean()):.3f})")
    print(f"(b) fractions match 3/3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) matches >= 3x floor: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} and {PT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
