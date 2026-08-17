"""bilin12 FULL ATLAS: every layer's MLP top-8 span and attention (24
components), extending bilin12_fingerprints.pt. REGISTERED (mirroring the
bilin18 atlas bars, section 173): (a) median pairwise |Spearman| <= 0.3;
(b) depth-smooth: >= 60% of components' nearest same-type fingerprint within
+-2 layers; (c) type-marked: within-type median similarity > cross-type.

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
     'bilin12_atlas_results.json')
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'bilin12_atlas.pt')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('bilin12', device=DEV)
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
    MLPL=list(range(12)); ATTL=list(range(12))
    stats={'mlp':{li:[] for li in MLPL},'attn':{li:[] for li in ATTL}}
    hs=[]
    for li in MLPL:
        def mk(li=li):
            return lambda mod,i_,o_: stats['mlp'][li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mk()))
    for li in ATTL:
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
    from bilin18_joint_removal import orth
    ce0=per_token([])
    fps={}
    for li in MLPL:
        Y=torch.cat(stats['mlp'][li]); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T); cbar=Ybar@Q
        def hook(mod,i_,o_,Q=Q,cbar=cbar):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        h=m2.transformer.h[li].mlp.register_forward_hook(hook)
        fps[f'mlp{li}']=(per_token([])-ce0).cpu()
        h.remove()
        print(f'mlp{li}: net {float(fps[f"mlp{li}"].mean()):+.4f}',flush=True)
    for li in ATTL:
        mu=torch.cat(stats['attn'][li]).mean(0)
        def hook(mod,i_,o_,mu=mu):
            y,v1=o_
            return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
        h=m2.transformer.h[li].attn.register_forward_hook(hook)
        fps[f'attn{li}']=(per_token([])-ce0).cpu()
        h.remove()
        print(f'attn{li}: net {float(fps[f"attn{li}"].mean()):+.4f}',flush=True)
    torch.save({'base':ce0.cpu(),'fingerprints':fps},PT)
    keys=sorted(fps)
    S={}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            S[(a,b)]=abs(spearman(fps[a].float(),fps[b].float()))
    allv=sorted(S.values()); mpw=allv[len(allv)//2]
    def sim(a,b): return S.get((a,b),S.get((b,a),0))
    smooth=0; tot=0
    for a in keys:
        typ='mlp' if a.startswith('mlp') else 'attn'
        la=int(a[len(typ):])
        same=[b for b in keys if b.startswith(typ) and b!=a]
        best=max(same,key=lambda b:sim(a,b))
        lb=int(best[len(typ):])
        tot+=1
        if abs(lb-la)<=2: smooth+=1
    within=[v for (a,b),v in S.items()
            if (a.startswith('mlp')==b.startswith('mlp'))]
    cross=[v for (a,b),v in S.items()
           if (a.startswith('mlp')!=b.startswith('mlp'))]
    mw=sorted(within)[len(within)//2]; mc=sorted(cross)[len(cross)//2]
    pa=mpw<=0.3; pb=smooth/tot>=0.6; pc=mw>mc
    out={'nets':{k:float(v.mean()) for k,v in fps.items()},
         'median_pairwise':mpw,'depth_smooth_frac':smooth/tot,
         'within':mw,'cross':mc,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\n{len(keys)} comps | pairwise {mpw:.2f} | smooth {smooth}/{tot} | '
          f'within {mw:.2f} vs cross {mc:.2f}')
    print(f"(a) distinguishable: {'HELD' if pa else 'FAILED'}")
    print(f"(b) depth-smooth: {'HELD' if pb else 'FAILED'}")
    print(f"(c) type-marked: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} and {PT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
