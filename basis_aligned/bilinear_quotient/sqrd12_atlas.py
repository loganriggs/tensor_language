"""sqrd12 FULL ATLAS (closing section 184's coverage caveat): all 12 MLP-span
and 12 attention fingerprints, saved to sqrd12_atlas.pt (earlier runs computed
subsets without saving). Then the coverage-fair leverage retest.

REGISTERED PREDICTIONS: (a) atlas bars (distinguishable <=0.3, depth-smooth
>=60%, type-marked); (b) with matched coverage, sqrd12's leverage correlations
with the other three models all rise to >= 0.65 (the section-184 lows were
coverage artifact); (c) the four-model minimum pair >= 0.60."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'sqrd12_atlas_results.json')
PT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    'sqrd12_atlas.pt')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('sqrd12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]; NL=12
    mmu={};amu={}
    hs=[]
    for li in range(NL):
        def mkm(li=li):
            return lambda mod,i_,o_: mmu.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        def mka(li=li):
            def hook(mod,i_,o_):
                y,v1=o_
                amu.setdefault(li,[]).append(y.detach().reshape(-1,D).float())
            return hook
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mkm()))
        hs.append(m2.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    def per_token(mlp_patch=None, attn_patch=None):
        hs=[]
        if mlp_patch is not None:
            li,Q,cbar=mlp_patch
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            hs.append(m2.transformer.h[li].mlp.register_forward_hook(hook))
        if attn_patch is not None:
            li,mu=attn_patch
            def hook2(mod,i_,o_,mu=mu):
                y,v1=o_
                return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
            hs.append(m2.transformer.h[li].attn.register_forward_hook(hook2))
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
        for h in hs: h.remove()
        return torch.cat(ces)
    ce0=per_token()
    fps={}
    for li in range(NL):
        Y=torch.cat(mmu[li]); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        fps[f'mlp{li}']=(per_token(mlp_patch=(li,Q,Ybar@Q))-ce0).cpu().float()
    for li in range(NL):
        mu=torch.cat(amu[li]).mean(0)
        fps[f'attn{li}']=(per_token(attn_patch=(li,mu))-ce0).cpu().float()
    torch.save({'base':ce0.cpu(),'fingerprints':fps},PT)
    print(f'saved {len(fps)} fingerprints',flush=True)
    keys=sorted(fps)
    S={}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            S[(a,b)]=abs(spearman(fps[a],fps[b]))
    mpw=sorted(S.values())[len(S)//2]
    def sim(a,b): return S.get((a,b),S.get((b,a),0))
    smooth=0
    for a in keys:
        typ='mlp' if a.startswith('mlp') else 'attn'
        la=int(a[len(typ):])
        same=[b for b in keys if b.startswith(typ) and b!=a]
        best=max(same,key=lambda b:sim(a,b))
        if abs(int(best[len(typ):])-la)<=2: smooth+=1
    within=[v for (a,b),v in S.items()
            if a.startswith('mlp')==b.startswith('mlp')]
    cross=[v for (a,b),v in S.items()
           if a.startswith('mlp')!=b.startswith('mlp')]
    mw=sorted(within)[len(within)//2]; mc=sorted(cross)[len(cross)//2]
    pa=mpw<=0.3 and smooth/len(keys)>=0.6 and mw>mc
    print(f'atlas: pairwise {mpw:.2f} | smooth {smooth}/{len(keys)} | '
          f'{mw:.2f}v{mc:.2f} -> (a) {"HELD" if pa else "FAILED"}',flush=True)
    levsq=torch.stack([v.abs() for v in fps.values()]).sum(0)
    others={}
    for name,path in (('b18','bilin18_fingerprint_atlas.pt'),
                      ('b12','bilin12_atlas.pt'),
                      ('sw18','swiglu18_atlas.pt')):
        dd=torch.load('/workspace/tensor_language/basis_aligned/'
                      f'bilinear_quotient/{path}')
        others[name]=torch.stack([v.float().abs() for v in
                                  dd['fingerprints'].values()]).sum(0)
    cors={n:spearman(levsq,v) for n,v in others.items()}
    for n,v in cors.items(): print(f'sq12-{n}: {v:+.2f}',flush=True)
    pb=all(v>=0.65 for v in cors.values())
    import itertools
    alllev={'sq12':levsq,**others}
    mins=min(spearman(alllev[a],alllev[b])
             for a,b in itertools.combinations(alllev,2))
    pc=mins>=0.60
    out={'atlas':{'pairwise':mpw,'smooth':smooth,'within':mw,'cross':mc},
         'leverage':cors,'four_model_min':mins,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(b) coverage-fair >= 0.65: {'HELD' if pb else 'FAILED'}")
    print(f"(c) four-model min >= 0.60: {'HELD' if pc else 'FAILED'} ({mins:+.2f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} and {PT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
