"""Does the cross-family model land on the family's causal depth axis? Compute
sqrd12's 12 MLP-span fingerprints, then embed all THREE models' MLP fingerprints
(18+12+12) jointly. Section 169 predicts sqrd12 sits FRONT-SHIFTED on the axis.

REGISTERED PREDICTIONS: (a) the three-model joint coordinate still tracks depth
fraction at >= 0.7 (sqrd12 joins the axis at all); (b) front-shift signature:
for >= 8/12 sqrd12 components, the axis-implied fraction (nominal fraction of
the nearest bilinear-model component in the embedding coordinate) is LESS than
sqrd12's nominal fraction; (c) shuffled null <= 0.3."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'sqrd12_join_axis_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def fiedler(vecs):
    n=len(vecs)
    S=torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            S[i,j]=abs(spearman(vecs[i],vecs[j]))
    L=torch.diag(S.sum(1))-S
    ev,U=torch.linalg.eigh(L)
    return U[:,1]

@torch.no_grad()
def main():
    t0=time.time()
    m2,cfg=load_elriggs('sqrd12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]; NL=len(m2.transformer.h)
    stats={li:[] for li in range(NL)}
    hs=[]
    for li in range(NL):
        def mk(li=li):
            return lambda mod,i_,o_: stats[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m2.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    def per_token(patch=None):
        hs=[]
        if patch is not None:
            li,Q,cbar=patch
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            hs.append(m2.transformer.h[li].mlp.register_forward_hook(hook))
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
    fsq={}
    for li in range(NL):
        Y=torch.cat(stats[li]); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        fsq[li]=(per_token((li,Q,Ybar@Q))-ce0).cpu().float()
        print(f'sqrd12 mlp{li}: net {float(fsq[li].mean()):+.4f}',flush=True)
    a18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprint_atlas.pt')['fingerprints']
    a12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_atlas.pt')['fingerprints']
    vecs=[];labels=[]
    for li in range(18):
        vecs.append(a18[f'mlp{li}'].float()); labels.append(('b18',li/18))
    for li in range(12):
        vecs.append(a12[f'mlp{li}'].float()); labels.append(('b12',li/12))
    for li in range(12):
        vecs.append(fsq[li]); labels.append(('sq',li/12))
    f=fiedler(vecs)
    sfrac=abs(spearman(f,torch.tensor([l[1] for l in labels])))
    # orient axis with depth
    sign=1 if spearman(f,torch.tensor([l[1] for l in labels]))>0 else -1
    f=f*sign
    bl=[i for i,l in enumerate(labels) if l[0]!='sq']
    sq=[i for i,l in enumerate(labels) if l[0]=='sq']
    fronts=0
    for i in sq:
        nb=min(bl,key=lambda j:abs(float(f[j]-f[i])))
        if labels[nb][1]<labels[i][1]: fronts+=1
    g=torch.Generator().manual_seed(0)
    vecs_n=[v[torch.randperm(len(v),generator=g)] for v in vecs]
    fn=fiedler(vecs_n)
    sn=abs(spearman(fn,torch.tensor([l[1] for l in labels])))
    pa=sfrac>=0.7; pb=fronts>=8; pc=sn<=0.3
    out={'joint_fraction':sfrac,'null':sn,'front_shift_count':fronts,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'\nthree-model joint vs fraction: {sfrac:.2f} (null {sn:.2f}) | '
          f'sqrd12 front-shifted {fronts}/12')
    print(f"(a) sqrd12 joins the axis: {'HELD' if pa else 'FAILED'}")
    print(f"(b) front-shift >= 8/12: {'HELD' if pb else 'FAILED'}")
    print(f"(c) null: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
