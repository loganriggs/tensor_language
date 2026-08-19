"""Resolving ledger #13: the sqrd12 attention warp's direction was
instrument-dependent at n=3. Compute ALL 12 sqrd12 attention fingerprints and
run both instruments per component: (i) direct best-match implied fraction
(argmax correlation against bilin18's 18 attention fingerprints); (ii) axis
implied fraction (nearest bilinear component on the joint Fiedler coordinate).

REGISTERED PREDICTIONS: (a) at n=12 the two instruments' implied fractions
correlate >= 0.5 (they measure one thing given adequate sample); (b) the
median signed displacement (nominal minus implied, averaged across instruments)
has |median| >= 0.05 with a definite sign -- resolving the direction; if (a)
fails, the axis instrument is declared unreliable for weak fingerprints and
the direct-match reading stands with a caveat."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'sqrd12_full_attn_results.json')

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
    D=m2.transformer.wte.weight.shape[1]; NL=12
    amu={}
    hs=[]
    for li in range(NL):
        def mka(li=li):
            def hook(mod,i_,o_):
                y,v1=o_
                amu.setdefault(li,[]).append(y.detach().reshape(-1,D).float())
            return hook
        hs.append(m2.transformer.h[li].attn.register_forward_hook(mka()))
    for i in range(0,12,6):
        b=FW[i:i+6,:257].to(DEV)
        m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    def per_token(ablate=None):
        hs=[]
        if ablate is not None:
            ali,mu=ablate
            def hook(mod,i_,o_,mu=mu):
                y,v1=o_
                return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
            hs.append(m2.transformer.h[ali].attn.register_forward_hook(hook))
        ces=[]
        for i in range(384,448,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m2.transformer.h:
                x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            hs=[]
            if ablate is not None:
                ali,mu=ablate
                def hook(mod,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu[None,None,:].to(y.dtype).expand_as(y), v1)
                hs.append(m2.transformer.h[ali].attn.register_forward_hook(hook))
            lg=(30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    ce0=per_token()
    fsq={}
    for li in range(NL):
        mu=torch.cat(amu[li]).mean(0)
        fsq[li]=(per_token((li,mu))-ce0).cpu().float()
        print(f'sq attn{li}: net {float(fsq[li].mean()):+.4f}',flush=True)
    a18=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin18_fingerprint_atlas.pt')['fingerprints']
    f18={li:a18[f'attn{li}'].float() for li in range(18)}
    # instrument 1: direct best match
    direct={}
    for li in range(NL):
        cur={lj:abs(spearman(fsq[li],f18[lj])) for lj in range(18)}
        best=max(cur,key=cur.get)
        direct[li]=best/18
    # instrument 2: axis placement
    a12=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                   'bilin12_atlas.pt')['fingerprints']
    vecs=[];labels=[]
    for lj in range(18):
        vecs.append(f18[lj]); labels.append(('b',lj/18))
    for lj in range(12):
        vecs.append(a12[f'attn{lj}'].float()); labels.append(('b',lj/12))
    for li in range(NL):
        vecs.append(fsq[li]); labels.append(('s',li/12))
    f=fiedler(vecs)
    bl=[i for i,l in enumerate(labels) if l[0]=='b']
    sb=spearman(f[bl],torch.tensor([labels[i][1] for i in bl]))
    if sb<0: f=-f
    axis={}
    for k,li in enumerate(range(NL)):
        i=30+k
        nb=min(bl,key=lambda j:abs(float(f[j]-f[i])))
        axis[li]=labels[nb][1]
    dv=torch.tensor([direct[li] for li in range(NL)])
    av=torch.tensor([axis[li] for li in range(NL)])
    nom=torch.tensor([li/12 for li in range(NL)])
    rho=spearman(dv,av)
    disp=((nom-dv)+(nom-av))/2
    med=float(disp.median())
    for li in range(NL):
        print(f'attn{li}: nominal {li/12:.2f} | direct {direct[li]:.2f} | '
              f'axis {axis[li]:.2f}',flush=True)
    pa=rho>=0.5
    pb=abs(med)>=0.05
    out={'direct':{str(k):v for k,v in direct.items()},
         'axis':{str(k):v for k,v in axis.items()},
         'instr_agreement':rho,'median_displacement':med,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'\ninstrument agreement rho {rho:+.2f} | median displacement '
          f'(nominal-implied) {med:+.3f}')
    print(f"(a) instruments agree (>=0.5): {'HELD' if pa else 'FAILED'}")
    print(f"(b) direction resolved (|med|>=0.05): "
          f"{'HELD' if pb else 'FAILED'} "
          f"({'FRONT-shifted' if med>0 else 'BACK-shifted'} if held)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
