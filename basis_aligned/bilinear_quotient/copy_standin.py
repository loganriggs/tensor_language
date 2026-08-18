"""PROGRAMMATIC COPY STAND-IN for the induction band (attn3/4/5). At
induction sites (target seen earlier), replace each band component's
attention output with an explicit retrieval: full attention to the single
position after the last previous occurrence of the current token --
out = c_proj((1-lamb)*v[m] + lamb*v1[m]) with m the matched successor,
v computed from the layer's own normalized input inside the hook. If this
recovers what constants could not (3%, section 247), the band's mechanism
is causally certified as match-and-copy at the value level.

REGISTERED PREDICTIONS: (a) at induction sites the copy stand-in recovers
>= 40% of the band's joint on-slice ablation damage; (b) control: the same
stand-in pointing at a RANDOM earlier position recovers <= 10%; (c) floor:
the constant stand-in re-measured for the band alone (expected ~0-10%)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean, NH, HD
from circuit_dictionary import classify
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'copy_standin_results.json'
R0,R1=120,300
BAND=[3,4,5]

def match_index(r0,r1):
    """mm[row,pos] = position of successor of last previous occurrence of
    toks[pos], else -1. rnd[row,pos] = random earlier position."""
    import random; random.seed(0)
    MM=torch.full((r1-r0,256),-1,dtype=torch.long)
    RR=torch.full((r1-r0,256),-1,dtype=torch.long)
    for r in range(r0,r1):
        toks=FW[r,:257].tolist()
        last={}
        for pos in range(256):
            t=toks[pos]
            if t in last and last[t]+1<=pos:
                MM[r-r0,pos]=last[t]+1
                RR[r-r0,pos]=random.randint(0,max(pos-1,0))
            last[t]=pos
    return MM,RR

@torch.no_grad()
def main():
    t0=time.time()
    clsC=classify(R0,R1).to(DEV)
    ind=(clsC==8)
    MM,RR=match_index(R0,R1)
    MM=MM.to(DEV); RR=RR.to(DEV)
    amus={li:attn_mean(li) for li in BAND}
    # slice-mean constants for floor (fit window A quickly)
    caps={li:[] for li in BAND}; hs=[]
    for li in BAND:
        def mk(li=li):
            return lambda mo_,i_,o_: caps[li].append(
                (o_[0] if isinstance(o_,tuple) else o_)
                .detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].attn.register_forward_hook(mk()))
    clsA=classify(300,512).reshape(-1)
    for i in range(300,512,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    consts={li:torch.cat(caps[li])[clsA==8].mean(0) for li in BAND}
    trueA={li:torch.cat(caps[li]) for li in BAND}
    del caps
    # fit per-layer scalar calibration for the copy stand-in on window A
    MMA,_=match_index(300,512)
    MMA=MMA.to(DEV)
    standA={li:[] for li in BAND}; hs=[]
    curA={'b0':0}
    for li in BAND:
        a=m.transformer.h[li].attn
        def mk(li=li,a=a):
            def hook(mod,i_,o_):
                out=o_[0] if isinstance(o_,tuple) else o_
                B,T,_=out.shape
                hcur=i_[0]
                v=a.c_v(hcur).view(B,T,NH,HD)
                vv=((1-a.lamb)*v+a.lamb*v).reshape(B,T,-1)
                mm=MMA[curA['b0']:curA['b0']+B,:T].clamp_min(0)
                src=torch.gather(vv,1,mm[:,:,None]
                                 .expand(B,T,vv.shape[-1]))
                standA[li].append(a.c_proj(src).detach()
                                  .reshape(-1,D).float())
            return hook
        hs.append(m.transformer.h[li].attn.register_forward_hook(mk()))
    for i in range(300,512,4):
        curA['b0']=i-300
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    validA=(MMA.reshape(-1)>=0)&(clsA.to(DEV)==8)
    ALPHA={}
    for li in BAND:
        S=torch.cat(standA[li])[validA]; Tr=trueA[li].to(DEV)[validA]
        ALPHA[li]=float((S*Tr).sum()/S.pow(2).sum().clamp_min(1e-6))
        print(f'alpha attn{li}: {ALPHA[li]:+.4f}',flush=True)
    del standA, trueA
    cur={'b0':0}
    def pertok(mode):
        hs=[]
        if mode!='clean':
            for li in BAND:
                a=m.transformer.h[li].attn
                mu=amus[li]; cst=consts[li]
                def mk(li=li,a=a,mu=mu,cst=cst,mode=mode):
                    def hook(mod,i_,o_):
                        out=o_[0] if isinstance(o_,tuple) else o_
                        B,T,_=out.shape
                        mm=(MM if mode!='random' else RR)[
                            cur['b0']:cur['b0']+B,:T]
                        sl=ind.view(R1-R0,256)[cur['b0']:cur['b0']+B,:T]
                        eff=sl&(mm>=0)
                        if mode in ('copy','random','copy_cal'):
                            hcur=i_[0]
                            v=a.c_v(hcur).view(B,T,NH,HD)
                            v1=v  # band layers: lamb mixes with layer-1 v;
                                  # approximation noted: use own v
                            vv=((1-a.lamb)*v+a.lamb*v1).reshape(B,T,-1)
                            gath=mm.clamp_min(0)
                            src=torch.gather(vv,1,gath[:,:,None]
                                             .expand(B,T,vv.shape[-1]))
                            rep=a.c_proj(src)
                            if mode=='copy_cal': rep=rep*ALPHA[li]
                        elif mode=='const':
                            rep=cst[None,None,:].expand_as(out)
                        else:
                            rep=mu[None,None,:].expand_as(out)
                        new=torch.where(eff[:,:,None],rep.to(out.dtype),out)
                        if mode=='ablate':
                            newa=mu[None,None,:].to(out.dtype).expand_as(out)
                            new=torch.where(sl[:,:,None],newa,out)
                        if isinstance(o_,tuple): return (new,)+o_[1:]
                        return new
                    return hook
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(mk()))
        ces=[]
        for i in range(R0,R1,4):
            cur['b0']=i-R0
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return torch.cat(ces)
    base=pertok('clean')
    flat=ind.reshape(-1)
    abl=float((pertok('ablate')-base)[flat].mean())
    cop=float((pertok('copy')-base)[flat].mean())
    cal=float((pertok('copy_cal')-base)[flat].mean())
    rnd=float((pertok('random')-base)[flat].mean())
    cst=float((pertok('const')-base)[flat].mean())
    ra=1-cop/max(abl,1e-6); rb=1-rnd/max(abl,1e-6); rc=1-cst/max(abl,1e-6)
    rcal=1-cal/max(abl,1e-6)
    pa=rcal>=0.40; pb=rb<=0.10
    out={'ablate':round(abl,4),'copy':round(cop,4),
         'copy_cal':round(cal,4),'cal_recovery':round(rcal,3),
         'random':round(rnd,4),
         'const':round(cst,4),'copy_recovery':round(ra,3),
         'random_recovery':round(rb,3),'const_recovery':round(rc,3),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'band ablate on ind sites {abl:+.3f} | raw copy {ra:.0%} | '
          f'CALIBRATED copy {rcal:.0%} | random {rb:.0%} | const {rc:.0%}')
    print(f"(a) calibrated copy >=40%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random <=10%: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
