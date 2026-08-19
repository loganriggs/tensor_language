"""INDUCTION RESIDUAL -- anatomize the missing 23% (user: +0.138 at
match positions is too high for a mechanism we claim to understand).
The one-read code drops the pattern tail; three hypotheses for what
the tail contains, each with a registered test:
  H1 MULTI-MATCH: the heads read ALL prior occurrences of the
     current token (a soft vote), not just the latest -- top-k code
     should recover fast with k.
  H2 CONTEXT CORRUPTION: substituting code at ALL positions corrupts
     the context that later reads consume (the local read-site
     pricing law) -- code applied ONLY at match positions should be
     much cheaper.
  H3 STRUCTURED TAIL: the dropped pattern mass sits on the match
     FAMILY (all prior occurrences and their successors), so the
     mechanism is still induction, just plural.
REGISTERED PREDICTIONS (match positions, 64 held-out rows):
  (a) top-4 code cost <= +0.05 (vs +0.138 top-1);
  (b) >=40% of dropped |pattern| mass lies on the match family;
  (c) match-only substitution <= 60% of everywhere-substitution;
  (d) top-k curve reported (k=1,2,4,8)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_residual_results.json'
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    byl={}
    for li,hd in IND: byl.setdefault(li,[]).append(hd)
    def row_masks(row):
        toks=row[:T].tolist()
        M=torch.zeros(T,dtype=torch.bool)     # is match position
        FAM=torch.zeros(T,T,dtype=torch.bool) # match family keys
        occ={}
        for q in range(T):
            t=toks[q]
            if t in occ:
                M[q]=True
                for j in occ[t]:
                    FAM[q,j]=True
                    if j+1<=q: FAM[q,j+1]=True
            occ.setdefault(t,[]).append(q)
        return M,FAM
    MASKS=[row_masks(ROWS[r]) for r in range(96)]
    def code_hooks(K,match_only=False,batch_i=0):
        hs=[]
        for li,hds in byl.items():
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hds=hds,at=at,K=K,
                   match_only=match_only,batch_i=batch_i):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                are=sys.modules[type(at).__module__].apply_rotary_emb
                Bb,Tq=X.shape[0],X.shape[1]
                v=at.c_v(X).view(Bb,Tq,9,128)
                vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v)
                                          if v1 is not None else v)
                cos,sin=at.rotary(at.c_q(X).view(Bb,Tq,9,128))
                qf=F.rms_norm(at.c_q(X).view(Bb,Tq,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(Bb,Tq,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2=F.rms_norm(at.c_q2(X).view(Bb,Tq,9,128),(128,))
                k2=F.rms_norm(at.c_k2(X).view(Bb,Tq,9,128),(128,))
                q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())/128
                s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                k2.float())/128
                pat=(sc*s2)*torch.tril(torch.ones(Tq,Tq,device=DEV))
                z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                for hd in hds:
                    p1=pat[:,hd]
                    _,idx=p1.abs().topk(K,dim=-1)
                    msk=torch.zeros_like(p1).scatter(-1,idx,1.0)
                    zc=torch.einsum('bqk,bkd->bqd',p1*msk,
                                    vm[:,:,hd].float())
                    if match_only:
                        Mq=torch.stack([MASKS[batch_i+b][0]
                                        for b in range(Bb)]).to(DEV)
                        z[:,hd]=torch.where(Mq[...,None],zc,z[:,hd])
                    else:
                        z[:,hd]=zc
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,Tq,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
        return hs
    def ce_eval(mkhooks):
        ces=[]
        for i in range(32,96,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            hs=mkhooks(i)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            for h in hs: h.remove()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                       reduction='none').view(4,T).cpu())
        return torch.cat(ces)
    okm=torch.stack([MASKS[i][0] for i in range(32,96)])
    base=ce_eval(lambda i: [])
    res={}
    for K in (1,2,4,8):
        cK=ce_eval(lambda i,K=K: code_hooks(K,batch_i=i))
        res[f'top{K}']=round(float((cK-base)[okm].mean()),4)
        print(f'top-{K}: {res[f"top{K}"]:+.4f}',flush=True)
    cMO=ce_eval(lambda i: code_hooks(1,match_only=True,batch_i=i))
    res['match_only_top1']=round(float((cMO-base)[okm].mean()),4)
    print(f'match-only top-1: {res["match_only_top1"]:+.4f}',flush=True)
    # tail composition on 16 rows
    famfrac=[]; cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    for i in range(0,16,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li,_ in IND]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in IND:
            X,v1i=cap[li]
            at=m.transformer.h[li].attn
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqd,bkd->bqk',qf[:,:,hd].float(),
                            kf[:,:,hd].float())/128
            s2=torch.einsum('bqd,bkd->bqk',q2[:,:,hd].float(),
                            k2[:,:,hd].float())/128
            pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
            ks=pat.abs().argmax(-1)
            pm=pat.abs()
            pm.scatter_(-1,ks[...,None],0.0)     # drop top-1
            for b in range(4):
                Mq,FAM=MASKS[i+b]
                Mq=Mq.to(DEV); FAM=FAM.to(DEV)
                tot=pm[b][Mq].sum()
                fam=(pm[b]*FAM.float())[Mq].sum()
                if float(tot)>0: famfrac.append(float(fam/tot))
    ff=sum(famfrac)/len(famfrac)
    pa=res['top4']<=0.05
    pb=ff>=0.4
    pc_=res['match_only_top1']<=0.6*res['top1']
    out=dict(res); out.update({'tail_family_frac':round(ff,3),
        'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
        'pred_d':True})
    print(f'tail mass on match family: {ff:.2%}')
    print(f"(a) top-4 <=+0.05: {'HELD' if pa else 'FAILED'}")
    print(f"(b) family >=40%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) match-only <=60% of everywhere: {'HELD' if pc_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
