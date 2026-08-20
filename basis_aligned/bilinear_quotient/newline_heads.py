"""NEWLINE HEADS -- 496: step 2 of the behaviour-defined attempt.
Step 1 ranked by the wrong statistic (494). The corrected ranking
(495) found a real one: attention layer 12 costs +0.0777 nats at
newline targets against +0.0073 elsewhere -- a ratio of 10.64,
five times the next component, and 2.7x its own position-matched
control (3.95). Layers a10 (4.19), a16 (2.58), a15 (2.42), a14
(2.01) trail it. So a12 is the first component in this program
that is specific to a BEHAVIOUR rather than expensive in general.
Two questions a layer-level number cannot answer: does the
specificity survive decomposition into heads (or is it an
interaction among nine heads, none of which is newline-specific
alone), and does the responsible head do something visibly
different at newline targets?
The head atlas gives an independent advance bet. Of a12's nine
heads, eight have delete costs between -0.0008 and 0.0078; 12.6
costs 0.0726 -- ten times the next -- and is the only one with a
long-range profile (window_match 0.164, other-motif share 0.795,
punct read-enrichment 2.34). If a12's newline specificity is
carried by one head, the atlas says which head that must be, and
the atlas knew nothing about newlines.
Method: mean-ablate each of a12's nine heads individually (the
head's z replaced by its own mean over batch and position, exactly
the in-place substitution of 444), priced at the same newline
targets, position-matched controls, and remainder as 495. Then for
every head, the attention profile at newline queries against
control queries: signed share of the score mass landing on the
most recent PRECEDING newline key. There is no softmax, so shares
are computed against the sum of |score| over the causal window.
REGISTERED PREDICTIONS:
  (a) SURVIVES DECOMPOSITION: at least one head of a12 reaches a
      newline/elsewhere damage ratio >= 2.0. If none does, the
      layer effect is an interaction and the head story is dead;
  (b) THE ATLAS CALLED IT: the highest-ratio head is 12.6. This is
      a bet placed from delete cost alone, before any newline
      measurement, and it can fail eight ways;
  (c) IT LOOKS DIFFERENT THERE: the winning head's signed share on
      the most recent preceding newline key is at least 1.5x
      larger at newline-target queries than at control queries.
  NULL: the winning head's position-matched control ratio must be
      below its newline ratio; report both regardless."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_heads_results.json'
NFRESH=32

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=__import__('sys').modules[
        type(m.transformer.h[0].attn).__module__].apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    nl=torch.zeros(NFRESH,T,dtype=torch.bool)
    isnl=torch.zeros(NFRESH,T,dtype=torch.bool)   # key-side
    for r in range(NFRESH):
        for q in range(T):
            nl[r,q]=chr(10) in cl.d1(int(fresh[r,q+1]))
            isnl[r,q]=chr(10) in cl.d1(int(fresh[r,q]))
    ctrl=torch.zeros_like(nl)
    g=torch.Generator().manual_seed(29)
    for r in range(NFRESH):
        k=int(nl[r].sum())
        if k==0: continue
        pos=nl[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(0,T-1)
        ctrl[r,j]=True
    print(f'{int(nl.sum())} newline targets | {int(ctrl.sum())} '
          f'controls | {int(isnl.sum())} newline keys',flush=True)
    at=m.transformer.h[LJ].attn

    def make_hook(HD):
        def fh(mo,args,o_):
            y,v1r=o_
            X2=args[0]; B=X2.shape[0]
            v1b=args[1] if args[1] is not None else v1r
            vv=at.c_v(X2).view(B,T,NH,128)
            vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
            c2,s2=at.rotary(at.c_q(X2).view(B,T,NH,128))
            def r2(w):
                return are(F.rms_norm(w(X2).view(B,T,NH,128),
                                      (128,)),c2,s2)
            qq,kk=r2(at.c_q),r2(at.c_k)
            q22,k22=r2(at.c_q2),r2(at.c_k2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                            kk.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                             k22.float())/128
            p2=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
            zz[:,HD]=zz[:,HD].mean(dim=(0,1),keepdim=True)
            return (at.c_proj(zz.transpose(1,2).contiguous()
                    .view(B,T,-1).to(X2.dtype)),v1r)
        return fh

    def run(HD):
        acc={'nl':[0.0,0],'ctrl':[0.0,0],'rest':[0.0,0]}
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=([at.register_forward_hook(make_hook(HD))]
                if HD is not None else [])
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            a=nl[i:i+4]; c=ctrl[i:i+4]; rest=~(a|c)
            for nm,mk in (('nl',a),('ctrl',c),('rest',rest)):
                acc[nm][0]+=float(ce[mk].sum()); acc[nm][1]+=int(mk.sum())
            for h in hs: h.remove()
        return {k:acc[k][0]/max(acc[k][1],1) for k in acc}

    base=run(None)
    res={}
    for HD in range(NH):
        cur=run(HD)
        dn=cur['nl']-base['nl']; dc=cur['ctrl']-base['ctrl']
        dr=cur['rest']-base['rest']
        res[f'{LJ}.{HD}']={
            'd_newline':round(dn,4),'d_ctrl':round(dc,4),
            'd_rest':round(dr,4),
            'ratio_nl':round(dn/max(dr,1e-4),2),
            'ratio_ctrl':round(dc/max(dr,1e-4),2)}
        print(f'{LJ}.{HD}: nl {dn:+.4f} rest {dr:+.4f} '
              f'ratio {res[f"{LJ}.{HD}"]["ratio_nl"]} '
              f'(ctrl {res[f"{LJ}.{HD}"]["ratio_ctrl"]})',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)

    # attention profile: share on the most recent preceding newline
    prof={f'{LJ}.{h}':{'nl':[0.0,0],'ctrl':[0.0,0]} for h in range(NH)}
    cap={}
    hh=at.register_forward_pre_hook(
        lambda mo_,args: cap.__setitem__('X',args[0]))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X']
        c2,s2=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r2(w):
            return are(F.rms_norm(w(X).view(B,T,NH,128),(128,)),
                       c2,s2)
        qq,kk=r2(at.c_q),r2(at.c_k); q22,k22=r2(at.c_q2),r2(at.c_k2)
        sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),kk.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),k22.float())/128
        p2=((sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        for b in range(B):
            r=i+b
            keys=isnl[r].nonzero().squeeze(1).tolist()
            if not keys: continue
            for nm,mask in (('nl',nl[r]),('ctrl',ctrl[r])):
                for q in mask.nonzero().squeeze(1).tolist():
                    prev=[k for k in keys if k<q]
                    if not prev: continue
                    k=prev[-1]
                    for h in range(NH):
                        prof[f'{LJ}.{h}'][nm][0]+=float(
                            p2[b,h,q,k]/den[b,h,q])
                        prof[f'{LJ}.{h}'][nm][1]+=1
    hh.remove()
    P={k:{nm:round(v[nm][0]/max(v[nm][1],1),4) for nm in ('nl','ctrl')}
       for k,v in prof.items()}
    for k in P:
        P[k]['ratio']=round(P[k]['nl']/max(abs(P[k]['ctrl']),1e-4),2)
    ranked=sorted(res,key=lambda k:-res[k]['ratio_nl'])
    top=ranked[0]
    pa=res[top]['ratio_nl']>=2.0
    pb=(top==f'{LJ}.6')
    pc=abs(P[top]['nl'])>=1.5*abs(P[top]['ctrl'])
    null_ok=res[top]['ratio_nl']>res[top]['ratio_ctrl']
    out={'heads':res,'attn_profile_recent_newline':P,
         'ranked':ranked,'top':top,'base':base,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'null_ok':bool(null_ok),'runtime_s':time.time()-t0}
    print(f'\ntop head {top} ratio {res[top]["ratio_nl"]} | '
          f'recent-newline share nl {P[top]["nl"]} vs ctrl '
          f'{P[top]["ctrl"]}')
    for nm,v in (('a','a head reaches ratio >=2.0'),
                 ('b','the highest-ratio head is 12.6 (atlas bet)'),
                 ('c','its recent-newline share is >=1.5x at targets')):
        print(f"({nm}) {v}: {'HELD' if out['pred_'+nm] else 'FAILED'}")
    print(f"NULL (nl ratio > position-matched ctrl ratio): "
          f"{'ok' if null_ok else 'VIOLATED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
