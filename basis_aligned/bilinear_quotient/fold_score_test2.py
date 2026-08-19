"""FOLD SCORE TEST v2 -- 388: v1 dropped rotary and still hit 45%
on head 2.5. Rotary is a deterministic per-position rotation, so
include it: rotate each position's fold-code projections by its own
rotary angle (weights + tokens + positions only, no model forward).
REGISTERED PREDICTIONS:
  (a) early band (1,4),(2,5),(3,5): hit rate >=40% with rotary;
  (b) early band rank corr >=0.4;
  (c) deep heads reported (expected low: context in the match
      code, 387)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'fold_score_test2_results.json'
HEADS=[(1,4),(2,5),(3,5),(5,5),(8,4)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    # fold codes for tokens present in rows
    uniq=ROWS[:,:257].reshape(-1).unique()
    E=F.rms_norm(m.transformer.wte(uniq.to(DEV)),(D,))
    code=m.transformer.h[0].mlp(E).float()          # m0 fold
    t2i={int(t):i for i,t in enumerate(uniq.tolist())}
    res={}
    cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    for li,hd in HEADS:
        res[f'{li}.{hd}']={'hit':0,'n':0,'corr':[]}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li,_ in HEADS]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X,v1i=cap[li]
            a9,b9=hd*128,(hd+1)*128
            # fold-side projections for this head
            cq=F.rms_norm(code@at.c_q.weight.float()[a9:b9].T,(128,))
            ck=F.rms_norm(code@at.c_k.weight.float()[a9:b9].T,(128,))
            cq2=F.rms_norm(code@at.c_q2.weight.float()[a9:b9].T,(128,))
            ck2=F.rms_norm(code@at.c_k2.weight.float()[a9:b9].T,(128,))
            # real pattern
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())) \
                *torch.tril(torch.ones(T,T,device=DEV))
            for b in range(4):
                toks=ROWS[i+b,:T].tolist(); last={}
                ids=torch.tensor([t2i[t] for t in toks],device=DEV)
                fq1=cq[ids]; fk1=ck[ids]; fq2=cq2[ids]; fk2=ck2[ids]
                # rotary at actual positions (deterministic)
                po=torch.arange(T,device=DEV)
                fq1=are(fq1[None,:,None,:],cos[:,:T],sin[:,:T])[0,:,0]
                fk1=are(fk1[None,:,None,:],cos[:,:T],sin[:,:T])[0,:,0]
                fq2=are(fq2[None,:,None,:],cos[:,:T],sin[:,:T])[0,:,0]
                fk2=are(fk2[None,:,None,:],cos[:,:T],sin[:,:T])[0,:,0]
                fpat=(fq1@fk1.T)*(fq2@fk2.T)
                fpat=fpat*torch.tril(torch.ones(T,T,device=DEV))
                for q in range(T):
                    t=toks[q]
                    ism=t in last and last[t]+1<q
                    last[t]=q if t not in last else q
                    last[t]=q
                    if not ism or q<8: continue
                    st=res[f'{li}.{hd}']
                    kreal=int(pat[b,q,:q].abs().argmax())
                    kfold=int(fpat[q,:q].abs().argmax())
                    st['hit']+=int(kreal==kfold); st['n']+=1
                    pr=pat[b,q,:q].abs(); pf=fpat[q,:q].abs()
                    if q>=16:
                        rr=pr.argsort().argsort().float()
                        rf=pf.argsort().argsort().float()
                        st['corr'].append(float(torch.corrcoef(
                            torch.stack([rr,rf]))[0,1]))
        print(f'batch {i} done',flush=True)
    outj={}
    for k,st in res.items():
        outj[k]={'hit_rate':round(st['hit']/max(st['n'],1),3),
                 'rank_corr':round(sum(st['corr'])/max(len(st['corr']),1),3),
                 'n':st['n']}
        print(f"{k}: hit {outj[k]['hit_rate']} corr "
              f"{outj[k]['rank_corr']}",flush=True)
    early=['1.4','2.5','3.5']
    pa=all(outj[k]['hit_rate']>=0.4 for k in early)
    pb=all(outj[k]['rank_corr']>=0.4 for k in early)
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f"(a) early hit>=40%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) early corr>=0.4: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
