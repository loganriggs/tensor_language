"""RELAY CLOSURE -- 399: a4's channel into 5.5 is 95% a relay of
the mlp-ladder code. Close the remaining two heads under the same
one-code story.
Head 2.5 (channel = a1): arms ladder / a1_real (sanity 1.0) /
  a1_ladval (real patterns, values from ladder residual).
Head 8.4 (diffuse channel = a5,a6,a7): arms ladder / comb_real
  (all three real writes inserted) / comb_ladval (each recomputed
  with pure-ladder values) / comb_iter (values from the growing
  relay-augmented chain -- nested relays) / comb_shuf (values from
  a shuffled row's ladder residual -- null control).
REGISTERED PREDICTIONS:
  (a) 2.5 a1_ladval >= 0.95 (relay holds at depth 1);
  (b) 8.4 comb_real >= 0.75 (the three layers jointly close most
      of the gap);
  (c) 8.4 comb_iter recovers >= 80% of comb_real's lift over
      ladder (iterated relay of one code);
  (d) null: comb_shuf recovers < 25% of comb_real's lift (the
      relay carries position-specific code, not generic
      statistics)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'relay_closure_results.json'
NR=16
H25=(2,5); H84=(8,4); AJ84=(5,6,7)

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    res={'2.5':{a:{'hit':0,'n':0} for a in
                ('ladder','a1_real','a1_ladval')},
         '8.4':{a:{'hit':0,'n':0} for a in
                ('ladder','comb_real','comb_ladval','comb_iter',
                 'comb_shuf')}}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
        aout={}; mout={}; attin={}; pre={}
        hs=[]
        for lj in range(8):
            def ah(mo,i_,o_,lj=lj): aout[lj]=o_[0].detach().float()
            def mh(mo,i_,o_,lj=lj): mout[lj]=o_.detach().float()
            def phj(mo_,args,lj=lj):
                pre[lj]=(args[0],args[1])
            hs.append(m.transformer.h[lj].attn
                      .register_forward_hook(ah))
            hs.append(m.transformer.h[lj].mlp
                      .register_forward_hook(mh))
            hs.append(m.transformer.h[lj].attn
                      .register_forward_pre_hook(phj))
        for li,hd in (H25,H84):
            def ph(mo_,args,li=li): attin[li]=args[0]
            hs.append(m.transformer.h[li].attn
                      .register_forward_pre_hook(ph))
        x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        B=4
        def a_write(j,vsrc):
            at=m.transformer.h[j].attn
            Xj,v1j=pre[j]
            v=at.c_v(vsrc).view(B,T,9,128)
            vm=(1-at.lamb)*v+at.lamb*v1j.view_as(v)
            cos,sin=at.rotary(at.c_q(Xj).view(B,T,9,128))
            qf=F.rms_norm(at.c_q(Xj).view(B,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(Xj).view(B,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(Xj).view(B,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(Xj).view(B,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            patm=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',patm,vm.float())
            return at.c_proj(z.transpose(1,2).contiguous()
                             .view(B,T,-1).to(Xj.dtype)).float()
        def chain(upto,inserts,itermode=False,shuf=False):
            """inserts: set of blocks j whose a_j write is added.
            itermode: values from the growing chain; else values
            from the pure ladder chain (shuf: row-shuffled)."""
            xr=E.clone(); xl=E.clone()
            for lj in range(upto):
                blk=m.transformer.h[lj]
                lam=blk.lambdas.detach().float()
                xr=lam[0]*xr+lam[1]*E
                xl=lam[0]*xl+lam[1]*E
                if lj in inserts:
                    if inserts[lj]=='real': w=aout[lj]
                    else:
                        src=xr if itermode else xl
                        Xs=F.rms_norm(src,(D,)) \
                            .to(m.transformer.wte.weight.dtype)
                        if shuf: Xs=Xs[torch.tensor([1,2,3,0])]
                        w=a_write(lj,Xs)
                    xr=xr+w
                xr=xr+mout[lj]; xl=xl+mout[lj]
            lam=m.transformer.h[upto].lambdas.detach().float()
            return F.rms_norm(lam[0]*xr+lam[1]*E,(D,))
        codes={'2.5':{},'8.4':{}}
        codes['2.5']['ladder']=chain(2,{})
        codes['2.5']['a1_real']=chain(2,{1:'real'})
        codes['2.5']['a1_ladval']=chain(2,{1:'lad'})
        c84={j:'lad' for j in AJ84}
        codes['8.4']['ladder']=chain(8,{})
        codes['8.4']['comb_real']=chain(8,{j:'real' for j in AJ84})
        codes['8.4']['comb_ladval']=chain(8,c84)
        codes['8.4']['comb_iter']=chain(8,c84,itermode=True)
        codes['8.4']['comb_shuf']=chain(8,c84,shuf=True)
        for (li,hd),key in ((H25,'2.5'),(H84,'8.4')):
            at=m.transformer.h[li].attn
            X=attin[li]
            a9,b9=hd*128,(hd+1)*128
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            tril=torch.tril(torch.ones(T,T,device=DEV))
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float()))*tril
            for a,c in codes[key].items():
                fq1=F.rms_norm(c@at.c_q.weight.float()
                               [a9:b9].T,(128,))
                fk1=F.rms_norm(c@at.c_k.weight.float()
                               [a9:b9].T,(128,))
                fq2=F.rms_norm(c@at.c_q2.weight.float()
                               [a9:b9].T,(128,))
                fk2=F.rms_norm(c@at.c_k2.weight.float()
                               [a9:b9].T,(128,))
                fq1=are(fq1[:,:,None],cos,sin)[:,:,0]
                fk1=are(fk1[:,:,None],cos,sin)[:,:,0]
                fq2=are(fq2[:,:,None],cos,sin)[:,:,0]
                fk2=are(fk2[:,:,None],cos,sin)[:,:,0]
                fpat=(torch.einsum('bqd,bkd->bqk',fq1,fk1)
                      *torch.einsum('bqd,bkd->bqk',fq2,fk2))*tril
                for b in range(4):
                    toks=ROWS[i+b,:T].tolist(); last={}
                    for q in range(T):
                        t=toks[q]
                        ism=t in last and last[t]+1<q
                        last[t]=q
                        if not ism or q<8: continue
                        st=res[key][a]
                        kreal=int(pat[b,q,:q].abs().argmax())
                        kfold=int(fpat[b,q,:q].abs().argmax())
                        st['hit']+=int(kreal==kfold); st['n']+=1
        print(f'batch {i} done',flush=True)
    outj={k:{a:round(st['hit']/max(st['n'],1),3)
             for a,st in d.items()} for k,d in res.items()}
    for k,d in outj.items(): print(k,d,flush=True)
    pa=outj['2.5']['a1_ladval']>=0.95
    b84=outj['8.4']; lift=b84['comb_real']-b84['ladder']
    pb=b84['comb_real']>=0.75
    pc=(b84['comb_iter']-b84['ladder'])>=0.8*max(lift,1e-6)
    pd=(b84['comb_shuf']-b84['ladder'])<0.25*max(lift,1e-6)
    out={'arms':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'pred_d':bool(pd)}
    for nm,v in (('a','2.5 a1_ladval >=0.95'),
                 ('b','8.4 comb_real >=0.75'),
                 ('c','8.4 iter relay >=80% of lift'),
                 ('d','shuffled null <25% of lift')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
