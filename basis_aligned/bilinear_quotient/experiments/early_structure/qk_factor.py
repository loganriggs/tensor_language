"""QK FACTORIZATION -- bilin18 attention is a DOUBLE bilinear pattern:
pattern(q,j) = (s1/128)*(s2/128) with s1 = <Rq(x_q), Rk(x_j)> and s2
from a second, independent QK pair. The product is an AND gate: two
bilinear conditions must both fire. Hypothesis (user-registered): named
motifs FACTOR -- one score set carries the positional selector (e.g.
offset-1), the other acts as a broad content gate. Method: compute s1
and s2 argmax profiles SEPARATELY per head (same buckets as the census:
self/first/prev/prev2/ind/match, ind conditioned on eligible queries),
plus a flatness measure per set.
REGISTERED PREDICTIONS (on the 283-census heads):
  (a) >=60% of prev-motif heads are factored: one set's offset-1 argmax
      fraction >= 2x the other's and >= 0.5;
  (b) >=6 of 9 ind-motif heads carry the conditional ind signal >= 2x
      in one set;
  (c) among factored prev heads, the position-carrying set is the SAME
      set (s1 or s2) for >= 70% (a global convention exists) -- registered
      with honest uncertainty; failure = per-head convention;
  (d) full (s1-motif, s2-motif) joint census reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'qk_factor_results.json'
CA=300; NB=8
BUCKETS=['self','first','prev','prev2','ind','match','other']

@torch.no_grad()
def main():
    t0=time.time()
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    fam={}
    for li,hd,mo,fr in mt: fam[(li,hd)]=mo
    mod=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod.apply_rotary_emb
    Xs={li:[] for li in range(18)}
    hs=[]
    for li in range(18):
        def mk(li=li):
            def h(mo_,args): Xs[li].append(args[0].detach())
            return h
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    for i in range(CA,CA+NB*4,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    T=256
    tt=torch.cat([FW[i:i+4,:256] for i in range(CA,CA+NB*4,4)]).to(DEV)
    def elig(tk):
        e=torch.zeros(tk.shape[0],T,dtype=torch.bool,device=DEV)
        for b_ in range(tk.shape[0]):
            seen=set(); row=tk[b_].tolist()
            for qq in range(2,T):
                e[b_,qq]=row[qq] in seen
                seen.add(row[qq-1])
        return e
    EL=elig(tt)
    def bucketize(jm,tk):
        B=tk.shape[0]
        q=torch.arange(T,device=DEV)[None,None,:].expand_as(jm)
        code=torch.full_like(jm,6)
        tj=torch.gather(tk[:,None,:].expand(-1,9,-1),2,jm)
        tjm1=torch.gather(tk[:,None,:].expand(-1,9,-1),2,
                          (jm-1).clamp_min(0))
        tq=tk[:,None,:].expand(-1,9,-1)
        code[(tj==tq)]=5
        code[(jm>=1)&(tjm1==tq)]=4
        code[jm==q-2]=3
        code[jm==q-1]=2
        code[jm==0]=1
        code[jm==q]=0
        return code
    res={}
    for li in range(18):
        at=m.transformer.h[li].attn
        X=torch.cat(Xs[li]); Xs[li]=None
        B=X.shape[0]
        q=at.c_q(X).view(B,T,9,128); k=at.c_k(X).view(B,T,9,128)
        q2=at.c_q2(X).view(B,T,9,128); k2=at.c_k2(X).view(B,T,9,128)
        cos,sin=at.rotary(q)
        q=F.rms_norm(q,(128,)); k=F.rms_norm(k,(128,))
        q,k=are(q,cos,sin),are(k,cos,sin)
        q2=F.rms_norm(q2,(128,)); k2=F.rms_norm(k2,(128,))
        q2,k2=are(q2,cos,sin),are(k2,cos,sin)
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        el=EL[:,None,:].expand(-1,9,-1)
        for si,(qa,ka) in enumerate(((q,k),(q2,k2))):
            sc=torch.einsum('bqhd,bkhd->bhqk',qa.float(),ka.float())
            sc=(sc.abs())*mask
            jm=sc.argmax(3)
            code=bucketize(jm,tt)
            fr=torch.stack([(code==b).float().mean((0,2))
                            for b in range(7)],1)   # (H,7)
            ci=(((code==4)|(code==5))&el).float().sum((0,2))/\
                el.float().sum((0,2)).clamp_min(1)
            top=(sc.max(3).values/sc.sum(3).clamp_min(1e-9)).mean((0,2))
            for hd in range(9):
                res.setdefault((li,hd),{})[f's{si+1}']={
                    'frac':[round(float(x),3) for x in fr[hd]],
                    'ind_cond':round(float(ci[hd]),3),
                    'conc':round(float(top[hd]),3)}
        print(f'L{li} done',flush=True)
    prevheads=[(li,hd) for (li,hd),mo in fam.items() if mo=='prev']
    indheads=[(li,hd) for (li,hd),mo in fam.items() if mo=='ind']
    nfac=0; posset=[]
    for li,hd in prevheads:
        f1=res[(li,hd)]['s1']['frac'][2]; f2=res[(li,hd)]['s2']['frac'][2]
        hi,lo=max(f1,f2),min(f1,f2)
        if hi>=0.5 and hi>=2*max(lo,1e-3):
            nfac+=1; posset.append(1 if f1>f2 else 2)
    nfi=0
    for li,hd in indheads:
        c1=res[(li,hd)]['s1']['ind_cond']; c2=res[(li,hd)]['s2']['ind_cond']
        if max(c1,c2)>=2*max(min(c1,c2),1e-3): nfi+=1
    conv=max(posset.count(1),posset.count(2))/max(len(posset),1)
    pa=nfac>=0.6*len(prevheads)
    pb=nfi>=6
    pc=conv>=0.7
    out={'per_head':{f'L{li}h{hd}':v for (li,hd),v in res.items()},
         'prev_factored':nfac,'n_prev':len(prevheads),
         'ind_factored':nfi,'n_ind':len(indheads),
         'position_set_convention':round(conv,2),
         'posset_counts':{'s1':posset.count(1),'s2':posset.count(2)},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'prev factored {nfac}/{len(prevheads)} | ind factored '
          f'{nfi}/{len(indheads)} | position-set convention {conv:.2f} '
          f'(s1 {posset.count(1)} vs s2 {posset.count(2)})')
    print(f"(a) >=60% prev factored: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=6/9 ind factored: {'HELD' if pb else 'FAILED'}")
    print(f"(c) global convention >=70%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
