"""ATTENTION MOTIF CENSUS v3 -- v2 found prev (27 heads, 11 layers) and
self (51 heads, 16 layers) but ZERO induction-target heads, despite
certified induction ownership at attn3-5. Suspected dilution: induction
heads fire only on ELIGIBLE queries (those with a real match earlier in
context), a minority of positions. v3 scores ind/match motifs
CONDITIONALLY: P(argmax hits the bucket | query eligible), head counts
as ind-motif if conditional fraction >= 0.4 (>=500 eligible samples).
REGISTERED PREDICTIONS:
  (a) >=1 conditional-ind head exists in layers 3-5 (if this FAILS the
      pattern-level and ownership-level pictures are in real tension --
      induction ownership without argmax-visible induction patterns);
  (b) prev and self remain multi-layer families;
  (c) shuffle null conditional fraction <= half of real."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn_motifs3_results.json'
CA=300; NB=8
BUCKETS=['self','first','prev','prev2','ind','match','other']

@torch.no_grad()
def main():
    t0=time.time()
    mod=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod.apply_rotary_emb
    g=torch.Generator().manual_seed(0)
    Xs={li:[] for li in range(18)}
    hs=[]
    for li in range(18):
        def mk(li=li):
            def h(mo,args):
                Xs[li].append(args[0].detach())
            return h
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    toks=[]
    for i in range(CA,CA+NB*4,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
        toks.append(bb[:,:-1])
    for h in hs: h.remove()
    T=256
    tt=torch.cat(toks)
    tsh=torch.stack([r[torch.randperm(T,generator=g)] for r in tt.cpu()])\
        .to(DEV)
    def bucketize(jm,tk):
        # jm: (B,H,T) argmax key position per query; returns (B,H,T) codes
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
    frac=torch.zeros(18,9,7); conc=torch.zeros(18,9)
    fracN=torch.zeros(18,9,2)
    condI=torch.zeros(18,9); condN=torch.zeros(18,9); nel={}
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
        sc=torch.einsum('bqhd,bkhd->bhqk',q.float(),k.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
        pat=(sc*sc2).abs()
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        pat=pat*mask
        jm=pat.argmax(3)                      # (B,H,T)
        top=pat.max(3).values
        tot=pat.sum(3).clamp_min(1e-9)
        conc[li]=(top/tot).mean((0,2)).cpu()
        code=bucketize(jm,tt)
        codeN=bucketize(jm,tsh)
        for b in range(7):
            frac[li,:,b]=(code==b).float().mean((0,2)).cpu()
        # eligibility: query q has some j in [1,q-1] with tok[j-1]==tok[q]
        def elig(tk):
            e=torch.zeros(tk.shape[0],T,dtype=torch.bool,device=DEV)
            for b_ in range(tk.shape[0]):
                seen={}
                row=tk[b_].tolist()
                for qq in range(2,T):
                    e[b_,qq]=row[qq] in seen
                    seen[row[qq-1]]=True
            return e
        if li==0:
            global EL,ELs
            EL=elig(tt); ELs=elig(tsh)
        el=EL[:,None,:].expand(-1,9,-1)
        els=ELs[:,None,:].expand(-1,9,-1)
        ci=((code==4)|(code==5))&el
        cin=((codeN==4)|(codeN==5))&els
        condI[li]=(ci.float().sum((0,2))/el.float().sum((0,2))
                   .clamp_min(1)).cpu()
        condN[li]=(cin.float().sum((0,2))/els.float().sum((0,2))
                   .clamp_min(1)).cpu()
        nel[li]=float(el[:,0,:].float().sum())
        print(f'L{li} done',flush=True)
    motif=[]
    for li in range(18):
        for hd in range(9):
            p=frac[li,hd]
            if condI[li,hd]>=0.4 and nel[li]>=500:
                mo='ind'; fr=float(condI[li,hd])
            else:
                b=int(p.argmax())
                mo=BUCKETS[b] if (p[b]>=0.25 and b!=6) else 'diffuse'
                fr=float(p[b])
            motif.append((li,hd,mo,round(fr,3)))
    census={}
    for li,hd,mo,fr in motif:
        census.setdefault(mo,[]).append((li,hd,fr))
    for mo,lst in sorted(census.items(),key=lambda kv:-len(kv[1])):
        lys=sorted({li for li,_,_ in lst})
        print(f'{mo:8s}: {len(lst):3d} heads, layers {lys}')
    indh=[(li,hd) for li,hd,mo,_ in motif if mo=='ind']
    ind35=sum(1 for li,_ in indh if 3<=li<=5)
    realm=[float(condI[li,hd]) for li,hd in indh]
    nullm=[float(condN[li,hd]) for li,hd in indh]
    rn=(sum(realm)/max(len(realm),1),sum(nullm)/max(len(nullm),1))
    top5=sorted(((float(condI[li,hd]),li,hd) for li in range(18)
                 for hd in range(9)),reverse=True)[:8]
    print('top conditional-ind heads:',
          [(li,hd,round(c,3)) for c,li,hd in top5])
    pa=ind35>=1
    pb=(len({li for li,_,_ in census.get('prev',[])})>=2
        and len({li for li,_,_ in census.get('self',[])})>=2)
    pc=(rn[1]<=0.5*rn[0]) if indh else False
    out={'census':{mo:len(l) for mo,l in census.items()},
         'ind_heads':indh,'ind_cond_real':round(rn[0],3),
         'ind_cond_null':round(rn[1],3),
         'top_ind':[(li,hd,round(c,3)) for c,li,hd in top5],
         'motif_table':motif,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) conditional-ind head in L3-5 ({ind35}/{len(indh)}): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) prev+self multi-layer: {'HELD' if pb else 'FAILED'}")
    print(f"(c) null <= half ({rn[1]:.3f} vs {rn[0]:.3f}): "
          f"{'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
