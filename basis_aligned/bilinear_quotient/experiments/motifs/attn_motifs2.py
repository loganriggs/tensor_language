"""ATTENTION MOTIF CENSUS v2 -- v1 bucketed absolute pattern MASS and
found 155/162 heads 'other': unnormalized squared attention spreads
magnitude across the long tail, so mass fractions measure the noise
floor, not the function. v2 buckets the ARGMAX: for each head and query
position, where does the head look hardest? Buckets (priority order):
self / first-token / prev (j=q-1) / prev2 (j=q-2) / induction-target
(tok[j-1]==tok[q]) / token-match (tok[j]==tok[q]) / other. A head's
motif = its most common argmax bucket if >=0.25 of query positions,
else diffuse. Also reports top-1 concentration (share of |mass| on the
argmax). Null: content buckets recomputed on token-shuffled rows.
REGISTERED PREDICTIONS:
  (a) >=3 motif families (excluding other/diffuse) each present in >=2
      layers with >=5 heads total;
  (b) >=half of induction-target heads live in layers 3-5;
  (c) shuffle null: ind+match argmax fraction <= half of real for the
      ind-motif heads."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn_motifs2_results.json'
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
        for j,b in enumerate((4,5)):
            fracN[li,:,j]=(codeN==b).float().mean((0,2)).cpu()
        print(f'L{li} done',flush=True)
    motif=[]
    for li in range(18):
        for hd in range(9):
            p=frac[li,hd]
            b=int(p.argmax())
            mo=BUCKETS[b] if (p[b]>=0.25 and b!=6) else 'diffuse'
            motif.append((li,hd,mo,round(float(p[b]),3),
                          round(float(conc[li,hd]),3)))
    census={}
    for li,hd,mo,fr,cc in motif:
        census.setdefault(mo,[]).append((li,hd,fr,cc))
    for mo,lst in sorted(census.items(),key=lambda kv:-len(kv[1])):
        lys=sorted({li for li,_,_,_ in lst})
        print(f'{mo:8s}: {len(lst):3d} heads, layers {lys}')
    fams=[mo for mo,lst in census.items()
          if mo not in ('diffuse','other')
          and len({li for li,_,_,_ in lst})>=2]
    nfam=sum(len(census[mo]) for mo in fams)
    indh=[(li,hd) for li,hd,mo,_,_ in motif if mo=='ind']
    ind35=sum(1 for li,_ in indh if 3<=li<=5)
    realm=[float(frac[li,hd,4]+frac[li,hd,5]) for li,hd in indh]
    nullm=[float(fracN[li,hd].sum()) for li,hd in indh]
    rn=(sum(realm)/max(len(realm),1),sum(nullm)/max(len(nullm),1))
    pa=len(fams)>=3 and nfam>=5
    pb=(ind35>=max(1,len(indh)/2)) if indh else False
    pc=(rn[1]<=0.5*rn[0]) if indh else False
    out={'census':{mo:len(l) for mo,l in census.items()},
         'families_multilayer':fams,'ind_heads':indh,
         'ind_frac_real':round(rn[0],3),'ind_frac_null':round(rn[1],3),
         'motif_table':motif,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) >=3 multi-layer families, >=5 heads ({fams}): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) ind heads in L3-5: {ind35}/{len(indh)}: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffle null <= half ({rn[1]:.3f} vs {rn[0]:.3f}): "
          f"{'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
