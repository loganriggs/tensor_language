"""ATTENTION MOTIF CENSUS -- the cross-layer repertoire hypothesis: the
same attention pattern-function (look back 1 token; attend the induction
target; attend token matches; sink to first position) recurs at many
layers with different semantics bound by different inputs. If true, the
pattern side of all 18 attention layers compresses to a small motif
library plus per-head bindings.
Method: for every head (18 layers x 9 heads), compute its unnormalized
squared-attention pattern on window-A text and bucket its absolute mass:
self / first-token / induction-target (tok[j-1]==tok[i]) / token-match
(tok[j]==tok[i]) / offset-1 / offset-2 / offset-3..8 / other. A head's
motif = its dominant bucket if that bucket holds >=0.3 of mass, else
'diffuse'. Null for the content-defined buckets: recompute masks on
token-shuffled rows.
REGISTERED PREDICTIONS:
  (a) >=4 motif families each appear in >=2 different layers (the
      repertoire is real, not layer-idiosyncratic);
  (b) >=half of induction-target-motif heads live in layers 3-5 (matches
      the certified induction owners);
  (c) shuffle null: induction+match mass <= half of real for those heads;
  (d) full census reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn_motifs_results.json'
CA=300; NB=8
BUCKETS=['self','first','ind','match','off1','off2','off3_8','other']

@torch.no_grad()
def main():
    t0=time.time()
    mod=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod.apply_rotary_emb
    g=torch.Generator().manual_seed(0)
    # capture attn inputs for all layers over NB batches
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
    tt=torch.cat(toks)                      # (NB*4, T)
    tsh=torch.stack([r[torch.randperm(T,generator=g)] for r in tt.cpu()])\
        .to(DEV)
    def masks(tk):
        B=tk.shape[0]
        ii=torch.arange(T,device=DEV)
        sf=(ii[:,None]==ii[None,:]).expand(B,T,T)
        fr=(ii[None,None,:]==0).expand(B,T,T)
        mt=(tk[:,:,None]==tk[:,None,:])
        nd=torch.zeros(B,T,T,dtype=torch.bool,device=DEV)
        nd[:,:,1:]=(tk[:,:,None]==tk[:,None,:-1])
        off=ii[:,None]-ii[None,:]
        o1=(off==1).expand(B,T,T); o2=(off==2).expand(B,T,T)
        o38=((off>=3)&(off<=8)).expand(B,T,T)
        return sf,fr,nd,mt,o1,o2,o38
    MK=masks(tt); MKs=masks(tsh)
    prof=torch.zeros(18,9,8); profN=torch.zeros(18,9,2)
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
        tot=pat.sum((0,2,3)).clamp_min(1e-9)
        used=torch.zeros_like(pat,dtype=torch.bool)
        for bi,mk_ in enumerate(MK):
            mm=mk_[:,None].expand(-1,9,-1,-1)&mask&~used
            prof[li,:,bi]=(pat*mm).sum((0,2,3)).cpu()/tot.cpu()
            used|=mm
        prof[li,:,7]=1-prof[li,:,:7].sum(1)
        usedS=torch.zeros_like(pat,dtype=torch.bool)
        for j,mk_ in enumerate(MKs[:4]):
            mm=mk_[:,None].expand(-1,9,-1,-1)&mask&~usedS
            if j in (2,3):
                profN[li,:,j-2]=(pat*mm).sum((0,2,3)).cpu()/tot.cpu()
            usedS|=mm
        print(f'L{li} done',flush=True)
    motif=[]
    for li in range(18):
        for hd in range(9):
            p=prof[li,hd]
            b=int(p.argmax())
            motif.append((li,hd,BUCKETS[b] if p[b]>=0.3 else 'diffuse',
                          round(float(p[b]),3)))
    census={}
    for li,hd,mo,fr in motif:
        census.setdefault(mo,[]).append((li,hd,fr))
    for mo,lst in sorted(census.items(),key=lambda kv:-len(kv[1])):
        lys=sorted({li for li,_,_ in lst})
        print(f'{mo:8s}: {len(lst):3d} heads, layers {lys}')
    fams=[mo for mo,lst in census.items()
          if mo!='diffuse' and len({li for li,_,_ in lst})>=2]
    indh=[(li,hd) for li,hd,mo,_ in motif if mo=='ind']
    ind35=sum(1 for li,_ in indh if 3<=li<=5)
    realm=[]; nullm=[]
    for li,hd in indh:
        realm.append(float(prof[li,hd,2]+prof[li,hd,3]))
        nullm.append(float(profN[li,hd].sum()))
    rn=(sum(realm)/max(len(realm),1),sum(nullm)/max(len(nullm),1))
    pa=len(fams)>=4
    pb=(ind35>=max(1,len(indh)/2)) if indh else False
    pc=(rn[1]<=0.5*rn[0]) if indh else False
    out={'census':{mo:len(l) for mo,l in census.items()},
         'families_multilayer':fams,'ind_heads':indh,
         'ind_mass_real':round(rn[0],3),'ind_mass_null':round(rn[1],3),
         'motif_table':motif,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) >=4 multi-layer families ({fams}): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) ind heads in L3-5: {ind35}/{len(indh)}: "
          f"{'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffle null <= half ({rn[1]:.3f} vs {rn[0]:.3f}): "
          f"{'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
