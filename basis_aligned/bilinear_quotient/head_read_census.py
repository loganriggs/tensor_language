"""HEAD READ CENSUS -- successor to the failed pattern-matrix
dictionary (364): measure FUNCTION (where each head's read mass
actually lands), not form. For all 162 heads over 16 census rows:
top-1 pattern share of output variance, and the location class of
the top key per query: self (k=q), prev (k=q-1), first (k<=1),
match (k in {j,j+1}), newline-anchor, other.
REGISTERED PREDICTIONS:
  (a) functional agreement: for heads the motif census labeled
      self/prev/first, the modal location class matches the label
      >=70% of heads;
  (b) >=100/162 heads have top-1 share >=0.4 (sparse reads are the
      norm, not a motif-head specialty);
  (c) per-head profiles written (the functional dictionary)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_read_census_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:16]
    import sys as s_
    are=s_.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    def row_info(row):
        toks=row[:T].tolist(); last={}
        J=torch.full((T,),-1,dtype=torch.long)
        for q in range(T):
            t=toks[q]
            if t in last: J[q]=last[t]
            last[t]=q
        NLpos=torch.tensor([chr(10) in cl.d1(t) for t in toks])
        return J,NLpos
    INFO=[row_info(ROWS[r]) for r in range(16)]
    cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    acc={ (li,hd):{'num':0.,'den':0.,'loc':torch.zeros(6),'n':0}
          for li in range(18) for hd in range(9)}
    for i in range(0,16,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li in range(18)]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for li,blkm in enumerate(m.transformer.h):
            x,v1=blkm(x,v1,x0)
            at=m.transformer.h[li].attn
            X,v1i=cap[li]
            Bb=4
            v=at.c_v(X).view(Bb,T,9,128)
            vm=(1-at.lamb)*v+at.lamb*(v1i.view_as(v)
                                      if v1i is not None else v)
            cos,sin=at.rotary(at.c_q(X).view(Bb,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(Bb,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(Bb,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(Bb,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(Bb,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
            pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,
                           vm.float())
            ks=pat.abs().argmax(-1)                    # (B,H,T)
            for hd in range(9):
                p1=pat[:,hd]; k1=ks[:,hd]
                w=p1.gather(-1,k1[...,None]).squeeze(-1)
                vv=vm[:,:,hd].float().gather(
                    1,k1[...,None].expand(-1,-1,128))
                ztop=w[...,None]*vv
                a=acc[(li,hd)]
                a['num']+=float(((z[:,hd]-ztop)**2).sum())
                a['den']+=float((z[:,hd]**2).sum())
                qq=torch.arange(T,device=DEV)[None,:]
                for b in range(Bb):
                    J,NLp=INFO[i+b]
                    J=J.to(DEV); NLp=NLp.to(DEV)
                    kb=k1[b]
                    loc=torch.full((T,),5,device=DEV)
                    loc[NLp[kb.clamp(0,T-1).cpu()].to(DEV)]=4
                    okj=J>=0
                    loc[okj&((kb==J)|(kb==(J+1).clamp(max=T-1)))]=3
                    loc[kb<=1]=2
                    loc[kb==qq[0]-1]=1
                    loc[kb==qq[0]]=0
                    for c in range(6):
                        a['loc'][c]+=float((loc==c).sum())
                    a['n']+=T
        for h in hs: h.remove()
        print(f'rows {i}-{i+3} done',flush=True)
    LOCN=['self','prev','first','match','nl','other']
    prof={}
    n04=0
    for (li,hd),a in acc.items():
        share=1-a['num']/max(a['den'],1e-6)
        lp=(a['loc']/max(a['n'],1))
        prof[f'{li}.{hd}']={'top1_share':round(share,3),
            'modal':LOCN[int(lp.argmax())],
            'profile':{LOCN[c]:round(float(lp[c]),3)
                       for c in range(6)}}
        if share>=0.4: n04+=1
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    lab={'self':'self','prev':'prev','first':'first','ind':'match'}
    agree=0; tot=0; dis=[]
    for li,hd,mo,fr in mt:
        if mo in ('self','prev','first'):
            tot+=1
            if prof[f'{li}.{hd}']['modal']==lab[mo]: agree+=1
            else: dis.append((f'{li}.{hd}',mo,
                              prof[f'{li}.{hd}']['modal']))
    pa=tot>0 and agree/tot>=0.7
    pb=n04>=100
    out={'profiles':prof,'agreement':f'{agree}/{tot}',
         'n_top1_ge_04':n04,'disagreements':dis[:15],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True}
    print(f'functional agreement {agree}/{tot} | top1>=0.4: '
          f'{n04}/162')
    print(f"(a) agreement >=70%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=100 sparse: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
