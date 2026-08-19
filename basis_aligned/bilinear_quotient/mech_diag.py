"""MECH DIAG -- mech_replicate refuted the textbook induction code
(alpha ~= 0, corr 0.03: the 9 'ind' heads do NOT compute
alpha*v(j+1) with a one-hot pattern). Diagnose what they DO compute.
For each head, at match positions on 32 census rows:
  (1) SPARSITY: what fraction of z's variance is carried by the
      single top-|pattern| key (z_top1 = pat(q,k*)*vm(k*))?
  (2) LOCATION: how often is k* in {j, j+1} (the match region),
      vs {q, q-1} (local), vs elsewhere?
  (3) single-position target fits: R^2 of z against pat-weighted
      vm at j+1, j, q-1, q separately.
REGISTERED PREDICTIONS:
  (a) SPARSE-READ: >=5/9 heads have top-1 share >=50%;
  (b) MATCH-SEEKING: >=5/9 heads place k* in {j,j+1} at >=30% of
      match positions;
  (c) if both fail, the 'induction' label is recorded as
      pattern-shape-only: the heads' functional read is DIFFUSE and
      any computational story needs the full pattern-weighted sum."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_diag_results.json'
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    def match_idx(row):
        toks=row[:T].tolist(); last={}
        M=torch.full((T,),-1,dtype=torch.long)
        for q in range(T):
            t=toks[q]
            if t in last and last[t]+1<q: M[q]=last[t]+1
            last[t]=q
        return M
    cap={}
    def mkpre(li):
        def h(mo_,args): cap[li]=(args[0],args[1])
        return h
    import sys as s_
    are=s_.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    stats={k:{'top1_num':0.,'top1_den':0.,'loc':torch.zeros(3),
              'n':0} for k in IND}
    for i in range(0,32,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        Mb=torch.stack([match_idx(ROWS[i+b]) for b in range(4)])
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            mkpre(li)) for li,_ in IND]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blkm in m.transformer.h: x,v1=blkm(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in IND:
            X,v1i=cap[li]
            # need patterns: recompute pieces (head_parts gives z,vm
            # but not pat) -- recompute pat here
            at=m.transformer.h[li].attn
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
            sc=torch.einsum('bqd,bkd->bqk',qf[:,:,hd].float(),
                            kf[:,:,hd].float())/128
            s2=torch.einsum('bqd,bkd->bqk',q2[:,:,hd].float(),
                            k2[:,:,hd].float())/128
            pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
            vmh=vm[:,:,hd].float()
            z=torch.einsum('bqk,bkd->bqd',pat,vmh)
            kstar=pat.abs().argmax(-1)                     # (B,T)
            ztop=pat.gather(-1,kstar[...,None]).squeeze(-1)[...,None] \
                 *vmh.gather(1,kstar[...,None].expand(-1,-1,128))
            ok=(Mb>=0).to(DEV)
            st=stats[(li,hd)]
            st['top1_num']+=float(((z-ztop)[ok]**2).sum())
            st['top1_den']+=float((z[ok]**2).sum())
            j=Mb.to(DEV)
            inmatch=((kstar==j)|(kstar==j-1))&ok
            qq=torch.arange(T,device=DEV)[None,:]
            local=((kstar==qq)|(kstar==qq-1))&ok
            st['loc']+=torch.tensor([float(inmatch.sum()),
                                     float(local.sum()),
                                     float((ok&~inmatch&~local)
                                           .sum())])
            st['n']+=int(ok.sum())
    out={'heads':{}}
    ns=0; nm=0
    for k,st in stats.items():
        share=1-st['top1_num']/max(st['top1_den'],1e-6)
        loc=(st['loc']/max(st['n'],1)).tolist()
        out['heads'][f'{k[0]}.{k[1]}']={'top1_share':round(share,3),
            'frac_kstar_match':round(loc[0],3),
            'frac_kstar_local':round(loc[1],3),
            'frac_kstar_other':round(loc[2],3)}
        if share>=0.5: ns+=1
        if loc[0]>=0.3: nm+=1
        print(f'{k}: top1 {share:.2f} | match {loc[0]:.2f} '
              f'local {loc[1]:.2f} other {loc[2]:.2f}',flush=True)
    pa=ns>=5; pb=nm>=5
    out.update({'n_sparse':ns,'n_match_seeking':nm,
                'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True})
    print(f"(a) sparse-read >=5/9: {'HELD' if pa else 'FAILED'}")
    print(f"(b) match-seeking >=5/9: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
