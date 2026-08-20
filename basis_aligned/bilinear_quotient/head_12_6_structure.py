"""HEAD 12.6 STRUCTURE TEST -- 490: the token-class profile (489)
refutes salience and suggests something better. At match positions
head 12.6's distant reads are enriched for PUNCTUATION (2.33x),
capitalised tokens (1.79x), digits (1.33x) and newlines (1.27x),
and DEPLETED for ordinary word content (subword 0.38x, space-word
0.68x) and for rare tokens (0.69x -- the salience hypothesis is
dead). Its layer-mate 12.3 is the mirror image: capitalised 3.72x
and subword 1.4x, but punctuation 0.34x and newline 0.09x.
So 12.6 looks like a LONG-RANGE STRUCTURE READER -- it scans back
to punctuation, digits, capitals and line breaks, the anchors of
layout and clause boundaries, and ignores prose content. If that
is what it does, its contribution should depend on how structured
the text is.
Test functionally: split fresh FineWeb rows into quartiles by
structural density (fraction of punctuation and newline tokens),
and measure 12.6's 4-token-window damage at match positions in
each quartile, with 12.3 as control.
REGISTERED PREDICTIONS:
  (a) STRUCTURE GRADIENT: 12.6's match-position window cost in
      the top structural quartile is >= 2x its cost in the bottom
      quartile;
  (b) CONTROL: head 12.3 shows no such gradient (< 1.5x);
  (c) per-quartile numbers reported for both heads."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_12_6_structure_results.json'
NFRESH=48
HEADS=[6,3]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    def isstruct(t):
        s=cl.d1(int(t)); st=s.strip()
        return (bool(st) and not any(c.isalnum() for c in st)) \
            or (chr(10) in s)
    dens=torch.tensor([sum(isstruct(int(fresh[r,q]))
                           for q in range(T))/T
                       for r in range(NFRESH)])
    qs=torch.quantile(dens,torch.tensor([0.25,0.5,0.75]))
    bucket=torch.bucketize(dens,qs)
    print('structural density quartile sizes:',
          [int((bucket==b).sum()) for b in range(4)],
          '| median density',round(float(dens.median()),3),
          flush=True)
    def run(hd,active):
        acc={b:[0.0,0] for b in range(4)}
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if active:
                def fh(mo_,args,o_,hd=hd):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=r2(at.c_q),r2(at.c_k)
                    q2,k2=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    ar=torch.arange(T,device=DEV)
                    win=tril*((ar[:,None]-ar[None,:])<K).float()
                    z=torch.einsum('bhqk,bkhd->bhqd',
                                   (sc*sc2)*tril,vm.float())
                    zw=torch.einsum('bhqk,bkhd->bhqd',
                                    (sc*sc2)*win,vm.float())
                    z[:,hd]=zw[:,hd]
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            for b in range(B):
                r=i+b
                mk=torch.zeros(T,dtype=torch.bool)
                toks=fresh[r,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk[q]=True
                    last[t]=q
                bq=int(bucket[r])
                acc[bq][0]+=float(ce[b][mk].sum())
                acc[bq][1]+=int(mk.sum())
            for h in hs: h.remove()
        return {b:acc[b][0]/max(acc[b][1],1) for b in range(4)}
    base=run(6,False)
    out={}
    for hd in HEADS:
        cur=run(hd,True)
        d={b:round(cur[b]-base[b],4) for b in range(4)}
        out[f'12.{hd}']=d
        print(f'12.{hd}: by structural quartile {d}',flush=True)
    a=out['12.6']; c=out['12.3']
    ra=a[3]/max(a[0],1e-6); rc=c[3]/max(c[0],1e-6)
    pa=ra>=2.0
    pb=rc<1.5
    out.update({'ratio_12_6_top_over_bottom':round(ra,2),
                'ratio_12_3_top_over_bottom':round(rc,2),
                'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
                'runtime_s':time.time()-t0})
    print(f'ratios: 12.6 {ra:.2f}x, 12.3 {rc:.2f}x')
    for nm,v in (('a','12.6 shows a >=2x structure gradient'),
                 ('b','12.3 shows none (<1.5x)'),
                 ('c','quartile numbers reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
