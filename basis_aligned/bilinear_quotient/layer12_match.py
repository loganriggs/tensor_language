"""LAYER 12 AT MATCH -- 487: the depth sweeps found exactly one
layer with a genuinely match-specific long-range cost. Restricting
layer 12 to a 4-token read window costs +0.2094 at match positions
against +0.0207 elsewhere -- a tenfold ratio -- and unlike layer 5
it has nothing to do with the position-0 sink (484). It is also
nowhere near the induction band (layers 1-8), whose heads are
already accounted for (485/486).
So layer 12 is doing long-range work at repeat positions that this
program has never characterised. Find out which head, and whether
it is reading the match.
Arms: window each of layer 12's nine heads individually (4-token
window, everything else intact), scored at match positions; then
for the worst head, measure where its top reads actually land.
REGISTERED PREDICTIONS:
  (a) CONCENTRATED: one head carries >= 50% of layer 12's
      +0.2094 match-specific cost;
  (b) IT IS A MATCH READER: that head's top read lands on a
      position holding the SAME TOKEN as the query at >= 30% of
      match positions, against a frequency-matched null under
      5% -- which would make it an induction-like head the band
      list missed;
  (c) CONTROL: the median layer-12 head carries under 0.05."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'layer12_match_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    at=m.transformer.h[LJ].attn
    are=sys.modules[type(at).__module__].apply_rotary_emb
    def matchmask(i,B):
        mk=torch.zeros(B,T,dtype=torch.bool)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                if t in last and last[t]+1<q and q>=8: mk[b,q]=True
                last[t]=q
        return mk
    def run(heads):
        tm=tn=0.0; nm_=nn_=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if heads:
                def fh(mo_,args,o_,heads=heads):
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
                    pat=(sc*sc2)*tril
                    patw=(sc*sc2)*win
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                    zw=torch.einsum('bhqk,bkhd->bhqd',patw,
                                    vm.float())
                    for h in heads: z[:,h]=zw[:,h]
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
            mk=matchmask(i,B)
            tm+=float(ce[mk].sum()); nm_+=int(mk.sum())
            tn+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for h in hs: h.remove()
        return tm/max(nm_,1),tn/max(nn_,1)
    bm,bn=run([])
    per={}
    for h in range(9):
        pm,pn=run([h])
        per[h]={'match':round(pm-bm,4),'nonmatch':round(pn-bn,4)}
        print(f"head 12.{h}: match {per[h]['match']:+.4f} "
              f"non-match {per[h]['nonmatch']:+.4f}",flush=True)
    allm,_=run(list(range(9)))
    total=allm-bm
    worst=max(per,key=lambda h:per[h]['match'])
    share=per[worst]['match']/max(total,1e-6)
    # where does the worst head read at match positions?
    same=0; nulls=0; n=0
    g=torch.Generator().manual_seed(5)
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hh=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hh.remove()
        X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def r3(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,worst][:,:,None],cos,sin)[:,:,0]
        qf,kf=r3(at.c_q),r3(at.c_k); q2,k2=r3(at.c_q2),r3(at.c_k2)
        pat=((torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())/128)
             *(torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())/128)) \
            *torch.tril(torch.ones(T,T,device=DEV))
        mk=matchmask(i,B)
        for b in range(B):
            toks=ROWS[i+b,:T].tolist()
            for q in range(T):
                if not mk[b,q]: continue
                k=int(pat[b,q,:q].abs().argmax())
                same+=int(toks[k]==toks[q]); n+=1
                kr=int(torch.randint(0,q,(1,),generator=g))
                nulls+=int(toks[kr]==toks[q])
    rate=same/max(n,1); nl=nulls/max(n,1)
    med=sorted(per[h]['match'] for h in per)[4]
    pa=share>=0.50
    pb=(rate>=0.30 and nl<0.05)
    pc=med<0.05
    out={'layer_total_match':round(total,4),'per_head':per,
         'worst_head':worst,'worst_share':round(share,3),
         'same_token_read_rate':round(rate,3),
         'null_rate':round(nl,3),'median_head_match':round(med,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'layer total {total:+.4f} | worst head 12.{worst} '
          f'({share:.1%}) | same-token reads {rate:.3f} vs null '
          f'{nl:.3f} | median head {med:+.4f}')
    for nm,v in (('a','one head carries >=50%'),
                 ('b','it reads the match token (>=30% vs <5%)'),
                 ('c','median head under 0.05')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
