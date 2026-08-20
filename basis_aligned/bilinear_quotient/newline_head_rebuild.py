"""NEWLINE HEAD REBUILD -- is the dense pair structure actually
needed, or only densely distributed?
519 computed head 12.6's score exactly as a sum over 625 writer
pairs (reconstruction 5.14e-7) and found the mass dense: the top
ten pairs carry 12.2% against 1.6% for uniform, and the leading
pairs at newline targets are almost the same as at control
positions. Mass, though, is not function. A pair can carry little
absolute mass and still be what tips the score across the
threshold that matters, and conversely most of the mass can be a
common-mode term that cancels. The question density cannot answer
is SUFFICIENCY: how many pairs does the model actually need?
Method: rebuild head 12.6's score from its top K pairs only --
score_K(q,k) = SUM_{(i,j) in topK} (1/128) Q_i(q).K_j(k) *
factor2(q,k) -- run the REAL model with that score in place of the
head's own, and price the head's newline benefit against its own
ablation:
    retention(K) = (CE_ablated - CE_rebuilt) / (CE_ablated - CE_real)
at newline-target positions. 1.0 means the K pairs reproduce the
head; 0.0 means they reproduce nothing; the curve over K is the
answer rather than any single point.
K = 10, 25, 50, 100, 200, 400, 625. The last is the whole
decomposition and is a sanity check, not a result.
REGISTERED PREDICTIONS:
  (0) SANITY: retention at K=625 is >= 0.95. The full rebuild IS
      the head, up to float error, so anything else means the
      substitution machinery is broken and the run is VOID;
  (a) COMPRESSIBLE: some K <= 100 (16% of the pairs) reaches
      retention >= 0.70. This is the claim that the head has a
      small computational description even though its mass is
      spread;
  (b) BEATS RANDOM: at that same K, the top-K rebuild retains at
      least twice what a random-K rebuild does (three draws). If
      random pairs do as well, the ranking is not informative and
      (a) means only that the head is robust to truncation;
  (c) CURVE REPORTED: retention is reported at every K, for
      newline targets AND for position-matched controls, so that
      a difference in compressibility between the two is visible
      rather than assumed.
  NULL: retention at K=10 must be below retention at K=625. A
      curve that does not increase with K means the truncation is
      not doing what it claims.
If (a) fails -- if the model needs hundreds of pairs to reproduce
one head's behaviour -- that is the strongest statement this
program can make that the computation is genuinely high-rank at
the compositional level, and it would close the tier-4 question
for this head with a negative rather than leaving it open."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_rebuild_results.json'
NFRESH=32
KS=[10,25,50,100,200,400,625]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    nl=(fresh[:,1:257]==NLID)
    g=torch.Generator().manual_seed(29)
    ctrl=torch.zeros_like(nl)
    for r in range(NFRESH):
        k=int(nl[r].sum())
        if k==0: continue
        pos=nl[r].nonzero().squeeze(1)
        ctrl[r,(torch.randint(-6,7,(k,),generator=g)+pos)
             .clamp(0,T-1)]=True
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    NW=len(WR)
    TRI=torch.tril(torch.ones(T,T,device=DEV))
    prev=json.load(open(PT+'newline_head_pairs_results.json'))
    order=[(WR.index(a),WR.index(b))
           for a,b,_ in prev['top_pairs_newline']]
    # the stored file keeps only the top 10; recompute the full
    # ranking here so the curve can extend past K=10
    RANK={'built':False,'order':None}
    errs=[]

    def pieces(X,E,outs,B):
        parts=cl.writer_parts(LJ,E,outs,'a')
        tot=sum(parts.values())
        errs.append(float((F.rms_norm(tot,(D,))-X.float()).norm()
                    /X.float().norm().clamp_min(1e-9)))
        s=(X.float().norm(dim=-1,keepdim=True)
           /tot.norm(dim=-1,keepdim=True).clamp_min(1e-9))
        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
        out={}
        for nm,W in (('q',at.c_q),('k',at.c_k),
                     ('q2',at.c_q2),('k2',at.c_k2)):
            full=W(X).view(B,T,NH,128)[:,:,HD].float()
            a=full.pow(2).mean(-1,keepdim=True).sqrt().clamp_min(1e-9)
            per=torch.stack([
                W((parts[w]*s).to(X.dtype)).view(B,T,NH,128)[:,:,HD]
                .float() for w in WR],0)
            per=are(per.permute(1,2,0,3),cq,sq).permute(2,0,1,3)
            out[nm]=per/a[None]
        return out

    def forward(mode,K=None,pairs=None):
        """mode: 'real' | 'ablate' | 'rebuild'"""
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            outs={}; hs=[]
            if mode!='real':
                for lj in range(LJ):
                    for kind,mod in (('a',m.transformer.h[lj].attn),
                                     ('m',m.transformer.h[lj].mlp)):
                        def mk(k9=f'{kind}{lj}'):
                            def h(mo,i_,o_):
                                y=o_[0] if isinstance(o_,tuple) else o_
                                outs[k9]=y.detach().float()
                            return h
                        hs.append(mod.register_forward_hook(mk()))
                def fh(mo,args,o_):
                    y,v1r=o_; X=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    z,vm=cl.head_parts(LJ,X,v1b)
                    z=z.clone()
                    if mode=='ablate':
                        z[:,HD]=z[:,HD].mean(dim=(0,1),keepdim=True)
                    else:
                        P=pieces(X,None,outs,B)
                        sq2=P['q2'].sum(0); sk2=P['k2'].sum(0)
                        f2=torch.einsum('bqd,bkd->bqk',sq2,sk2)/128
                        acc=torch.zeros(B,T,T,device=DEV)
                        for (ii,jj) in pairs:
                            acc+=torch.einsum(
                                'bqd,bkd->bqk',P['q'][ii],P['k'][jj])
                        sc=(acc/128)*f2*TRI
                        z[:,HD]=torch.einsum('bqk,bkd->bqd',sc,
                                             vm[:,:,HD].float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
        return ce

    # full pair ranking, recomputed
    mass=torch.zeros(NW,NW)
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        outs={}; hs=[]
        for lj in range(LJ):
            for kind,mod in (('a',m.transformer.h[lj].attn),
                             ('m',m.transformer.h[lj].mlp)):
                def mk(k9=f'{kind}{lj}'):
                    def h(mo,i_,o_):
                        y=o_[0] if isinstance(o_,tuple) else o_
                        outs[k9]=y.detach().float()
                    return h
                hs.append(mod.register_forward_hook(mk()))
        cap={}
        hs.append(at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0])))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        X=cap['X']; P=pieces(X,None,outs,B)
        sq2=P['q2'].sum(0); sk2=P['k2'].sum(0)
        f2=torch.einsum('bqd,bkd->bqk',sq2,sk2)/128
        for b in range(B):
            qs=nl[i+b].nonzero().squeeze(1).to(DEV)
            if not len(qs): continue
            pr=torch.einsum('iqd,jkd->ijqk',P['q'][:,b][:,qs],
                            P['k'][:,b])
            term=(pr/128)*f2[b][qs][None,None]*TRI[qs][None,None]
            mass+=term.abs().sum(dim=(2,3)).cpu()
    fl=mass.flatten().argsort(descending=True)
    full_order=[(int(t)//NW,int(t)%NW) for t in fl]
    print(f'exactness of parts: {max(errs):.3e}',flush=True)
    real=forward('real'); abl=forward('ablate')
    def price(ce,mask):
        return float(ce[mask].mean())
    base_nl=price(real,nl); abl_nl=price(abl,nl)
    base_ct=price(real,ctrl); abl_ct=price(abl,ctrl)
    print(f'head benefit at newline targets: '
          f'{abl_nl-base_nl:+.4f} nats | at controls '
          f'{abl_ct-base_ct:+.4f}',flush=True)
    gg=torch.Generator().manual_seed(7)
    res={}
    for K in KS:
        ce=forward('rebuild',pairs=full_order[:K])
        rt=(abl_nl-price(ce,nl))/max(abl_nl-base_nl,1e-9)
        rc=(abl_ct-price(ce,ctrl))/max(abl_ct-base_ct,1e-9)
        rnd=[]
        if K<625:
            for s in range(3):
                pick=[full_order[int(t)] for t in
                      torch.randperm(NW*NW,generator=gg)[:K]]
                cer=forward('rebuild',pairs=pick)
                rnd.append(round((abl_nl-price(cer,nl))
                                 /max(abl_nl-base_nl,1e-9),3))
        res[K]={'retention_newline':round(rt,3),
                'retention_control':round(rc,3),'random':rnd}
        print(f'K={K:>3}: retention {rt:+.3f} (control {rc:+.3f}) '
              f'| random {rnd}',flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    p0=res[625]['retention_newline']>=0.95
    print(f"(0) full rebuild reproduces the head "
          f"({res[625]['retention_newline']:.3f} >= 0.95): "
          f"{'HELD' if p0 else 'FAILED -- RUN VOID'}")
    hit=[K for K in KS if K<=100
         and res[K]['retention_newline']>=0.70]
    pa=bool(hit)
    pb=False
    if hit:
        K=hit[0]
        rr=max(res[K]['random']) if res[K]['random'] else 0
        pb=res[K]['retention_newline']>=2*max(rr,1e-9)
        print(f'(b) at K={K}: top {res[K]["retention_newline"]:.3f} '
              f'vs best random {rr:.3f}')
    print(f"(a) some K<=100 reaches retention >=0.70: "
          f"{'HELD ('+str(hit[0])+')' if pa else 'FAILED'}")
    nul=res[KS[0]]['retention_newline']<res[625]['retention_newline']
    print(f"NULL (retention increases with K): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'curve':{str(k):v for k,v in res.items()},
         'head_benefit_newline':round(abl_nl-base_nl,4),
         'head_benefit_control':round(abl_ct-base_ct,4),
         'exactness':max(errs),'pred_0':bool(p0),'pred_a':bool(pa),
         'pred_b':bool(pb),'null_ok':bool(nul),
         'first_K_over_70':hit[0] if hit else None,
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
