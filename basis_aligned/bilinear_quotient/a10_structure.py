"""A10 -- one structure component, or three coincidences?
Attention layer 10 keeps turning up. The behaviour atlas (518) has
it leading TWO classes: opening quotes (+0.272 nats at target
against +0.021 elsewhere, the second largest concentrated effect
in the model) and sentence ends (+0.065 vs +0.021). Independently,
514 found it carrying 48% of head 12.6's document gate -- the
signal that makes the newline head push harder in text that is
already full of line breaks -- on the key/value path, where the
entire query side of that head managed 7%. And 519's exact pair
decomposition put a10 x a10 as the highest-ranked non-MLP writer
pair feeding 12.6's score.
Three separate measurements, three different methods, one
component, and it has never been studied directly. Either a10
contains one thing that tracks document structure -- where
sentences end, where quotations open, whether this text is
line-broken -- or it contains three unrelated circuits that happen
to share a layer. Those predict different head decompositions, so
the question is decidable.
Each of a10's nine heads is mean-ablated in turn and priced on all
three jobs at once:
  opening-quote targets, with position-matched and random controls
  sentence-end targets, same controls
  the newline document gate: the difference in head 12.6's push on
    trigger tokens between newline-dense and newline-sparse
    documents, which 501 measured at +0.050 and 514 showed a10
    carries -0.024 of
REGISTERED PREDICTIONS:
  (a) CONCENTRATION: some head carries >= 40% of a10's
      opening-quote damage;
  (b) ONE COMPONENT OR THREE: the head leading opening quotes is
      the SAME as the head leading sentence ends. This is the
      structure-tracker hypothesis made falsifiable -- if
      different heads lead, a10 is a shared address and not a
      shared computation, and the "structure component" reading
      of 518 must be withdrawn;
  (c) IT REACHES THE NEWLINE HEAD: ablating that same head alone
      reduces the newline document gate by >= 0.010, i.e. at
      least 40% of what the whole of a10 contributes. This ties
      three independent measurements to one head or breaks the
      connection cleanly.
  NULL 1: on a fully random target set of the same size, no head
      reaches 20% of the leader's opening-quote damage.
  NULL 2: the leader's damage at position-matched control
      positions is reported as an absolute pair against its damage
      at targets; no ratio is scored (the rule from 520).
If (b) fails the honest conclusion is that a10 is three circuits,
and the atlas map should be read as a map of LAYERS with dedicated
machinery rather than of components with jobs."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=10; NH=9; NLID=198; NLHEAD=(12,6)
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'a10_structure_results.json'
NFRESH=96

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    oq=torch.zeros(NFRESH,T,dtype=torch.bool)
    se=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            s=cl.d1(int(nxt[r,q])); t=s.strip()
            if t in ('"',"'",'``',"'") and (s.startswith(' ') or s==t):
                oq[r,q]=True
            if t in ('.','!','?'): se[r,q]=True
    g=torch.Generator().manual_seed(29)
    def controls(mask):
        c=torch.zeros_like(mask); rr=torch.zeros_like(mask)
        for r in range(NFRESH):
            k=int(mask[r].sum())
            if k==0: continue
            pos=mask[r].nonzero().squeeze(1)
            c[r,(torch.randint(-6,7,(k,),generator=g)+pos)
              .clamp(0,T-1)]=True
            rr[r,torch.randint(0,T,(k,),generator=g)]=True
        return c,rr
    oqc,oqr=controls(oq); sec,ser=controls(se)
    isnl=(nxt==NLID)
    dens=isnl.float().mean(dim=1); hi=(dens>dens.median())
    TRIG={int(t) for t in cur.unique()
          if cl.d1(int(t)) in ('\n','.','"','?','!')}
    print(f'{int(oq.sum())} opening-quote targets | {int(se.sum())} '
          f'sentence-end targets | {int(isnl.sum())} newline '
          f'targets',flush=True)
    if int(oq.sum())<30:
        print('*** opening-quote class unpopulated -- VOID ***')
        json.dump({'void':'oq unpopulated','n':int(oq.sum())},
                  open(OUT,'w'),indent=1); return
    at=m.transformer.h[LJ].attn
    nat=m.transformer.h[NLHEAD[0]].attn

    def hooks(a10_head=None,kill_nl=False):
        hs=[]
        if a10_head is not None:
            def fh(mo,args,o_):
                y,v1r=o_; X=args[0]; B=X.shape[0]
                v1b=args[1] if args[1] is not None else v1r
                z,_=cl.head_parts(LJ,X,v1b); z=z.clone()
                z[:,a10_head]=z[:,a10_head].mean(dim=(0,1),
                                                 keepdim=True)
                return (at.c_proj(z.transpose(1,2).contiguous()
                        .view(B,T,-1).to(X.dtype)),v1r)
            hs.append(at.register_forward_hook(fh))
        if kill_nl:
            def fh2(mo,args,o_):
                y,v1r=o_; X=args[0]; B=X.shape[0]
                v1b=args[1] if args[1] is not None else v1r
                z,_=cl.head_parts(NLHEAD[0],X,v1b); z=z.clone()
                z[:,NLHEAD[1]]=z[:,NLHEAD[1]].mean(dim=(0,1),
                                                   keepdim=True)
                return (nat.c_proj(z.transpose(1,2).contiguous()
                        .view(B,T,-1).to(X.dtype)),v1r)
            hs.append(nat.register_forward_hook(fh2))
        return hs

    def run(a10_head=None,kill_nl=False):
        ce=torch.zeros(NFRESH,T); l198=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=hooks(a10_head,kill_nl)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            l198[i:i+B]=lg[:,:,NLID].cpu()
            for h in hs: h.remove()
        return ce,l198

    def gate(l_on,l_off):
        push=l_on-l_off
        gh=[];gl=[]
        for r in range(NFRESH):
            for q in range(T):
                if int(cur[r,q]) in TRIG:
                    (gh if hi[r] else gl).append(float(push[r,q]))
        return (sum(gh)/max(len(gh),1))-(sum(gl)/max(len(gl),1))

    base_ce,base_on=run()
    _,base_off=run(kill_nl=True)
    g0=gate(base_on,base_off)
    print(f'baseline newline document gate: {g0:+.4f}',flush=True)
    res={}
    for h in range(NH):
        ce,on=run(a10_head=h)
        _,off=run(a10_head=h,kill_nl=True)
        d=ce-base_ce
        row={'oq':round(float(d[oq].mean()),5),
             'oq_ctrl':round(float(d[oqc].mean()),5),
             'oq_rand':round(float(d[oqr].mean()),5),
             'se':round(float(d[se].mean()),5),
             'se_ctrl':round(float(d[sec].mean()),5),
             'gate':round(gate(on,off),4),
             'd_gate':round(gate(on,off)-g0,4),
             'global':round(float(d.mean()),5)}
        res[f'{LJ}.{h}']=row
        print(f"{LJ}.{h}: quote {row['oq']:+.5f} (ctrl "
              f"{row['oq_ctrl']:+.5f}) | sent-end {row['se']:+.5f} "
              f"| gate {row['gate']:+.4f} ({row['d_gate']:+.4f})",
              flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    tot=sum(max(v['oq'],0) for v in res.values())
    topq=max(res,key=lambda k:res[k]['oq'])
    tops=max(res,key=lambda k:res[k]['se'])
    share=res[topq]['oq']/tot if tot>0 else 0.0
    va,_=cl.score_bar('a',share,0.40)
    vb='HELD' if topq==tops else 'FAILED'
    vc,_=cl.score_bar('c',-res[topq]['d_gate'],0.010)
    worst=max(v['oq_rand'] for v in res.values())
    n1=worst<0.20*max(res[topq]['oq'],1e-9)
    print(f"\nquote leader {topq} ({share*100:.0f}% of a10's "
          f"opening-quote damage); sentence-end leader {tops}")
    print(f"(b) same head leads both: {vb}")
    print(f"(c) {topq} alone moves the newline gate by "
          f"{res[topq]['d_gate']:+.4f} (all of a10: -0.0243)")
    print(f"NULL 1 (worst random-target damage {worst:+.5f} < 20% "
          f"of {res[topq]['oq']:+.5f}): {'ok' if n1 else 'VIOLATED'}")
    print(f"NULL 2 (pair) leader at targets {res[topq]['oq']:+.5f} "
          f"vs position-matched controls {res[topq]['oq_ctrl']:+.5f}")
    out={'heads':res,'baseline_gate':round(g0,4),
         'quote_leader':topq,'sentence_leader':tops,
         'quote_share':round(share,3),
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','null1_ok':bool(n1),
         'n_oq':int(oq.sum()),'n_se':int(se.sum()),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
