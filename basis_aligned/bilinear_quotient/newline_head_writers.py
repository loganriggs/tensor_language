"""NEWLINE HEAD WRITERS -- who tells 12.6 that this is
line-broken text?
501 established what the head does: it predicts a line break with
AUC 0.769 (same-layer control 0.450), pushing the newline logit
+0.119 at breaks and -0.074 between them, switched on by
sentence-final punctuation and by a preceding newline, and gated
by document type -- the same period gets 2.14x the push in
newline-dense text. Two inputs, so at least two upstream sources,
and neither is named yet.
The residual entering layer 12 is an exact sum of writer
contributions: each block rescales the running residual as
x = lam0*x + lam1*x0, so writer j arrives multiplied by the
product of lam0 over blocks j+1..12 (cl.writer_coeffs). The first
attempt at this run used a flat lam0 instead and its exactness
gate correctly voided it at 68% error -- see writeup 503, which
that failure uncovered. So each writer's contribution to head
12.6's QUERY can be replaced by its own mean over positions,
killing what that writer says about THIS position while leaving
its average level, the key side, the value side, and every other
head untouched. Whatever the head loses is what that writer was
telling it.
The head's push is defined exactly as in 501: the newline logit
with the head intact minus the newline logit with the head
mean-ablated. The head-off run does not depend on the query, so it
is computed once and reused for every writer condition.
Two things are measured per writer: detector quality (AUC for
"next token is a newline") and document gating (mean push on
trigger tokens in newline-dense documents minus in sparse ones).
If the local trigger and the document gate arrive from different
places, these should degrade under different writers.
REGISTERED PREDICTIONS:
  (0) EXACTNESS, checked before anything is scored: the residual
      rebuilt from writer parts must match the captured attention
      input to within 1e-4 relative. A reconstruction that is
      merely close is the failure mode of 443 and 447;
  (a) SOMETHING CARRIES IT: silencing the single most important
      writer drops AUC by >= 0.05 from the unmodified value;
  (b) TWO DIFFERENT SOURCES: the writer whose removal most reduces
      the document-gating gap is NOT the writer whose removal most
      reduces AUC. This makes 501's two-input reading falsifiable
      -- one writer leading both means a single source;
  (c) CONTEXT IS REQUIRED: keeping ONLY wte at the query (all
      component writers silenced together) drops AUC below 0.65,
      i.e. the current token alone does not make this detector.
  CONTROL: a random direction of matched norm added to the query
      input, three seeds, must move AUC by less than 0.02. If
      noise of the same size costs as much as a named writer, the
      attribution means nothing.
Reporting rule from 497/500/501: pairs, not quotients, and every
bar scored through census_lib.score_bar."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_writers_results.json'
NFRESH=48

def auc(pos,neg):
    if not pos or not neg: return float('nan')
    allv=sorted([(v,1) for v in pos]+[(v,0) for v in neg])
    r=0.0;i=0
    while i<len(allv):
        j=i
        while j<len(allv) and allv[j][0]==allv[i][0]: j+=1
        rank=(i+j+1)/2.0
        r+=sum(rank for v,l in allv[i:j] if l==1); i=j
    n1=len(pos);n0=len(neg)
    return (r-n1*(n1+1)/2)/(n1*n0)

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    nxt=fresh[:,1:257]; cur=fresh[:,:256]
    isnl_next=(nxt==NLID)
    dens=isnl_next.float().mean(dim=1); hi=(dens>dens.median())
    TRIG={int(t) for t in cur.unique()
          if cl.d1(int(t)) in ('\n','.','"','?','!')}
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    exact=[]

    def run(mode,kill=False,seed=0):
        """mode: None | writer name | 'wte_only' | 'rand'.
        kill=True mean-ablates head 12.6 (query mode irrelevant)."""
        out=torch.zeros(NFRESH,T)
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
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()

            def qhook(mo,args,o_):
                y,v1r=o_; X=args[0]
                v1b=args[1] if args[1] is not None else v1r
                parts=cl.writer_parts(LJ,E,outs,'a')
                tot=sum(parts.values())
                Xr=F.rms_norm(tot,(D,)).to(X.dtype)
                exact.append(float((Xr-X).norm()
                                   /X.norm().clamp_min(1e-9)))
                if mode is None or kill: Xq=X
                elif mode=='wte_only':
                    t2=parts['wte']+sum(
                        p.mean(dim=(0,1),keepdim=True)
                        for w,p in parts.items() if w!='wte')
                    Xq=F.rms_norm(t2,(D,)).to(X.dtype)
                elif mode=='rand':
                    gg=torch.Generator(device=DEV).manual_seed(seed)
                    rr=torch.randn(tot.shape,generator=gg,device=DEV)
                    sc_=sum(float((parts[w]-parts[w].mean(
                              dim=(0,1),keepdim=True)).norm())
                            for w in WR[1:])/len(WR[1:])
                    rr=rr/rr.norm()*sc_
                    Xq=F.rms_norm(tot+rr,(D,)).to(X.dtype)
                else:
                    p=parts[mode]
                    t2=tot-p+p.mean(dim=(0,1),keepdim=True)
                    Xq=F.rms_norm(t2,(D,)).to(X.dtype)
                vv=at.c_v(X).view(B,T,NH,128)
                vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
                cq,sq=at.rotary(at.c_q(Xq).view(B,T,NH,128))
                ck,sk=at.rotary(at.c_q(X).view(B,T,NH,128))
                def rr_(w,Z,c,s):
                    return are(F.rms_norm(w(Z).view(B,T,NH,128),
                                          (128,)),c,s)
                qq=rr_(at.c_q,Xq,cq,sq); q22=rr_(at.c_q2,Xq,cq,sq)
                kk=rr_(at.c_k,X,ck,sk); k22=rr_(at.c_k2,X,ck,sk)
                sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),
                                kk.float())/128
                sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                                 k22.float())/128
                # only head HD sees the modified query; the others
                # must use the real one
                scT=torch.einsum('bqhd,bkhd->bhqk',
                                 rr_(at.c_q,X,ck,sk).float(),
                                 kk.float())/128
                sc2T=torch.einsum('bqhd,bkhd->bhqk',
                                  rr_(at.c_q2,X,ck,sk).float(),
                                  k22.float())/128
                scT[:,HD]=sc[:,HD]; sc2T[:,HD]=sc2[:,HD]
                p2=(scT*sc2T)*torch.tril(torch.ones(T,T,device=DEV))
                zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
                if kill:
                    zz[:,HD]=zz[:,HD].mean(dim=(0,1),keepdim=True)
                return (at.c_proj(zz.transpose(1,2).contiguous()
                        .view(B,T,-1).to(X.dtype)),v1r)
            hs.append(at.register_forward_hook(qhook))
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            out[i:i+B]=lg[:,:,NLID].cpu()
            for h in hs: h.remove()
        return out

    intact=run(None)
    relerr=max(exact)
    print(f'reconstruction relative error: {relerr:.3e}',flush=True)
    if relerr>1e-4:
        print('*** (0) EXACTNESS FAILED -- run is VOID ***')
        json.dump({'pred_0':False,'relerr':relerr},
                  open(OUT,'w'),indent=1); return
    print('(0) exactness: HELD',flush=True)
    off=run(None,kill=True)

    def metrics(lgv):
        push=lgv-off
        P=[];N=[];gh=[];gl=[]
        for r in range(NFRESH):
            for q in range(T):
                v=float(push[r,q])
                (P if isnl_next[r,q] else N).append(v)
                if int(cur[r,q]) in TRIG: (gh if hi[r] else gl).append(v)
        mh=sum(gh)/max(len(gh),1); ml=sum(gl)/max(len(gl),1)
        return auc(P,N),mh,ml,len(gh),len(gl)

    a0,mh0,ml0,ngh,ngl=metrics(intact)
    print(f'baseline: AUC {a0:.4f} | gate hi {mh0:+.4f} (n={ngh}) '
          f'lo {ml0:+.4f} (n={ngl}) gap {mh0-ml0:+.4f}',flush=True)
    res={}
    for w in WR[1:]+['wte','ALL_BUT_WTE']:
        mode='wte_only' if w=='ALL_BUT_WTE' else w
        a,mh,ml,_,_=metrics(run(mode))
        res[w]={'auc':round(a,4),'d_auc':round(a-a0,4),
                'gate_gap':round(mh-ml,4),
                'd_gate_gap':round((mh-ml)-(mh0-ml0),4),
                'gate_hi':round(mh,4),'gate_lo':round(ml,4)}
        print(f"{w}: AUC {a:.4f} ({a-a0:+.4f}) | gate gap "
              f"{mh-ml:+.4f} ({(mh-ml)-(mh0-ml0):+.4f})",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    ctrl=[]
    for s in (11,23,37):
        a,_,_,_,_=metrics(run('rand',seed=s)); ctrl.append(a-a0)
        print(f'control rand seed {s}: dAUC {a-a0:+.4f}',flush=True)
    comps=[w for w in res if w!='ALL_BUT_WTE']
    top_auc=min(comps,key=lambda w:res[w]['d_auc'])
    top_gate=min(comps,key=lambda w:res[w]['d_gate_gap'])
    va,_=cl.score_bar('a',-res[top_auc]['d_auc'],0.05)
    vb='HELD' if top_auc!=top_gate else 'FAILED'
    print(f'(b) AUC leader {top_auc} vs gate leader {top_gate}: {vb}')
    vc,_=cl.score_bar('c',0.65-res['ALL_BUT_WTE']['auc'],0.0)
    ok=max(abs(c) for c in ctrl)<0.02
    print(f"CONTROL (max |dAUC| over 3 random seeds "
          f"{max(abs(c) for c in ctrl):.4f} < 0.02): "
          f"{'ok' if ok else 'VIOLATED'}")
    out={'writers':res,'baseline_auc':round(a0,4),
         'baseline_gate_gap':round(mh0-ml0,4),
         'auc_leader':top_auc,'gate_leader':top_gate,
         'control_dauc':[round(c,4) for c in ctrl],
         'relerr':relerr,'pred_0':True,
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','control_ok':bool(ok),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
