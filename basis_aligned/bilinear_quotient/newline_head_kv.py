"""NEWLINE HEAD KV -- where is the document gate?
506 silenced each writer's contribution to head 12.6's QUERY and
found the detector's input diffuse (no writer worth more than
0.0154 AUC, all context together worth 0.140) and the document
gate UNLOCATED: every writer moved the gate gap by less than 0.008
against a gap of 0.050, and silencing wte or all context made the
gap BIGGER, which is the opposite of a writer supplying it.
That run only touched the query. A head has three input paths, and
"is this line-broken text" is exactly the kind of signal that
could live on the key side (which positions look like line ends)
or in the values (what gets moved) rather than in the query. This
runs the identical surgery on the KEY side, the VALUE side, and
both together.
REGISTERED PREDICTIONS:
  (0) EXACTNESS to 1e-4 relative before anything is scored;
  (a) THE GATE HAS A SOURCE: some writer, silenced on the key or
      the value side, reduces the document-gating gap by >= 0.015
      -- a third of the 0.050 gap, and five times the best the
      query side managed;
  (b) THE SIDES DIVIDE THE WORK: the writer/side that costs the
      most detector quality is not the writer/side that costs the
      most gating. If one side carries both, the two-input reading
      of 501 collapses into one pathway and must be withdrawn;
  (c) KEY/VALUE CONTEXT MATTERS AT ALL: silencing every component
      writer at once on key+value drops AUC below 0.70 (query-side
      all-off gave 0.643 from a baseline of 0.783).
  CONTROL: matched random directions on each side, three seeds,
      must move AUC by less than 0.02.
Whatever happens, the honest outcome is recorded: if no side
carries the gate, then the 2.14x document dependence of 501 is a
property the head inherits from its inputs collectively rather
than a signal any component supplies, and the newline circuit's
input side is closed as diffuse."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_kv_results.json'
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

    SIDE={'v':'query'}

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
                sd=SIDE['v']
                Xk=Xq if sd in ('key','kv') else X
                Xv=Xq if sd in ('value','kv') else X
                Xqq=Xq if sd=='query' else X
                vv=at.c_v(Xv).view(B,T,NH,128)
                vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
                cq,sq=at.rotary(at.c_q(Xqq).view(B,T,NH,128))
                ck,sk=at.rotary(at.c_q(X).view(B,T,NH,128))
                def rr_(w,Z,c,s):
                    return are(F.rms_norm(w(Z).view(B,T,NH,128),
                                          (128,)),c,s)
                qq=rr_(at.c_q,Xqq,cq,sq); q22=rr_(at.c_q2,Xqq,cq,sq)
                kk=rr_(at.c_k,Xk,ck,sk); k22=rr_(at.c_k2,Xk,ck,sk)
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

    ALL={}
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
    for sd in ('key','value','kv'):
        SIDE['v']=sd
        res={}
        for w in WR[1:]+['wte','ALL_BUT_WTE']:
            mode='wte_only' if w=='ALL_BUT_WTE' else w
            a,mh,ml,_,_=metrics(run(mode))
            res[w]={'auc':round(a,4),'d_auc':round(a-a0,4),
                    'gate_gap':round(mh-ml,4),
                    'd_gate_gap':round((mh-ml)-(mh0-ml0),4)}
            print(f"[{sd}] {w}: AUC {a:.4f} ({a-a0:+.4f}) | gate gap "
                  f"{mh-ml:+.4f} ({(mh-ml)-(mh0-ml0):+.4f})",flush=True)
        ctrl=[]
        for sdd in (11,23,37):
            a,_,_,_,_=metrics(run('rand',seed=sdd)); ctrl.append(a-a0)
        ALL[sd]={'writers':res,'control_dauc':[round(c,4) for c in ctrl]}
        print(f'[{sd}] control max |dAUC| '
              f'{max(abs(c) for c in ctrl):.4f}',flush=True)
        json.dump(ALL,open(OUT,'w'),indent=1)
    best_gate=min(((v['d_gate_gap'],sd,w)
                   for sd,D_ in ALL.items()
                   for w,v in D_['writers'].items()
                   if w!='ALL_BUT_WTE'),default=(0,None,None))
    best_auc=min(((v['d_auc'],sd,w)
                  for sd,D_ in ALL.items()
                  for w,v in D_['writers'].items()
                  if w!='ALL_BUT_WTE'),default=(0,None,None))
    va,_=cl.score_bar('a',-best_gate[0],0.015)
    vb='HELD' if (best_gate[1],best_gate[2])!=(best_auc[1],best_auc[2]) \
       else 'FAILED'
    kvall=ALL['kv']['writers']['ALL_BUT_WTE']['auc']
    vc,_=cl.score_bar('c',0.70-kvall,1e-9)
    ok=all(max(abs(c) for c in D_['control_dauc'])<0.02
           for D_ in ALL.values())
    print(f"\nbest gate mover: {best_gate[2]} on the {best_gate[1]} "
          f"side ({best_gate[0]:+.4f}); best AUC mover: "
          f"{best_auc[2]} on {best_auc[1]} ({best_auc[0]:+.4f})")
    print(f"(b) different writer/side for gate vs detector: {vb}")
    print(f"(c) kv all-context-off AUC {kvall:.4f} < 0.70")
    print(f"CONTROL random |dAUC| < 0.02 on every side: "
          f"{'ok' if ok else 'VIOLATED'}")
    out={'baseline_auc':round(a0,4),
         'baseline_gate_gap':round(mh0-ml0,4),
         'sides':ALL,'best_gate_mover':best_gate,
         'best_auc_mover':best_auc,'relerr':relerr,'pred_0':True,
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','control_ok':bool(ok),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
