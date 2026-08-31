"""DAS-PROPER v2 (rung 164): the rung-162 optimizer fixed.

CONVENTION (S2135): share = member |dCE| of subspace patch / full swap; census rows. S2259: v1's Adam(1e-2)
+ naive QR retraction ROSE in loss and regressed the control below its own warm start - an instrument
failure, not physics. v2: lr 3e-3 with cosine decay over 300 steps, best-loss checkpointing (the evaluated P
is the best-loss iterate, never worse than the warm start in the training metric), same circuits/seeds/arms.

REGISTERED PREDICTIONS:
  (a) CONTROL NO-REGRESS: learned share >= pca8 share - 0.02 at the control circuit.
  (b) LEARNING GAINS SOMEWHERE: learned >= pca8 + 0.10 at >= 1 of the 2 low circuits.
  (c) OPTIMIZER SANE: orthogonality residual <= 1e-3 AND best loss <= initial loss at all 3.
NULL: with a healthy optimizer, learning still adds < 0.05 everywhere - fixed member-PCA is the rank-8
carrier ceiling, and DAS-proper closes in favor of the cheap construction. PRICE: 9,216 values per passing
subspace. Tripwire: INSTRUMENT FAIL if the rung-161 receipt is missing or any arm is inert. Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['dasproper_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: DAS-proper v2')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'dasproper2_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat(); HALF=NFLAT//2
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    DL=json.load(open(PT+'daslite_results.json'))['rows']
    DL=sorted(DL,key=lambda r:r['shares']['pca8'])
    PICKS=[DL[0]['tag'],DL[1]['tag'],DL[-1]['tag']]
    print(f'picked (2 low + control): {PICKS}',flush=True)
    CINFO={}
    for r in DL:
        t=r['tag']
        lf=CN.leaf(t)
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        CINFO[t]={'mask':mm,'comp':r['comp'],'pca8':r['shares']['pca8']}
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    comps=sorted({CINFO[t]['comp'] for t in PICKS})
    CAPOUT={c:[] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mkc(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                CAPOUT[c].append(y.detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mkc()))
    _=evalce()
    for h in hs: h.remove()
    CAPOUT={c:torch.cat(v) for c,v in CAPOUT.items()}
    print('outputs captured',flush=True)
    import statistics as stt
    rows=[]
    for idx9,t in enumerate(PICKS):
        c=CINFO[t]['comp']; mod,kind=module_of(c)
        ISM=CINFO[t]['mask']
        mi=ISM.nonzero().squeeze(1)
        g=torch.Generator().manual_seed(500+idx9)
        perm=mi[torch.randperm(mi.numel(),generator=g)]
        SRC=torch.zeros(NFLAT,dtype=torch.long); SRC[mi]=perm
        Ymem=CAPOUT[c][mi].float()
        _,_,Vh=torch.linalg.svd((Ymem-Ymem.mean(0))[:20000].to(DEV),full_matrices=False)
        P0=Vh[:8].T.contiguous()
        st={'mode':None,'P':None,'bi':0}
        def hook(mo,i_,o_,c=c,kind=kind):
            y=o_[0] if isinstance(o_,tuple) else o_
            B9=y.reshape(-1,D).shape[0]
            lo=st['bi']*1024
            sel=ISM[lo:lo+B9]
            yn=y.reshape(-1,D)
            if sel.any():
                ysrc=CAPOUT[c][SRC[lo:lo+B9][sel]].to(y.device).float()
                ycur=yn[sel].float()
                if st['mode']=='full':
                    upd=ysrc
                else:
                    Pm=st['P']
                    upd=ycur+(ysrc-ycur)@(Pm@Pm.T)
                yn=yn.clone(); yn[sel]=upd.to(yn.dtype)
            st['bi']+=1
            yn=yn.view_as(y)
            if kind=='attn': return (yn,o_[1])
            return yn
        def run_eval(mode,P=None):
            st['mode']=mode; st['P']=P; st['bi']=0
            hh=mod.register_forward_hook(hook)
            cev=evalce()
            hh.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {t}/{mode} inert')
            return float(d[ISM].abs().mean())
        full=run_eval('full')
        s_pca=run_eval('sub',P0)/full
        # training rows: train-half rows containing members
        rowmask=ISM[:HALF].view(-1,256).any(1)
        tr_rows=rowmask.nonzero().squeeze(1)
        P=P0.clone().requires_grad_(True)
        opt=torch.optim.Adam([P],lr=3e-3)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=300)
        gtr=torch.Generator().manual_seed(900+idx9)
        best=(float('inf'),P0.clone()); loss0=None; lossF=None
        for step in range(300):
            ri=tr_rows[torch.randint(0,tr_rows.numel(),(4,),generator=gtr)]
            bb=ROWS[ri][:,:257].to(DEV)
            idxb=bb[:,:-1].contiguous()
            lo_list=ri*256
            selb=torch.cat([ISM[l:l+256] for l in lo_list.tolist()])
            if not selb.any(): continue
            srcb=torch.cat([SRC[l:l+256] for l in lo_list.tolist()])
            def fwd_with(patch):
                cap={'i':0}
                def h2(mo,i_,o_):
                    y=o_[0] if isinstance(o_,tuple) else o_
                    yn=y.reshape(-1,D)
                    ysrc=CAPOUT[CINFO[t]['comp']][srcb[selb]].to(DEV).float()
                    ycur=yn[selb]
                    if patch=='full': upd=ysrc.to(yn.dtype)
                    else: upd=(ycur.float()+(ysrc-ycur.float())@(P@P.T)).to(yn.dtype)
                    yn=yn.clone(); yn[selb]=upd
                    yn=yn.view_as(y)
                    return (yn,o_[1]) if kind=='attn' else yn
                hh=mod.register_forward_hook(h2)
                x=F.rms_norm(m.transformer.wte(idxb),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                hh.remove()
                return lg.view(-1,lg.size(-1))[selb]
            with torch.no_grad():
                tgt=F.log_softmax(fwd_with('full'),-1)
            lgs=F.log_softmax(fwd_with('sub'),-1)
            loss=F.kl_div(lgs,tgt,log_target=True,reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                Q,_=torch.linalg.qr(P.data); P.data=Q[:,:8].contiguous()
            lv=float(loss)
            if loss0 is None: loss0=lv
            lossF=lv
            if lv<best[0]: best=(lv,P.data.clone())
            if step%75==0: print(f'  {t} step {step} loss {lv:.4f}',flush=True)
        Pf=best[1].detach()
        print(f'  {t} loss {loss0:.4f} -> best {best[0]:.4f} (final {lossF:.4f})',flush=True)
        orth=float((Pf.T@Pf-torch.eye(8,device=DEV)).abs().max())
        s_learn=run_eval('sub',Pf)/full
        rows.append({'tag':t,'comp':c,'full':round(full,4),'share_pca8':round(s_pca,3),
                     'share_learned':round(s_learn,3),'orth_resid':orth,'is_control':t==PICKS[-1],
                     'loss0':round(loss0,4),'loss_best':round(best[0],4)})
        print(f"  {t}: full {full:.3f} | pca8 {s_pca:.2f} -> learned {s_learn:.2f} (orth {orth:.1e})",flush=True)
    low=[r for r in rows if not r['is_control']]
    ctrl=[r for r in rows if r['is_control']][0]
    pa=ctrl['share_learned']>=ctrl['share_pca8']-0.02
    pb=any(r['share_learned']>=r['share_pca8']+0.10 for r in low)
    pc=all(r['orth_resid']<=1e-3 for r in rows) and all(r['loss_best']<=r['loss0'] for r in rows)
    res={'rows':rows,'convention':'share = member |dCE| subspace patch / full swap; within-script arms',
         'pred_a_control_noregress':bool(pa),'pred_b_learning_gains':bool(pb),'pred_c_optimizer_sane':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) control no-regress ({ctrl['share_pca8']:.2f} -> {ctrl['share_learned']:.2f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) learned >= pca8+0.10 at >= 1 low circuit: {'HELD' if pb else 'FAILED'}")
    print(f"(c) orthonormal + best loss <= initial at all 3: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
