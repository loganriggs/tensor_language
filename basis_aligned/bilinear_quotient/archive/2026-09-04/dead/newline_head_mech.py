"""NEWLINE HEAD MECHANISM -- 499: what does head 12.6 compute?
497 localized it: deleting 12.6 costs +0.0682 nats at line breaks,
+0.0057 six tokens either side, and nothing elsewhere, and it puts
3.7x more score mass on the most recent preceding newline at line
breaks than at control positions. That says WHERE it looks, not
WHAT it computes.
The hypothesis this run tests is line-length regularity. Text with
regular line structure -- verse, lists, tables, chat logs, code --
breaks lines at a predictable interval, so a head that reads the
DISTANCE back to the previous newline can predict the next break
without reading any content at all. If that is what 12.6 does, its
help must depend on whether the current line matches the
document's own rhythm, and it should be useless on prose that
wraps arbitrarily.
Every newline target is labelled by its gap (tokens since the
previous newline) against that document's median gap:
  REGULAR    |gap - median| <= 0.25 x median
  IRREGULAR  everything else
and separately by whether the previous token is itself a newline
(BLANK, a paragraph break, where the answer needs no arithmetic).
Also measured: what the head's deletion does to the logit of the
newline token specifically, against the strongest non-newline
competitor at the same positions; and a descriptive breakdown of
where its score mass lands at line breaks (most recent newline,
line start, position 0, self, previous token, other).
REGISTERED PREDICTIONS:
  (a) RHYTHM: cost on REGULAR newline targets is >= 2x cost on
      IRREGULAR ones. This is the whole hypothesis -- if the head
      helped equally on both, it is not doing line arithmetic;
  (b) NOT JUST PARAGRAPHS: the regular-vs-irregular gap survives
      excluding BLANK targets, i.e. it is not merely that the
      head handles double newlines and blank lines happen to be
      regular. Bar: >= 1.5x among non-blank targets alone;
  (c) IT PUSHES NEWLINE: deleting 12.6 lowers the newline token's
      logit at newline targets by >= 0.10 more than it lowers the
      best non-newline competitor at the same positions.
  NULL: the same regular/irregular split computed at the
      position-matched control positions must show no comparable
      gap (< 1.5x). If control positions show the same pattern,
      the split is tracking sequence position, not line rhythm."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=12; HD=6; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_head_mech_results.json'
NFRESH=48

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    NLTOK=set()
    nl=torch.zeros(NFRESH,T,dtype=torch.bool)
    isnl=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            if chr(10) in cl.d1(int(fresh[r,q+1])):
                nl[r,q]=True; NLTOK.add(int(fresh[r,q+1]))
            if chr(10) in cl.d1(int(fresh[r,q])): isnl[r,q]=True
    NLTOK=sorted(NLTOK)
    ctrl=torch.zeros_like(nl)
    g=torch.Generator().manual_seed(29)
    for r in range(NFRESH):
        k=int(nl[r].sum())
        if k==0: continue
        pos=nl[r].nonzero().squeeze(1)
        j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(0,T-1)
        ctrl[r,j]=True
    # label each target REGULAR/IRREGULAR by the document's rhythm,
    # and BLANK when the previous token is itself a newline
    lab={}
    for name,mask in (('nl',nl),('ctrl',ctrl)):
        reg=torch.zeros_like(mask); irr=torch.zeros_like(mask)
        blank=torch.zeros_like(mask)
        for r in range(NFRESH):
            keys=isnl[r].nonzero().squeeze(1).tolist()
            if len(keys)<3: continue
            gaps=[b-a for a,b in zip(keys,keys[1:])]
            gaps=sorted(gaps); med=gaps[len(gaps)//2]
            if med<=0: continue
            for q in mask[r].nonzero().squeeze(1).tolist():
                prev=[k for k in keys if k<=q]
                if not prev: continue
                gap=(q+1)-prev[-1]
                (reg if abs(gap-med)<=0.25*med else irr)[r,q]=True
                if q>0 and isnl[r,q]: blank[r,q]=True
        lab[name]={'reg':reg,'irr':irr,'blank':blank}
    print(f'targets {int(nl.sum())} | regular '
          f'{int(lab["nl"]["reg"].sum())} irregular '
          f'{int(lab["nl"]["irr"].sum())} blank '
          f'{int(lab["nl"]["blank"].sum())} | newline token ids '
          f'{NLTOK}',flush=True)
    at=m.transformer.h[LJ].attn

    def fh(mo,args,o_):
        y,v1r=o_; X2=args[0]; B=X2.shape[0]
        v1b=args[1] if args[1] is not None else v1r
        vv=at.c_v(X2).view(B,T,NH,128)
        vm=(1-at.lamb)*vv+at.lamb*v1b.view_as(vv)
        c2,s2=at.rotary(at.c_q(X2).view(B,T,NH,128))
        def r2(w):
            return are(F.rms_norm(w(X2).view(B,T,NH,128),(128,)),
                       c2,s2)
        qq,kk=r2(at.c_q),r2(at.c_k); q22,k22=r2(at.c_q2),r2(at.c_k2)
        sc=torch.einsum('bqhd,bkhd->bhqk',qq.float(),kk.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q22.float(),
                         k22.float())/128
        p2=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
        zz=torch.einsum('bhqk,bkhd->bhqd',p2,vm.float())
        zz[:,HD]=zz[:,HD].mean(dim=(0,1),keepdim=True)
        return (at.c_proj(zz.transpose(1,2).contiguous()
                .view(B,T,-1).to(X2.dtype)),v1r)

    CE={}; LGD={'nl_tok':[0.0,0],'best_other':[0.0,0]}
    for abl in (False,True):
        acc={}
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=[at.register_forward_hook(fh)] if abl else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            for h in hs: h.remove()
            sl=slice(i,i+B)
            groups={
              'nl':nl[sl],'ctrl':ctrl[sl],
              'nl_reg':lab['nl']['reg'][sl],
              'nl_irr':lab['nl']['irr'][sl],
              'nl_reg_nb':lab['nl']['reg'][sl]&~lab['nl']['blank'][sl],
              'nl_irr_nb':lab['nl']['irr'][sl]&~lab['nl']['blank'][sl],
              'nl_blank':lab['nl']['blank'][sl],
              'ctrl_reg':lab['ctrl']['reg'][sl],
              'ctrl_irr':lab['ctrl']['irr'][sl],
              'rest':~(nl[sl]|ctrl[sl])}
            for nm,mk in groups.items():
                a=acc.setdefault(nm,[0.0,0])
                a[0]+=float(ce[mk].sum()); a[1]+=int(mk.sum())
            # logit of the newline token vs best non-newline, at
            # newline targets only
            lgc=lg.cpu(); mk=nl[sl]
            if int(mk.sum()):
                sel=lgc[mk]
                ntl=sel[:,NLTOK].max(dim=-1).values
                oth=sel.clone(); oth[:,NLTOK]=-1e9
                bo=oth.max(dim=-1).values
                key='abl' if abl else 'base'
                LGD.setdefault(key,{'nl':[0.0,0],'oth':[0.0,0]})
                LGD[key]['nl'][0]+=float(ntl.sum())
                LGD[key]['nl'][1]+=len(ntl)
                LGD[key]['oth'][0]+=float(bo.sum())
                LGD[key]['oth'][1]+=len(bo)
        CE['abl' if abl else 'base']={
            k:(v[0]/max(v[1],1),v[1]) for k,v in acc.items()}
    d={k:round(CE['abl'][k][0]-CE['base'][k][0],4) for k in CE['base']}
    n={k:CE['base'][k][1] for k in CE['base']}
    lgm={k:{'nl':LGD[k]['nl'][0]/max(LGD[k]['nl'][1],1),
            'oth':LGD[k]['oth'][0]/max(LGD[k]['oth'][1],1)}
         for k in ('base','abl')}
    drop_nl=lgm['base']['nl']-lgm['abl']['nl']
    drop_oth=lgm['base']['oth']-lgm['abl']['oth']

    # descriptive: where the score mass goes at line breaks
    dest={c:[0.0,0] for c in ('recent_nl','line_start','pos0','self',
                              'prev','other')}
    cap={}
    hh=at.register_forward_pre_hook(
        lambda mo_,args: cap.__setitem__('X',args[0]))
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        X=cap['X']
        c2,s2=at.rotary(at.c_q(X).view(B,T,NH,128))
        def r2(w):
            return are(F.rms_norm(w(X).view(B,T,NH,128),(128,)),
                       c2,s2)
        qq,kk=r2(at.c_q),r2(at.c_k); q22,k22=r2(at.c_q2),r2(at.c_k2)
        sc=torch.einsum('bqd,bkd->bqk',qq[:,:,HD].float(),
                        kk[:,:,HD].float())/128
        sc2=torch.einsum('bqd,bkd->bqk',q22[:,:,HD].float(),
                         k22[:,:,HD].float())/128
        p2=((sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
        den=p2.abs().sum(-1).clamp_min(1e-6)
        for b in range(B):
            r=i+b
            keys=isnl[r].nonzero().squeeze(1).tolist()
            for q in nl[r].nonzero().squeeze(1).tolist():
                prev=[k for k in keys if k<q]
                rec=prev[-1] if prev else None
                ls=(rec+1) if (rec is not None and rec+1<=q) else None
                cls={}
                for k in range(q+1):
                    if k==rec: c='recent_nl'
                    elif k==ls: c='line_start'
                    elif k==0: c='pos0'
                    elif k==q: c='self'
                    elif k==q-1: c='prev'
                    else: c='other'
                    cls[c]=cls.get(c,0.0)+float(p2[b,q,k]/den[b,q])
                for c in dest:
                    dest[c][0]+=cls.get(c,0.0); dest[c][1]+=1
    hh.remove()
    DEST={c:round(v[0]/max(v[1],1),4) for c,v in dest.items()}

    ratio_ri=d['nl_reg']/max(d['nl_irr'],1e-4)
    ratio_nb=d['nl_reg_nb']/max(d['nl_irr_nb'],1e-4)
    ratio_ct=d['ctrl_reg']/max(abs(d['ctrl_irr']),1e-4)
    pa=d['nl_reg']>=2.0*d['nl_irr'] and d['nl_irr']>0
    pb=d['nl_reg_nb']>=1.5*d['nl_irr_nb'] and d['nl_irr_nb']>0
    pc=(drop_nl-drop_oth)>=0.10
    null_ok=ratio_ct<1.5
    out={'dce':d,'n':n,'ratio_reg_irr':round(ratio_ri,2),
         'ratio_reg_irr_nonblank':round(ratio_nb,2),
         'ratio_ctrl_reg_irr':round(ratio_ct,2),
         'logit':{'base':lgm['base'],'abl':lgm['abl'],
                  'drop_newline':round(drop_nl,4),
                  'drop_best_other':round(drop_oth,4)},
         'attn_destinations_at_linebreaks':DEST,
         'newline_token_ids':NLTOK,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'null_ok':bool(null_ok),'runtime_s':time.time()-t0}
    print('\ndCE by group:',{k:d[k] for k in
          ('nl','nl_reg','nl_irr','nl_reg_nb','nl_irr_nb',
           'nl_blank','ctrl','ctrl_reg','ctrl_irr','rest')})
    print('counts:',{k:n[k] for k in
          ('nl_reg','nl_irr','nl_reg_nb','nl_irr_nb','nl_blank')})
    print(f'logit drop: newline {drop_nl:+.4f} vs best other '
          f'{drop_oth:+.4f}')
    print('score mass at line breaks:',DEST)
    for nm,v in (('a',f'regular >= 2x irregular '
                      f'({ratio_ri:.2f}x)'),
                 ('b',f'survives excluding blanks '
                      f'({ratio_nb:.2f}x)'),
                 ('c',f'newline logit drops >=0.10 more than best '
                      f'competitor ({drop_nl-drop_oth:+.4f})')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    print(f"NULL (control positions show no rhythm effect, "
          f"{ratio_ct:.2f}x < 1.5): "
          f"{'ok' if null_ok else 'VIOLATED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
