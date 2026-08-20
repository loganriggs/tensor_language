"""HEAD ATLAS -- 493: a per-head reference table for all 162
heads, built to SOURCE future circuits instead of discovering them
from scratch. The two circuits this program closed to code level
(induction, the position-0 bias) both came from chasing a single
anomalous head; the damage-cluster census produced none (472). So
the productive move is to make anomalies cheap to spot.
For every head, in one sweep: its 4-token-window need overall and
at match positions, its top-read motif profile (previous token,
self, position 0, successor-of-a-repeat, other), and the token
classes its reads are enriched for. Deletion cost is joined from
the existing full cost map (429).
REGISTERED PREDICTIONS:
  (a) NEW ANOMALIES: at least 3 heads besides 5.7 and 12.6 have a
      match-position window need >= 0.05 -- i.e. the atlas finds
      long-range heads the sweeps have not already named;
  (b) MOTIFS PARTITION: >= 60% of heads have a dominant motif
      taking more than 40% of their top reads;
  (c) LINK: heads with high match-window-need are enriched for
      successor-reading relative to the rest (mean successor rate
      at least 2x)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'head_atlas_results.json'
NR=8

def cls_of(tok):
    s=cl.d1(int(tok)); st=s.strip()
    return ('punct' if (bool(st) and not any(c.isalnum()
                                             for c in st))
            else 'newline' if chr(10) in s
            else 'digit' if st.isdigit()
            else 'capitalized' if (s.startswith(' ') and bool(st)
                                   and st[:1].isupper())
            else 'space_word' if (s.startswith(' ')
                                  and st.isalpha())
            else 'subword' if st.isalpha() else 'other')

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    cost=json.load(open(PT+'head_cost_map_results.json'))['heads']
    # ---- pass 1: motifs and read classes, all heads at once ----
    motif={f'{l}.{h}':{'prev':0,'self':0,'pos0':0,'succ':0,
                       'other':0,'n':0}
           for l in range(18) for h in range(9)}
    rcls={f'{l}.{h}':{} for l in range(18) for h in range(9)}
    basec={}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hs=[m.transformer.h[l].attn.register_forward_pre_hook(
            (lambda l: lambda mo_,a_: cap.__setitem__(l,a_[0]))(l))
            for l in range(18)]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for b in range(B):
            for q in range(T):
                c=cls_of(int(ROWS[i+b,q]))
                basec[c]=basec.get(c,0)+1
        tril=torch.tril(torch.ones(T,T,device=DEV))
        for l in range(18):
            at=m.transformer.h[l].attn; X=cap[l]
            cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
            def rr(w):
                return are(F.rms_norm(w(X).view(B,T,9,128),
                           (128,)),cos,sin)
            qf,kf=rr(at.c_q),rr(at.c_k)
            q2,k2=rr(at.c_q2),rr(at.c_k2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            pat=((sc*sc2)*tril).abs()
            am=pat.argmax(-1).cpu()
            for h in range(9):
                key=f'{l}.{h}'; s=motif[key]
                for b in range(B):
                    toks=ROWS[i+b,:T].tolist(); last={}
                    for q in range(8,T,2):
                        t=toks[q]
                        prev=last.get(t)
                        for qq in range(max(0,q-1),q+1):
                            last[toks[qq]]=qq
                        k=int(am[b,h,q])
                        s['n']+=1
                        if k==q: s['self']+=1
                        elif k==q-1: s['prev']+=1
                        elif k==0: s['pos0']+=1
                        elif prev is not None and k==prev+1:
                            s['succ']+=1
                        else: s['other']+=1
                        c=cls_of(int(ROWS[i+b,k]))
                        rcls[key][c]=rcls[key].get(c,0)+1
        print(f'motif batch {i} done ({time.time()-t0:.0f}s)',
              flush=True)
    # ---- pass 2: window need per head ----
    def win_run(l,h):
        tm=tn=0.0; nm_=nn_=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            if l is not None:
                at=m.transformer.h[l].attn
                def fh(mo_,args,o_,at=at,h=h):
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
                    pat=(sc*sc2)
                    z=torch.einsum('bhqk,bkhd->bhqd',pat*tril,
                                   vm.float())
                    z[:,h]=torch.einsum('bqk,bkd->bqd',
                                        pat[:,h]*win,
                                        vm[:,:,h].float())
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
            mk=torch.zeros(B,T,dtype=torch.bool)
            for b in range(B):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk[b,q]=True
                    last[t]=q
            tm+=float(ce[mk].sum()); nm_+=int(mk.sum())
            tn+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for hh in hs: hh.remove()
        return tm/max(nm_,1),tn/max(nn_,1)
    bm,bn=win_run(None,None)
    nb=sum(basec.values())
    baser={c:basec[c]/max(nb,1) for c in basec}
    atlas={}
    for l in range(18):
        for h in range(9):
            key=f'{l}.{h}'
            pm,pn=win_run(l,h)
            s=motif[key]; n=max(s['n'],1)
            prof={k:round(s[k]/n,3) for k in
                  ('prev','self','pos0','succ','other')}
            dom=max(prof,key=prof.get)
            tot=sum(rcls[key].values()) or 1
            enr={c:round((rcls[key].get(c,0)/tot)
                         /max(baser.get(c,1e-6),1e-6),2)
                 for c in baser}
            atlas[key]={'delete_cost':cost.get(key,{})
                        .get('dce_all'),
                        'window_match':round(pm-bm,4),
                        'window_overall':round(pn-bn,4),
                        'motif':prof,'dominant_motif':dom,
                        'dominant_share':prof[dom],
                        'read_class_enrichment':enr}
        print(f'layer {l} atlas done ({time.time()-t0:.0f}s)',
              flush=True)
        json.dump(atlas,open(OUT,'w'),indent=1)
    known={'5.7','12.6'}
    hits=[k for k,v in atlas.items()
          if v['window_match']>=0.05 and k not in known]
    domfrac=sum(1 for v in atlas.values()
                if v['dominant_share']>0.40)/len(atlas)
    hi=[v for k,v in atlas.items() if v['window_match']>=0.05]
    lo=[v for k,v in atlas.items() if v['window_match']<0.05]
    ms=(sum(v['motif']['succ'] for v in hi)/max(len(hi),1))
    ml=(sum(v['motif']['succ'] for v in lo)/max(len(lo),1))
    pa=len(hits)>=3; pb=domfrac>=0.60; pc=ms>=2*max(ml,1e-6)
    out={'atlas':atlas,'new_long_range_heads':hits,
         'dominant_motif_fraction':round(domfrac,3),
         'succ_rate_high_window':round(ms,4),
         'succ_rate_low_window':round(ml,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'new long-range heads: {hits}')
    print(f'dominant-motif fraction {domfrac:.2f} | succ rate '
          f'high {ms:.4f} vs low {ml:.4f}')
    for nm,v in (('a','>=3 new long-range heads found'),
                 ('b','>=60% of heads have a dominant motif'),
                 ('c','long-range heads read successors more')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
