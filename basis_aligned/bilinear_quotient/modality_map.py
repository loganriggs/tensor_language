"""MODALITY MAP -- destination-first over the behaviour-leading
heads, to map the fixed-query account across the atlas properly.
566 tested modality on unverified targets and went 0/3; 567 fixed
the protocol (destination first, modality only on a verified
target). This applies the corrected protocol as a BATCH over the
behaviour-leading heads, so the result is an authoritative map of
which structural heads attend to an earlier referent (and are thus
candidates for positional discrimination) versus which are local/
detection heads with no referent.
For each head, PHASE 1 measures the signed score-mass share over
several candidate referent classes; the referent is whichever
named class is enriched >= 2x its position-matched control. PHASE 2
runs the recent-vs-second modality test ONLY on a verified
referent, with the rotary probe. Heads with no verified referent
are reported as 'no-referent' rather than forced into a modality.
Heads (behaviour-leading, from behaviour_atlas2 + head_atlas):
  open_bracket  a17.h2   candidate referent: recent open bracket
  colon         a15.h?   candidate: recent colon / sentence end
  capitalized   a15.h3   candidate: recent newline / sentence end
  digit         a8.h3    candidate: recent digit
Candidate referent classes measured for every head: recent
same-class, recent newline, recent sentence-end, prev, self.
REGISTERED PREDICTIONS:
  (0) each head has >= 30 targets;
  (a) THE MAP IS INFORMATIVE: at least 2 of the 4 heads have a
      verified referent (some named class >= 2x its control). This
      is a weak bar -- the point is the map, not a pass;
  (b) VERIFIED REFERENTS GET A MODALITY: for each head with a
      verified referent, report positional (ratio >= 2, rotary-
      collapses) or detection (ratio < 1.5);
  (c) report the full table -- referent, its enrichment, modality
      -- for every head. No bar;
  NULL: a head whose only enriched class is 'self' or 'prev'
      (generic) is reported no-referent, not positional -- generic
      attention is not a referent."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'modality_map_results.json'
NFRESH=256
OPENS={'(':')','[':']','{':'}'}

def openb(t): return cl.d1(int(t)).strip() in ('(','[','{')
def iscolon(t): return cl.d1(int(t)).strip() in (':',';')
def iscap(t):
    z=cl.d1(int(t)).strip(); return bool(z) and z[:1].isupper()
def isdig(t):
    z=cl.d1(int(t)).strip(); return bool(z) and z[0].isdigit()
def isnl(t): return chr(10) in cl.d1(int(t))
def issent(t): return cl.d1(int(t)).strip() in ('.','!','?')

# behaviour -> (target fn, same-class key fn, head)
TASKS=[('open_bracket',openb,openb,(17,2)),
       ('colon',iscolon,iscolon,(15,4)),
       ('capitalized',iscap,isnl,(15,3)),
       ('digit',isdig,isdig,(8,3))]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    # precompute class key positions per row
    NLk=[[q for q in range(T) if isnl(cur[r,q])] for r in range(NFRESH)]
    SEk=[[q for q in range(T) if issent(cur[r,q])] for r in range(NFRESH)]
    g=torch.Generator().manual_seed(29); res={}
    for beh,tgtfn,samefn,(LJ,HD) in TASKS:
        samek=[[q for q in range(T) if samefn(cur[r,q])]
               for r in range(NFRESH)]
        tgt=torch.zeros(NFRESH,T,dtype=torch.bool); info={}
        for r in range(NFRESH):
            for q in range(T):
                if tgtfn(nxt[r,q]):
                    sp=[k for k in samek[r] if k<q]
                    info[(r,q)]={
                      'recent_same':sp[-1] if sp else None,
                      'second_same':sp[-2] if len(sp)>=2 else None,
                      'recent_nl':max([k for k in NLk[r] if k<q],
                                      default=None),
                      'recent_se':max([k for k in SEk[r] if k<q],
                                      default=None)}
                    tgt[r,q]=True
        ctrl=torch.zeros_like(tgt); cinfo={}
        for r in range(NFRESH):
            k=int(tgt[r].sum())
            if k==0: continue
            pos=tgt[r].nonzero().squeeze(1)
            j=(torch.randint(-6,7,(k,),generator=g)+pos).clamp(1,T-1)
            for a,q in zip(pos.tolist(),j.tolist()):
                if tgt[r,q]: continue
                d=info[(r,a)]
                cinfo[(r,q)]={kk:(vv if vv is not None and vv<q else None)
                              for kk,vv in d.items()}
                ctrl[r,q]=True
        n=int(tgt.sum())
        if n<30:
            res[beh]={'head':f'{LJ}.{HD}','n':n,'status':'too few'}
            print(f'{beh} ({LJ}.{HD}): {n} targets, skipped',flush=True)
            continue
        at=m.transformer.h[LJ].attn; cap={}
        GR=['recent_same','recent_nl','recent_se','prev','self']
        acc={'t':{k:[0.0,0] for k in GR},'c':{k:[0.0,0] for k in GR}}
        mod={'real':[0.0,0.0,0],'norot':[0.0,0.0,0]}
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            hc=at.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0]))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            hc.remove()
            X=cap['X']
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            def rot(W): return are(F.rms_norm(W(X).view(B,T,NH,128),
                            (128,)),cq,sq)[:,:,HD].float()
            def raw(W): return F.rms_norm(W(X).view(B,T,NH,128),
                            (128,))[:,:,HD].float()
            for tag,fn in (('real',rot),('norot',raw)):
                s1=torch.einsum('bqd,bkd->bqk',fn(at.c_q),fn(at.c_k))/128
                s2=torch.einsum('bqd,bkd->bqk',fn(at.c_q2),fn(at.c_k2))/128
                p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
                den=p2.abs().sum(-1).clamp_min(1e-6)
                if tag=='real':
                    for nm,mask,src in (('t',tgt,info),('c',ctrl,cinfo)):
                        for r in range(i,min(i+4,NFRESH)):
                            if not mask[r].any(): continue
                            b=r-i
                            for q in mask[r].nonzero().squeeze(1).tolist():
                                d=src.get((r,q))
                                if d is None: continue
                                sh=lambda k:float(p2[b,q,k]/den[b,q])
                                for gk,dk in (('recent_same','recent_same'),
                                    ('recent_nl','recent_nl'),
                                    ('recent_se','recent_se')):
                                    kk=d.get(dk)
                                    if kk is not None:
                                        acc[nm][gk][0]+=sh(kk); acc[nm][gk][1]+=1
                                if q>0: acc[nm]['prev'][0]+=sh(q-1); acc[nm]['prev'][1]+=1
                                acc[nm]['self'][0]+=sh(q); acc[nm]['self'][1]+=1
                # modality on recent_same vs second_same
                for r in range(i,min(i+4,NFRESH)):
                    if not tgt[r].any(): continue
                    b=r-i
                    for q in tgt[r].nonzero().squeeze(1).tolist():
                        d=info[(r,q)]
                        if d['recent_same'] is None or d['second_same'] is None:
                            continue
                        mod[tag][0]+=abs(float(p2[b,q,d['recent_same']]/den[b,q]))
                        mod[tag][1]+=abs(float(p2[b,q,d['second_same']]/den[b,q]))
                        mod[tag][2]+=1
        S={nm:{k:round(acc[nm][k][0]/max(acc[nm][k][1],1),4) for k in GR}
           for nm in ('t','c')}
        # verified referent = a NAMED lookback class (not prev/self)
        named={k:S['t'][k] for k in ('recent_same','recent_nl','recent_se')}
        ref=max(named,key=lambda k:abs(named[k]))
        verified=abs(S['t'][ref])>=2*abs(S['c'][ref])
        entry={'head':f'{LJ}.{HD}','n':n,'shares':S['t'],
               'control':{k:S['c'][k] for k in named},
               'referent':ref if verified else None,
               'referent_enrichment':round(
                   abs(S['t'][ref])/max(abs(S['c'][ref]),1e-6),2)}
        if verified and mod['real'][2]>=10:
            rr=(mod['real'][0]/mod['real'][2])/max(
                mod['real'][1]/mod['real'][2],1e-9)
            nr=(mod['norot'][0]/mod['norot'][2])/max(
                mod['norot'][1]/mod['norot'][2],1e-9)
            entry['modality']=('positional' if (rr>=2 and (rr-nr)/max(rr,1e-9)>=0.4)
                               else 'detection' if rr<1.5 else 'mixed')
            entry['ratio']=round(rr,2); entry['ratio_norot']=round(nr,2)
        else:
            entry['modality']='no-referent'
        res[beh]=entry
        print(f"{beh} ({LJ}.{HD}): referent={entry['referent']} "
              f"(enrich {entry['referent_enrichment']}x) -> "
              f"{entry['modality']}"
              +(f" ratio {entry.get('ratio')}->{entry.get('ratio_norot')}"
                if 'ratio' in entry else ''),flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    verified_ct=sum(1 for v in res.values()
                    if v.get('referent') is not None)
    pa=verified_ct>=2
    print(f"\n(a) >= 2 heads have a verified referent: "
          f"{verified_ct}/4 -> {'HELD' if pa else 'FAILED'}")
    print(f"(c) map:")
    for beh,v in res.items():
        print(f"   {beh:>13}: {v.get('modality','-')}")
    out={'map':res,'n_verified':verified_ct,'pred_a':bool(pa),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
