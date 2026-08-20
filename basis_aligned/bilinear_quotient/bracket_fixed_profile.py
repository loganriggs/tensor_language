"""BRACKET FIXED PROFILE -- is the 'adaptation' real, or just a
fixed query's rotary profile?
559 showed the bracket head's query is effectively CONSTANT: a
single fixed vector (the bracket-position average) reproduces the
head free. That puts a caveat on 531's 'the pointer adapts':
a fixed query under rotary has a fixed look-back PROFILE over
relative offsets, and if it decays slowly it could reproduce 531's
per-distance numbers with no adaptation at all.
This decides it. The bracket-average FIXED query is priced by
distance bin -- match-share on the true matching opener, split by
how far back it is -- against the REAL per-position query, on the
same targets. If a fixed query with no per-position information
already reproduces the real query's distance profile, the head
does not adapt; it has one look-back profile and brackets that
fall in it get selected. If the real query beats the fixed one
specifically at long distance, a genuine per-position adaptation
survives there and is quantified.
Bins by match distance (query - opener): 1-2, 3-5, 6-11, 12+.
REGISTERED PREDICTIONS:
  (0) POPULATED: every bin holds >= 10 targets; thin bins are
      reported unevaluable, not scored;
  (a) THE FIXED QUERY REPRODUCES THE NEAR PROFILE: in the 1-2 and
      3-5 bins, the fixed query's match share is within 0.05 of
      the real query's. Near brackets need no adaptation;
  (b) THE TEST OF ADAPTATION: in the 6-11 and 12+ bins, report
      whether the real query beats the fixed query by >= 0.05
      match share. If it does NOT, 531's 'adapts' is refuted and
      the head is a fixed-profile look-back; if it does, real
      adaptation survives at range and equals that gap;
  (c) report both full profiles. No bar;
  NULL: a query fixed to the ALL-position mean (not the bracket
      mean) must do worse than the bracket-mean fixed query in
      every bin -- confirms the fixed query carries bracket-
      relevant direction, not just any constant."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_fixed_profile_results.json'
NFRESH=256
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}
BINS=[(1,2),(3,5),(6,11),(12,999)]

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    tgt=torch.zeros(NFRESH,T,dtype=torch.bool); match={}
    for r in range(NFRESH):
        stack=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s))
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                if mt is not None: tgt[r,q]=True; match[(r,q)]=mt
    at=m.transformer.h[LJ].attn
    # fixed queries: bracket-average and all-average pre-rotary
    qbsum=torch.zeros(128,device=DEV); qbn=0
    qasum=torch.zeros(128,device=DEV); qan=0
    cap={}
    for i in range(0,NFRESH,4):
        bb=fresh[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); B=bb.shape[0]
        hc=at.register_forward_pre_hook(
            lambda mo_,a_: cap.__setitem__('X',a_[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        hc.remove()
        X=cap['X']
        qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD] \
            .float()
        qasum+=qpre.reshape(-1,128).sum(0); qan+=qpre.reshape(-1,128).shape[0]
        for r in range(i,min(i+4,NFRESH)):
            if not tgt[r].any(): continue
            b=r-i
            for q in tgt[r].nonzero().squeeze(1).tolist():
                qbsum+=qpre[b,q]; qbn+=1
    qbrk=qbsum/max(qbn,1); qall=qasum/max(qan,1)

    def binof(d):
        for lo,hi in BINS:
            if lo<=d<=hi: return f'{lo}-{hi if hi<999 else "+"}'
        return None

    def profile(mode):
        acc={}
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
            qpre=F.rms_norm(at.c_q(X).view(B,T,NH,128),(128,))[:,:,HD] \
                .float()
            if mode=='real': qp=qpre
            elif mode=='brk': qp=qbrk[None,None].expand(B,T,128)
            else: qp=qall[None,None].expand(B,T,128)
            qrot=are(qp[:,:,None].expand(B,T,NH,128).contiguous(),
                     cq,sq)[:,:,HD].float()
            def rk(W):
                return are(F.rms_norm(W(X).view(B,T,NH,128),(128,)),
                           cq,sq)[:,:,HD].float()
            s1=torch.einsum('bqd,bkd->bqk',qrot,rk(at.c_k))/128
            s2=torch.einsum('bqd,bkd->bqk',rk(at.c_q2),rk(at.c_k2))/128
            sc=(s1*s2)*torch.tril(torch.ones(T,T,device=DEV))
            den=sc.abs().sum(-1).clamp_min(1e-6)
            for r in range(i,min(i+4,NFRESH)):
                if not tgt[r].any(): continue
                b=r-i
                for q in tgt[r].nonzero().squeeze(1).tolist():
                    mt=match[(r,q)]; bn=binof(q-mt)
                    if bn is None: continue
                    e=acc.setdefault(bn,[0.0,0])
                    e[0]+=abs(float(sc[b,q,mt]/den[b,q])); e[1]+=1
        return {k:(round(v[0]/max(v[1],1),4),v[1])
                for k,v in acc.items()}
    pr={m_:profile(m_) for m_ in ('real','brk','all')}
    order=[f'{lo}-{hi if hi<999 else "+"}' for lo,hi in BINS]
    print('bin       n   real    fixed(brk)  fixed(all)',flush=True)
    rows={}
    for bn in order:
        if bn not in pr['real']: continue
        rr=pr['real'][bn][0]; bk=pr['brk'].get(bn,(0,0))[0]
        al=pr['all'].get(bn,(0,0))[0]; n=pr['real'][bn][1]
        rows[bn]={'n':n,'real':rr,'brk':bk,'all':al,
                  'real_minus_brk':round(rr-bk,4)}
        print(f'{bn:>6} {n:>5} {rr:>7.4f} {bk:>10.4f} {al:>10.4f}',
              flush=True)
    json.dump(rows,open(OUT,'w'),indent=1)
    unpop=[b for b in order if b not in rows or rows[b]['n']<10]
    p0=not unpop
    near_ok=all(abs(rows[b]['real_minus_brk'])<0.05
                for b in ('1-2','3-5') if b in rows)
    far=[b for b in ('6-11','12-+') if b in rows]
    adapt={b:rows[b]['real_minus_brk'] for b in far}
    adapts=any(v>=0.05 for v in adapt.values())
    nul=all(rows[b]['brk']>=rows[b]['all'] for b in rows)
    print(f"\n(0) all bins populated: "
          f"{'HELD' if p0 else 'FAILED for '+str(unpop)}")
    print(f"(a) fixed query matches real near (1-5 tokens): "
          f"{'HELD' if near_ok else 'FAILED'}")
    print(f"(b) real beats fixed at range >=0.05: {adapt} -> "
          f"{'ADAPTS' if adapts else 'NO ADAPTATION (fixed profile)'}")
    print(f"NULL (bracket-mean beats all-mean in every bin): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'bins':rows,'bracket_mean_vs_all_mean_null':bool(nul),
         'adaptation_gap':adapt,'adapts':bool(adapts),
         'pred_0':bool(p0),'pred_a':bool(near_ok),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
