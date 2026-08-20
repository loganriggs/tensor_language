"""BRACKET RANGE -- does the positional pointer adapt?
529 corrected the bracket story. Disabling rotary for head 13.8
collapses the match-versus-distractor ratio from 6.48 to 1.08 --
the two become indistinguishable -- while removing the token
embedding from the key side leaves it at 6.81. The discrimination
is POSITION, and token identity contributes nothing to which key
is chosen. And the matching opener sits a median of TWO tokens
back in this corpus, so "points at the match" and "points two
tokens back" were largely the same claim.
One question survives that, and it decides whether "pointer" is
the right word at all. Match distances range from 1 to 32. If the
head still lands on the match when it is eight tokens back, it
computes a distance rather than having one baked in, and an
adaptive positional pointer is a real mechanism. If the share
collapses to distractor level as distance grows, the head
implements a fixed short-range rule, the matcher language should
be dropped from the ledger, and the 0.825-nat effect is explained
by "brackets usually close almost immediately".
Same measurement as 529, split by distance instead of pooled, on
twice the rows so the far bins are populated. Bins: 1-2, 3-5,
6-11, 12+.
REGISTERED PREDICTIONS:
  (0) POPULATED: every bin holds at least 12 cells, checked before
      scoring. Bins that do not are reported as unevaluable rather
      than scored -- the rule that has bitten this program four
      times (465, 500, 513, and the close_quote row of 518);
  (a) ADAPTIVE: in the 6-11 bin the match share is at least 3x the
      distractor share at the same distance. This is the claim
      that the pointer follows the match out to real distances;
  (b) NO CLIFF: the match share in the 6-11 bin is at least half
      the match share in the 1-2 bin. A pointer that adapts should
      weaken gradually, not fall off;
  (c) ROTARY STILL EXPLAINS IT: with rotary disabled, the ratio
      falls below 2.0 in EVERY populated bin, not just pooled.
  NULL: the distractor share must not itself rise with distance --
      if it does, the contrast is being manufactured by the
      distance split rather than measured.
If (a) holds and (b) fails, the honest description is a pointer
with a short but real range, and the number where it breaks is
worth recording."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_range_results.json'
NFRESH=384
OPENS={'(':')','[':']','{':'}'}
CLOSES={v:k for k,v in OPENS.items()}

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    cur=fresh[:,:256]; nxt=fresh[:,1:257]
    cells={}
    for r in range(NFRESH):
        stack=[]; opos=[]
        for q in range(T):
            s=cl.d1(int(cur[r,q])).strip()
            if s in OPENS: stack.append((q,s)); opos.append(q)
            elif s in CLOSES and stack: stack.pop()
            n=cl.d1(int(nxt[r,q])).strip()
            if n in CLOSES:
                mt=None
                for p,ch in reversed(stack):
                    if OPENS[ch]==n: mt=p; break
                ds=[p for p in opos if p<=q and p!=mt]
                if mt is not None and ds:
                    cells.setdefault(r,[]).append((q,mt,ds[-1]))
    allc=[(r,q,mt,ds) for r,v in cells.items() for (q,mt,ds) in v]
    dists=sorted(q-mt for _,q,mt,_ in allc)
    if not dists:
        print('*** no cells -- VOID ***')
        json.dump({'void':'no cells'},open(OUT,'w'),indent=1); return
    iqr=dists[int(0.75*len(dists))]-dists[int(0.25*len(dists))]
    print(f'{len(allc)} cells | match distance median '
          f'{dists[len(dists)//2]} IQR {iqr} range '
          f'{dists[0]}-{dists[-1]}',flush=True)
    at=m.transformer.h[LJ].attn
    WR=['wte']+[f'{k}{l}' for l in range(LJ) for k in ('a','m')]
    ARMS=['real','no_rotary','kill_key_wte','kill_key_m12',
          'kill_key_m10']

    def shares(arm):
        out={'match':[0.0,0],'dist':[0.0,0]}
        raw=[]
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
            if not rows: continue
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            outs={}; hs=[]
            if arm.startswith('kill_key'):
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
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            X=cap['X']
            Xk=X
            if arm.startswith('kill_key'):
                w=arm.replace('kill_key_','')
                parts=cl.writer_parts(LJ,E,outs,'a')
                if w not in parts:
                    print(f'*** {w} not a writer into a{LJ} -- '
                          f'arm void ***'); return None,None
                tot=sum(parts.values())
                p=parts[w]
                Xk=F.rms_norm(tot-p+p.mean(dim=(0,1),keepdim=True),
                              (D,)).to(X.dtype)
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            def rot(W,Z):
                v=F.rms_norm(W(Z).view(B,T,NH,128),(128,))
                if arm=='no_rotary':
                    return v[:,:,HD].float()
                return are(v,cq,sq)[:,:,HD].float()
            s1=torch.einsum('bqd,bkd->bqk',rot(at.c_q,X),
                            rot(at.c_k,Xk))/128
            s2=torch.einsum('bqd,bkd->bqk',rot(at.c_q2,X),
                            rot(at.c_k2,Xk))/128
            p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
            den=p2.abs().sum(-1).clamp_min(1e-6)
            for r in rows:
                b=r-i
                for (q,mt,ds) in cells[r]:
                    out['match'][0]+=abs(float(p2[b,q,mt]/den[b,q]))
                    out['match'][1]+=1
                    out['dist'][0]+=abs(float(p2[b,q,ds]/den[b,q]))
                    out['dist'][1]+=1
                    raw.append(float(p2[b,q,mt]))
        return ({k:out[k][0]/max(out[k][1],1) for k in out},
                sum(abs(x) for x in raw)/max(len(raw),1))

    BINS=[(1,2),(3,5),(6,11),(12,999)]
    def binof(d):
        for lo,hi in BINS:
            if lo<=d<=hi: return f'{lo}-{hi if hi<999 else "+"}'
        return None
    def shares_binned(arm):
        out={}
        for i in range(0,NFRESH,4):
            rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
            if not rows: continue
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=bb.shape[0]
            outs={}; hs=[]
            cap={}
            hs.append(at.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0])))
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            X=cap['X']
            cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
            def rot(W,Z):
                v=F.rms_norm(W(Z).view(B,T,NH,128),(128,))
                if arm=='no_rotary': return v[:,:,HD].float()
                return are(v,cq,sq)[:,:,HD].float()
            s1=torch.einsum('bqd,bkd->bqk',rot(at.c_q,X),
                            rot(at.c_k,X))/128
            s2=torch.einsum('bqd,bkd->bqk',rot(at.c_q2,X),
                            rot(at.c_k2,X))/128
            p2=((s1*s2)*torch.tril(torch.ones(T,T,device=DEV))).cpu()
            den=p2.abs().sum(-1).clamp_min(1e-6)
            for r in rows:
                b=r-i
                for (q,mt,ds) in cells[r]:
                    bn=binof(q-mt)
                    if bn is None: continue
                    e=out.setdefault(bn,{'m':0.0,'d':0.0,'n':0})
                    e['m']+=abs(float(p2[b,q,mt]/den[b,q]))
                    e['d']+=abs(float(p2[b,q,ds]/den[b,q]))
                    e['n']+=1
        return {k:{'match':v['m']/max(v['n'],1),
                   'dist':v['d']/max(v['n'],1),'n':v['n']}
                for k,v in out.items()}
    real=shares_binned('real'); noro=shares_binned('no_rotary')
    order=[f'{lo}-{hi if hi<999 else "+"}' for lo,hi in BINS]
    print('\nbin      n    match   distractor  ratio | no-rotary ratio')
    res={}
    for bn in order:
        if bn not in real: continue
        r_=real[bn]; nr=noro.get(bn,{'match':0,'dist':1})
        rat=r_['match']/max(r_['dist'],1e-6)
        nrat=nr['match']/max(nr['dist'],1e-6)
        res[bn]={'n':r_['n'],'match':round(r_['match'],4),
                 'dist':round(r_['dist'],4),'ratio':round(rat,2),
                 'no_rotary_ratio':round(nrat,2)}
        print(f"{bn:>6} {r_['n']:>4} {r_['match']:8.4f} "
              f"{r_['dist']:11.4f} {rat:6.2f} | {nrat:6.2f}")
    unpop=[b for b in order if b not in res or res[b]['n']<12]
    p0=not unpop
    print(f"(0) every bin has >=12 cells: "
          f"{'HELD' if p0 else 'FAILED for '+str(unpop)}")
    FAR='6-11'; NEAR='1-2'
    va=vb=vc='UNEVALUABLE'
    if FAR in res and res[FAR]['n']>=12:
        va,_=cl.score_bar('a',res[FAR]['match']-3*res[FAR]['dist'],
                          1e-9)
        if NEAR in res:
            vb,_=cl.score_bar('b',res[FAR]['match']
                              -0.5*res[NEAR]['match'],1e-9)
    pops=[b for b in res if res[b]['n']>=12]
    vc='HELD' if all(res[b]['no_rotary_ratio']<2.0 for b in pops) \
       else 'FAILED'
    print(f"(c) rotary-off ratio < 2.0 in every populated bin: {vc}")
    ds=[res[b]['dist'] for b in pops]
    nul=(ds==sorted(ds,reverse=True)) or (max(ds)-min(ds)<0.02)
    print(f"NULL (distractor share does not rise with distance: "
          f"{[round(x,4) for x in ds]}): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'bins':res,'n_cells':len(allc),
         'match_distance_median':dists[len(dists)//2],
         'match_distance_range':[dists[0],dists[-1]],
         'unpopulated':unpop,'pred_0':bool(p0),
         'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
