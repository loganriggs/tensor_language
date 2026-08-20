"""BRACKET DISTANCE -- is the pointer positional or symbolic?
527 narrowed it to two possibilities. Head 13.8's entire effect
passes through one score cell (522), the pointer puts 15.8x more
mass on the matching opener than on any other bracket (523), and
the exact pair decomposition shows the SAME writer pairs dominate
at the match and at the distractor -- they simply evaluate 4.5x
larger at the match (527). Both cells share a query, so the whole
discrimination is key-side. But seven of the ten leading pairs
have WTE on the key side, and two "(" tokens have identical
embeddings, so the difference must come from one of:
  ROTARY -- the same embedding at a different position is a
    different rotated vector, which would make this head a
    RELATIVE-DISTANCE selector rather than a symbolic matcher;
  m12 -- the one non-wte key writer in the top ten, which can
    differ between the two openers because it sees context.
These are decidable by removing each and re-measuring the same
match-versus-distractor contrast.
Arms, all measuring the absolute score-mass share on the matching
opener and on the nearest non-matching opener at the same query:
  real          untouched (reference: 523 measured 0.367 vs 0.023
                on natural text; this run recomputes both on its
                own sample)
  no_rotary     head 13.8's queries and keys use unrotated
                vectors, so position information is removed from
                its score while all content is preserved
  kill_key_wte  the wte contribution to the KEY side is replaced
                by its mean over positions, so both openers lose
                their token identity but keep position and context
  kill_key_m12  the same for m12, the minority key writer
  kill_key_m10  control: a writer that appears on the QUERY side of
                the leading pairs but not the key side, so removing
                it from the key should do little
Also reported, because it is the direct form of the hypothesis:
the distribution of match distances (q - k) across targets, and
the head's mean score share on openers as a function of distance.
REGISTERED PREDICTIONS:
  (0) THE ARMS FIRE: each arm changes the head's score at target
      cells by a relative amount above 1e-6. An arm that is
      exactly zero did not fire and its result is void (the
      permanent guard from 446);
  (a) POSITIONAL: with rotary disabled, the ratio of match share
      to distractor share falls below 3.0, from a reference above
      10. This is the distance-selector claim;
  (b) NOT PURE LEXICAL LOOKUP: with the key-side wte removed, the
      ratio stays above 3.0 -- i.e. stripping token identity from
      the keys does NOT destroy the pointer. If both (a) and (b)
      hold, the head selects by position and uses the token only
      to know a bracket is there;
  (c) CONTROL: kill_key_m10 leaves the ratio above 10, and
      kill_key_m12 is reported either way.
  NULL: match distances must actually VARY -- if every match sits
      at the same distance, a fixed-offset head would look
      identical to an adaptive one and the run cannot separate
      them. The interquartile range of match distance must be at
      least 3 tokens; otherwise the run is reported as
      uninformative rather than scored.
Absolute pairs reported alongside every ratio (the rule from 520)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_distance_results.json'
NFRESH=192
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

    res={}; fired={}
    ref_raw=None
    for arm in ARMS:
        s,rawmag=shares(arm)
        if s is None: continue
        if arm=='real': ref_raw=rawmag
        else:
            rel=abs(rawmag-ref_raw)/max(abs(ref_raw),1e-12)
            fired[arm]=rel>1e-6
        res[arm]={'match':round(s['match'],4),
                  'dist':round(s['dist'],4),
                  'ratio':round(s['match']/max(s['dist'],1e-6),2),
                  'raw_score_mag':round(rawmag,5)}
        print(f"{arm:>14}: match {res[arm]['match']:.4f} distractor "
              f"{res[arm]['dist']:.4f} ratio {res[arm]['ratio']} "
              f"| |score| {rawmag:.4g}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    dead=[a for a,f in fired.items() if not f]
    if dead:
        print(f'*** ARMS THAT NEVER FIRED: {dead} -- void for those '
              f'arms ***')
    R=lambda a: res[a]['ratio'] if a in res else float('nan')
    p0=not dead
    va,_=cl.score_bar('a',3.0-R('no_rotary'),1e-9)
    vb,_=cl.score_bar('b',R('kill_key_wte')-3.0,1e-9)
    vc='HELD' if R('kill_key_m10')>=10.0 else 'FAILED'
    nul=iqr>=3
    print(f"(c) control kill_key_m10 ratio {R('kill_key_m10')} "
          f">= 10: {vc}  | kill_key_m12 ratio {R('kill_key_m12')}")
    print(f"NULL (match-distance IQR {iqr} >= 3 tokens): "
          f"{'ok' if nul else 'VIOLATED -- run uninformative'}")
    out={'arms':res,'n_cells':len(allc),
         'match_distance_median':dists[len(dists)//2],
         'match_distance_iqr':iqr,
         'match_distance_range':[dists[0],dists[-1]],
         'arms_never_fired':dead,
         'pred_0':bool(p0),'pred_a':va=='HELD','pred_b':vb=='HELD',
         'pred_c':vc=='HELD','null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
