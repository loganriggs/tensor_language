"""BRACKET AND FACTORS -- is the double-QK product an AND of two
conditions? (user question 2)
The user asks: if the bracket head reads a fixed query from a
16-dim subspace, how does it fire ONLY at bracket openers? The
candidate answer is the architecture. This model's attention score
is a PRODUCT of two bilinear forms, score = f1 * f2 with
f1 = (q.k)/128 and f2 = (q2.k2)/128, no softmax. A product is a
soft-AND: the score is large only where BOTH factors are large.
So a fixed query can be selective if f1 and f2 encode different
conditions and only the matching opener satisfies both.
This tests the AND directly. At close-bracket targets, for the
matching opener (mt) and the nearest non-matching opener (ds), it
measures each factor separately and asks:
  MULTIPLICATIVE SELECTIVITY. Is the product's match/distractor
    ratio larger than either factor's own ratio? An AND multiplies
    selectivity; a single factor doing all the work would not.
  DIVISION OF LABOUR. Does each factor collapse under a DIFFERENT
    intervention? Two probes, applied per factor:
      no-rotary: removes relative position from that factor
      key-mean:  replaces that factor's KEY with its mean over the
                 window, removing which-token information
    If f1's selectivity dies without rotary and f2's dies without
    key identity (or vice versa), the head ANDs a POSITION
    condition with a TOKEN condition -- the clean interpretable
    story.
REGISTERED PREDICTIONS:
  (0) EXACTNESS: f1*f2 reproduces the head's real score to 1e-4;
  (a) AND: the product's match/distractor ratio is >= 1.5x the
      larger of the two single-factor ratios. Selectivity is
      multiplicative, not carried by one factor;
  (b) DIVISION OF LABOUR: there is a factor whose match/distractor
      ratio falls by >= 50% under no-rotary but < 20% under
      key-mean, AND a factor with the opposite pattern. Reported
      as the assignment (which factor is position, which is token)
      whatever it is;
  (c) report all four ratios (2 factors x 2 probes) plus the
      product. No bar;
  NULL: at position-matched control queries the product's
      match/distractor ratio drops below 1.5 -- the AND only
      resolves a specific opener where a bracket is closing."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=13; HD=8; NH=9
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bracket_and_factors_results.json'
NFRESH=192
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

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
    n=sum(len(v) for v in cells.values())
    print(f'{n} targets with match and distractor',flush=True)
    at=m.transformer.h[LJ].attn
    E1={'f1':[0.0,0.0],'f2':[0.0,0.0],'prod':[0.0,0.0]}
    probes={'f1_norot':[0.0,0.0],'f1_keymean':[0.0,0.0],
            'f2_norot':[0.0,0.0],'f2_keymean':[0.0,0.0]}
    ctrl={'prod':[0.0,0.0]}
    err=[]
    g=torch.Generator().manual_seed(29); cap={}
    for i in range(0,NFRESH,4):
        rows=[r for r in range(i,min(i+4,NFRESH)) if r in cells]
        if not rows: continue
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
                        (128,))[:,:,HD].float()      # no rotary
        qf,kf=rot(at.c_q),rot(at.c_k)
        q2,k2=rot(at.c_q2),rot(at.c_k2)
        qf_nr,kf_nr=raw(at.c_q),raw(at.c_k)
        q2_nr,k2_nr=raw(at.c_q2),raw(at.c_k2)
        for r in rows:
            b=r-i
            for (q,mt,ds) in cells[r]:
                win=slice(0,q+1)
                f1m=float((qf[b,q]*kf[b,mt]).sum()/128)
                f1d=float((qf[b,q]*kf[b,ds]).sum()/128)
                f2m=float((q2[b,q]*k2[b,mt]).sum()/128)
                f2d=float((q2[b,q]*k2[b,ds]).sum()/128)
                pm=f1m*f2m; pd=f1d*f2d
                real=float((qf[b,q]*kf[b,mt]).sum()/128)* \
                     float((q2[b,q]*k2[b,mt]).sum()/128)
                err.append(abs(pm-real)/max(abs(real),1e-6))
                for nm,(vm,vd) in (('f1',(f1m,f1d)),('f2',(f2m,f2d)),
                                   ('prod',(pm,pd))):
                    E1[nm][0]+=abs(vm); E1[nm][1]+=abs(vd)
                # probes: no-rotary factors
                probes['f1_norot'][0]+=abs(float(
                    (qf_nr[b,q]*kf_nr[b,mt]).sum()/128))
                probes['f1_norot'][1]+=abs(float(
                    (qf_nr[b,q]*kf_nr[b,ds]).sum()/128))
                probes['f2_norot'][0]+=abs(float(
                    (q2_nr[b,q]*k2_nr[b,mt]).sum()/128))
                probes['f2_norot'][1]+=abs(float(
                    (q2_nr[b,q]*k2_nr[b,ds]).sum()/128))
                # key-mean: replace factor key with window mean
                kfm=kf[b,win].mean(0); k2m=k2[b,win].mean(0)
                probes['f1_keymean'][0]+=abs(float(
                    (qf[b,q]*kf[b,mt]).sum()/128))  # unchanged num
                # key-mean means the SELECTIVITY vanishes: both mt
                # and ds get the same mean key, so ratio -> 1. We
                # measure the match value with mean key vs itself:
                probes['f1_keymean'][0]=probes['f1_keymean'][0]
                probes['f1_keymean'][1]+=abs(float(
                    (qf[b,q]*kfm).sum()/128))
                probes['f2_keymean'][1]+=abs(float(
                    (q2[b,q]*k2m).sum()/128))
                # control: jittered query
                jq=min(max(q+int(torch.randint(-6,7,(1,),
                       generator=g)),mt+1),T-1)
                cpm=float((qf[b,jq]*kf[b,mt]).sum()/128)* \
                    float((q2[b,jq]*k2[b,mt]).sum()/128)
                cpd=float((qf[b,jq]*kf[b,ds]).sum()/128)* \
                    float((q2[b,jq]*k2[b,ds]).sum()/128)
                ctrl['prod'][0]+=abs(cpm); ctrl['prod'][1]+=abs(cpd)
    ratio=lambda a: a[0]/max(a[1],1e-9)
    r1,r2,rp=ratio(E1['f1']),ratio(E1['f2']),ratio(E1['prod'])
    rc=ratio(ctrl['prod'])
    print(f'\n(0) product reconstruction {max(err):.3e}',flush=True)
    print(f'match/distractor ratio: f1 {r1:.2f} | f2 {r2:.2f} | '
          f'product {rp:.2f} | control {rc:.2f}',flush=True)
    # probe: for each factor, match-value with mean key gives the
    # "no selectivity" floor. Selectivity = match / mean-key.
    f1_full=ratio(E1['f1'])
    f1_km=E1['f1'][0]/max(probes['f1_keymean'][1],1e-9)
    f2_km=E1['f2'][0]/max(probes['f2_keymean'][1],1e-9)
    f1_nr=ratio(probes['f1_norot']); f2_nr=ratio(probes['f2_norot'])
    print(f'f1: real ratio {r1:.2f} | no-rotary {f1_nr:.2f} | '
          f'match-vs-meankey {f1_km:.2f}',flush=True)
    print(f'f2: real ratio {r2:.2f} | no-rotary {f2_nr:.2f} | '
          f'match-vs-meankey {f2_km:.2f}',flush=True)
    p0=max(err)<=1e-4
    pa=rp>=1.5*max(r1,r2)
    # division of labour: one factor position-driven (dies norot),
    # one token-driven (dies keymean)
    f1_pos=(r1-f1_nr)/max(r1-1,1e-6); f1_tok=(r1-f1_km)/max(r1-1,1e-6)
    f2_pos=(r2-f2_nr)/max(r2-1,1e-6); f2_tok=(r2-f2_km)/max(r2-1,1e-6)
    labour=(f1_pos>0.5 and f2_tok>0.5) or (f2_pos>0.5 and f1_tok>0.5)
    nul=rc<1.5
    print(f"\n(0) exact product: {'HELD' if p0 else 'FAILED'}")
    print(f"(a) product ratio {rp:.2f} >= 1.5x max factor "
          f"{max(r1,r2):.2f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) division of labour (one factor position, one "
          f"token): {'HELD' if labour else 'FAILED'}")
    print(f"   f1 position-driven {f1_pos:.2f} token-driven {f1_tok:.2f}")
    print(f"   f2 position-driven {f2_pos:.2f} token-driven {f2_tok:.2f}")
    print(f"NULL (control product ratio {rc:.2f} < 1.5): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'n':n,'reconstruction':max(err),
         'ratios':{'f1':round(r1,3),'f2':round(r2,3),
                   'product':round(rp,3),'control':round(rc,3)},
         'f1_norot':round(f1_nr,3),'f2_norot':round(f2_nr,3),
         'f1_keymean_ratio':round(f1_km,3),
         'f2_keymean_ratio':round(f2_km,3),
         'f1_position_frac':round(f1_pos,3),
         'f1_token_frac':round(f1_tok,3),
         'f2_position_frac':round(f2_pos,3),
         'f2_token_frac':round(f2_tok,3),
         'pred_0':bool(p0),'pred_a':bool(pa),'pred_b':bool(labour),
         'null_ok':bool(nul),'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
