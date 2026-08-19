"""IOI HEADS v2 -- zero-deletion gave UNIFORM per-head drops in a14
(all nine ~1.64; recompute verified exact, delta 0.0000). Hypothesis:
zeroing 1/9 of c_proj's input is an off-manifold magnitude shock, so
the uniform drop measures scale sensitivity, not head content. v2
uses within-prompt MEAN ablation per head (replace head z at every
position with its mean over that prompt's positions: kills content,
keeps typical magnitude), on the two attention owners a14 and a5.
REGISTERED PREDICTIONS:
  (a) mean-ablation drops are DIFFERENTIATED: max/min per-head drop
      ratio >=3 in a14 (vs 1.0 under zeroing);
  (b) top-2 heads carry >=60% of the full-layer mean-ablation drop;
  (c) in a5, the induction head (5,5) or a first head (5,2)/(5,7)
      is the top head (IOI needs duplicate-name detection)."""
import json, time, itertools, sys, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ioi_heads2_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    enc=cl.enc()
    names=[' Mary',' John',' Anna',' Peter',' Sarah',' Tom',
           ' Alice',' Bob']
    pairs=list(itertools.combinations(names,2))[:8]
    TEMPL=['When{A} and{B} went to the store,{B} gave the drink to',
           'When{A} and{B} got home,{B} handed the keys to',
           'After{A} and{B} left the party,{B} gave the coat to',
           'Then{A} and{B} went to the park, and{B} threw the ball to',
           'While{A} and{B} were cooking,{B} passed the salt to',
           'When{A} and{B} finished lunch,{B} gave the bill to']
    prompts=[]
    for A,B in pairs:
        for a,b in ((A,B),(B,A)):
            for tpl in TEMPL:
                prompts.append((tpl.replace('{A}',a).replace('{B}',b),
                                enc.encode(a)[0],enc.encode(b)[0]))
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    def mean_heads(li,hds):
        at=m.transformer.h[li].attn
        def fh(mo_,args,out,at=at,hds=hds):
            y,v1r=out
            X=args[0]; v1=args[1] if args[1] is not None else v1r
            Bb,T=X.shape[0],X.shape[1]
            v=at.c_v(X).view(Bb,T,9,128)
            vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
            cos,sin=at.rotary(at.c_q(X).view(Bb,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(Bb,T,9,128),(128,))
            kf=F.rms_norm(at.c_k(X).view(Bb,T,9,128),(128,))
            qf,kf=are(qf,cos,sin),are(kf,cos,sin)
            q2=F.rms_norm(at.c_q2(X).view(Bb,T,9,128),(128,))
            k2=F.rms_norm(at.c_k2(X).view(Bb,T,9,128),(128,))
            q2,k2=are(q2,cos,sin),are(k2,cos,sin)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
            pat=(sc*s2)*torch.tril(torch.ones(T,T,device=DEV))
            z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
            for hd in hds:
                z[:,hd]=z[:,hd].mean(1,keepdim=True)
            yn=at.c_proj(z.transpose(1,2).contiguous()
                         .view(Bb,T,-1).to(X.dtype))
            return (yn,v1r)
        return [at.register_forward_hook(fh)]
    def margin(hooks=()):
        ms=[]
        for txt,ti,ts in prompts:
            ids=torch.tensor(enc.encode(txt))[None,:].to(DEV)
            x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F.rms_norm(x,(D,)))/30)).float()[0,-1]
            ms.append(float(lg[ti]-lg[ts]))
        for h in hooks: h.remove()
        return sum(ms)/len(ms)
    base=margin()
    print(f'base {base:+.3f}',flush=True)
    out={'base':round(base,3)}
    for li in (14,5):
        per=[]
        for hd in range(9):
            per.append((hd,round(base-margin(mean_heads(li,[hd])),3)))
        per.sort(key=lambda kv:-kv[1])
        full=round(base-margin(mean_heads(li,list(range(9)))),3)
        top2=round(base-margin(mean_heads(li,[per[0][0],
                                              per[1][0]])),3)
        out[f'a{li}']={'per_head':per,'full':full,'top2':top2}
        print(f'a{li}: {per} | full {full} | top2 {top2}',flush=True)
    p14=out['a14']['per_head']
    mx=max(v for _,v in p14); mn=min(v for _,v in p14)
    pa=mn<=0 or mx/max(mn,1e-3)>=3
    pb=out['a14']['top2']>=0.6*max(out['a14']['full'],1e-3)
    top5=out['a5']['per_head'][0][0]
    pc=top5 in (5,2,7)
    out.update({'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)})
    print(f"(a) differentiated (max/min>=3): {'HELD' if pa else 'FAILED'}")
    print(f"(b) top2 >=60% full a14: {'HELD' if pb else 'FAILED'}")
    print(f"(c) a5 top head in (5,2,7): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
