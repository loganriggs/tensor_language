"""IOI CHAIN -- owner-graph structure for the IOI circuit (354:
mover a14.h4, first-head a5.h7, induction a5.h5). If the a5 heads
feed the mover SERIALLY, ablating both should be SUBadditive (the
joint drop is bounded by the shared path, not the sum); independent
parallel contributors are additive. Mean-ablation combos.
REGISTERED PREDICTIONS:
  (a) SERIAL: joint drop of {a5.h7, a14.h4} <= 0.75 x (sum of
      single drops);
  (b) CONTROL: two near-zero heads {a5.h0, a14.h8} stay additive:
      |joint - sum| <= 0.1;
  (c) induction leg {a5.h5, a14.h4} also subadditive (<= 0.8 x sum)."""
import json, time, itertools, sys, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ioi_chain_results.json'

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
    def drop(spec):
        hs=[]
        for li,hds in spec.items(): hs+=mean_heads(li,hds)
        return round(base-margin(hs),3)
    singles={'a5h7':drop({5:[7]}),'a5h5':drop({5:[5]}),
             'a14h4':drop({14:[4]}),'a5h0':drop({5:[0]}),
             'a14h8':drop({14:[8]})}
    print('singles:',singles,flush=True)
    joints={'a5h7+a14h4':drop({5:[7],14:[4]}),
            'a5h5+a14h4':drop({5:[5],14:[4]}),
            'a5h0+a14h8':drop({5:[0],14:[8]}),
            'a5h7+a5h5':drop({5:[7,5]}),
            'all3':drop({5:[7,5],14:[4]})}
    print('joints:',joints,flush=True)
    s1=singles['a5h7']+singles['a14h4']
    s2=singles['a5h5']+singles['a14h4']
    s3=singles['a5h0']+singles['a14h8']
    pa=joints['a5h7+a14h4']<=0.75*s1
    pb=abs(joints['a5h0+a14h8']-s3)<=0.1
    pc=joints['a5h5+a14h4']<=0.8*s2
    out={'base':round(base,3),'singles':singles,'joints':joints,
         'sums':{'a5h7+a14h4':round(s1,3),'a5h5+a14h4':round(s2,3),
                 'ctl':round(s3,3)},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) serial h7->h4 (joint<=0.75xsum): {'HELD' if pa else 'FAILED'}")
    print(f"(b) control additive: {'HELD' if pb else 'FAILED'}")
    print(f"(c) serial h5->h4 (<=0.8xsum): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
