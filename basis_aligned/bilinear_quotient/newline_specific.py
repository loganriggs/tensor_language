"""NEWLINE SPECIFICITY -- 495: step 1 of the behaviour-defined
attempt found concentration but NOT specificity (494). The top
five components carry 57.5% of the summed newline-target cost and
the leader is not the sink -- but every one of them costs MORE at
non-newline positions than at newline ones (m1: +0.673 newline
against +6.942 elsewhere, ratio 0.10; m3 0.64; m2 0.35; m0 0.29;
a1 0.64). They are the front of the model doing everything, not a
newline circuit.
So rank by SPECIFICITY rather than magnitude: which component's
damage is most concentrated ON newline targets relative to its own
damage elsewhere. That is the quantity a behaviour-defined circuit
needs, and step 1 measured the wrong one.
Scored on the same 36 components already measured, plus a
positional control: newline targets are unevenly distributed in a
document, so a component that simply matters more late or early in
a sequence could masquerade as newline-specific. The control
compares each component's newline ratio against its ratio for a
POSITION-MATCHED random target set of the same size.
REGISTERED PREDICTIONS:
  (a) SOMETHING IS SPECIFIC: at least one component has a
      newline/other damage ratio >= 2.0;
  (b) NOT THE FRONT: the most specific component is not one of
      m0-m3 (the front-of-model block that dominates magnitude);
  (c) POSITION CONTROL: the most specific component's ratio
      exceeds its position-matched control ratio by >= 50%."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_specific_results.json'
NFRESH=32

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
    fresh=cl.fineweb_rows(NFRESH)
    nl=torch.zeros(NFRESH,T,dtype=torch.bool)
    for r in range(NFRESH):
        for q in range(T):
            nl[r,q]=chr(10) in cl.d1(int(fresh[r,q+1]))
    # position-matched control: same count per row, same position
    # distribution, random targets
    ctrl=torch.zeros_like(nl)
    g=torch.Generator().manual_seed(29)
    for r in range(NFRESH):
        k=int(nl[r].sum())
        if k==0: continue
        pos=nl[r].nonzero().squeeze(1)
        jitter=(torch.randint(-6,7,(k,),generator=g)+pos) \
            .clamp(0,T-1)
        ctrl[r,jitter]=True
    print(f'{int(nl.sum())} newline targets | '
          f'{int(ctrl.sum())} position-matched controls',
          flush=True)
    def hooks(key):
        mu=mus[key].to(DEV); mod=MODS[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        return [mod.register_forward_hook(fh)]
    def run(key):
        acc={'nl':[0.0,0],'ctrl':[0.0,0],'rest':[0.0,0]}
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]
            hs=hooks(key) if key else []
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            a=nl[i:i+4]; c=ctrl[i:i+4]; rest=~(a|c)
            for nm,mk in (('nl',a),('ctrl',c),('rest',rest)):
                acc[nm][0]+=float(ce[mk].sum())
                acc[nm][1]+=int(mk.sum())
            for h in hs: h.remove()
        return {k:acc[k][0]/max(acc[k][1],1) for k in acc}
    base=run(None)
    res={}
    for key in list(MODS):
        cur=run(key)
        dn=cur['nl']-base['nl']; dc=cur['ctrl']-base['ctrl']
        dr=cur['rest']-base['rest']
        res[key]={'d_newline':round(dn,4),'d_ctrl':round(dc,4),
                  'd_rest':round(dr,4),
                  'ratio_nl':round(dn/max(dr,1e-4),2),
                  'ratio_ctrl':round(dc/max(dr,1e-4),2)}
        json.dump(res,open(OUT,'w'),indent=1)
    ranked=sorted(res,key=lambda k:-res[k]['ratio_nl'])
    top=ranked[0]
    for k in ranked[:6]:
        print(f"{k}: nl {res[k]['d_newline']:+.4f} rest "
              f"{res[k]['d_rest']:+.4f} ratio {res[k]['ratio_nl']}"
              f" (control ratio {res[k]['ratio_ctrl']})",flush=True)
    pa=res[top]['ratio_nl']>=2.0
    pb=top not in ('m0','m1','m2','m3')
    pc=res[top]['ratio_nl']>=1.5*max(res[top]['ratio_ctrl'],1e-6)
    out={'components':res,'ranked_by_specificity':ranked[:10],
         'top':top,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc),'runtime_s':time.time()-t0}
    print(f'most newline-specific: {top} at ratio '
          f'{res[top]["ratio_nl"]}')
    for nm,v in (('a','some component is >=2x newline-specific'),
                 ('b','it is not the front block m0-m3'),
                 ('c','it beats its position-matched control')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
