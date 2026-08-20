"""NEWLINE CIRCUIT -- 494: first behaviour-defined circuit attempt
using the atlas. The program's two code-level circuits both came
from behaviourally-defined targets (induction) or cost anomalies
(the sink); the damage-cluster census produced none. So define a
task with crisp semantics and run the playbook.
Target: NEWLINE PREDICTION -- positions whose next token is a line
break. Three independent measurements point here. Newline is the
class most damaged by bundle ablation (464, +0.027 dissociation
where punctuation and digits are SPARED); it is the most causally
potent class for head 12.6 (492: 0.0126 damage per thousand
blocked positions, fifteen times punctuation's); and line
structure is what a long-range structure reader would track.
Step 1 of the playbook: find the components. Mean-ablate each of
the 36 components (18 attention layers, 18 MLPs) and measure the
CE cost specifically at newline-target positions, against
non-newline positions in the same passes.
REGISTERED PREDICTIONS:
  (a) CONCENTRATED: the top five components carry >= 50% of the
      summed newline-specific cost;
  (b) NOT THE SINK: the single largest newline-specific
      contributor is not attention layer 5 (the sink's layer);
  (c) TASK-SPECIFIC: the top component's newline cost exceeds its
      non-newline cost by >= 2x -- otherwise it is just a
      generally important component and the target is not
      isolating anything."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'newline_circuit_results.json'
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
    print(f'{int(nl.sum())} newline targets of {nl.numel()}',
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
        tn=to=0.0; nn_=no_=0
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
            mk=nl[i:i+4]
            tn+=float(ce[mk].sum()); nn_+=int(mk.sum())
            to+=float(ce[~mk].sum()); no_+=int((~mk).sum())
            for h in hs: h.remove()
        return tn/max(nn_,1),to/max(no_,1)
    bn,bo=run(None)
    res={}
    for key in list(MODS):
        pn,po=run(key)
        res[key]={'newline':round(pn-bn,4),
                  'other':round(po-bo,4),
                  'ratio':round((pn-bn)/max(po-bo,1e-4),2)}
        if abs(res[key]['newline'])>0.02:
            print(f"{key}: newline {res[key]['newline']:+.4f} "
                  f"other {res[key]['other']:+.4f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    pos={k:v['newline'] for k,v in res.items()
         if v['newline']>0}
    tot=sum(pos.values())
    top5=sorted(pos,key=pos.get,reverse=True)[:5]
    share=sum(pos[k] for k in top5)/max(tot,1e-6)
    top=top5[0] if top5 else None
    pa=share>=0.50
    pb=(top!='a5')
    pc=(top is not None and
        res[top]['newline']>=2*max(res[top]['other'],1e-4))
    out={'baseline_newline_ce':round(bn,4),'components':res,
         'top5':top5,'top5_share':round(share,3),
         'top_component':top,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'top five {top5} carry {share:.1%} | leader {top}')
    for nm,v in (('a','top five carry >=50%'),
                 ('b','the leader is not the sink layer'),
                 ('c','the leader is newline-specific (>=2x)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
