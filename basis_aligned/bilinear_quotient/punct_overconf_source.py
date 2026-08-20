"""PUNCT OVERCONFIDENCE SOURCE -- 459: the punctuation effect is a
MODEL DEFICIENCY, now measured (458, n=100 pooled helped
positions): at punctuation targets the intact model's top-1 is a
NON-punctuation continuation 75% of the time, and ablation
suppresses exactly that competitor by -0.108 probability against
+1.2e-06 for a random token -- roughly five orders of magnitude
of selectivity. The model over-continues at phrase boundaries and
damage relieves it.
Which machinery creates the over-confidence? The components whose
mean-ablation helps are a3, a6, a7, a8 and m7 (455); a12 is
clean. If they share the deficiency they should share a
direction: their writes at these positions should push the
competitor token, and push it together.
Method: at helped punctuation positions, take each component's
actual write, project it through the final norm and unembedding,
and measure its logit contribution to (i) the competitor token
the model wrongly prefers and (ii) the true punctuation target.
Then measure pairwise cosines between the components' mean writes
at those positions, against a random-direction null.
REGISTERED PREDICTIONS:
  (a) THEY PUSH THE COMPETITOR: for at least 3 of the 5 helping
      components, the mean logit contribution to the competitor
      exceeds that to the true target;
  (b) SHARED DIRECTION: mean pairwise cosine between the helping
      components' mean writes >= 0.30, and at least 3x the
      random-direction null;
  (c) CONTROL: a12 shows neither (competitor push below target
      push, or cosine to the helping set below the null)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_overconf_source_results.json'
TAGS=['r.18.2.0','r.13.2.1','r.11.1.2']
HELPERS=['a3','a6','a7','a8','m7']; CTRL='a12'
MAXROWS=60

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    rows=cl.rows()
    mus=cl.comp_means()
    MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
    MODS.update({f'm{li}':m.transformer.h[li].mlp
                 for li in range(18)})
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
    ispunct=lambda t:(lambda s: bool(s) and
                      not any(c.isalnum() for c in s))(
                          cl.d1(int(t)).strip())
    mem=sorted({g for t in TAGS
                for g in cl.leaf(t)['member'].tolist()})
    d=cl.ce_sweep(hooks('a7'))-cl.base_ce()
    sites=[g for g in mem
           if ispunct(int(rows[g//256,g%256+1])) and float(d[g])<0]
    print(f'{len(sites)} helped punct sites',flush=True)
    byrow={}
    for g in sites: byrow.setdefault(g//256,[]).append(g%256)
    rowsel=sorted(byrow)[:MAXROWS]
    W=m.lm_head.weight.float()
    keys=HELPERS+[CTRL]
    acc={k:{'comp':0.0,'tgt':0.0,'n':0,'vec':None,'cnt':0}
         for k in keys}
    for i in range(0,len(rowsel),4):
        rid=torch.tensor(rowsel[i:i+4])
        bb=rows[rid,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=len(rid); outs={}
        hs=[MODS[k].register_forward_hook(
            (lambda k: lambda mo,i_,o_: outs.__setitem__(
                k,(o_[0] if isinstance(o_,tuple) else o_)
                .detach().float()))(k)) for k in keys]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        for b,r in enumerate(rid.tolist()):
            for pos in byrow[r]:
                tgt=int(rows[r,pos+1])
                t1=int(lg[b,pos].argmax())
                for k in keys:
                    w=outs[k][b,pos]
                    wn=F.rms_norm(w[None],(D,))[0]
                    a=acc[k]
                    a['comp']+=float(wn@W[t1])
                    a['tgt']+=float(wn@W[tgt])
                    a['n']+=1
                    a['vec']=w.clone() if a['vec'] is None \
                        else a['vec']+w
                    a['cnt']+=1
        print(f'rows {i} done',flush=True)
    prof={}
    for k in keys:
        a=acc[k]
        prof[k]={'logit_competitor':round(a['comp']/max(a['n'],1),3),
                 'logit_target':round(a['tgt']/max(a['n'],1),3),
                 'pushes_competitor':bool(a['comp']>a['tgt'])}
        print(f"{k}: competitor {prof[k]['logit_competitor']} vs "
              f"target {prof[k]['logit_target']}",flush=True)
    vecs={k:(acc[k]['vec']/max(acc[k]['cnt'],1)) for k in keys}
    hv=[vecs[k] for k in HELPERS]
    cs=[float(F.cosine_similarity(hv[i],hv[j],dim=0))
        for i in range(len(hv)) for j in range(i+1,len(hv))]
    mc=sum(cs)/len(cs)
    g=torch.Generator(device=DEV).manual_seed(11)
    nl=[]
    for _ in range(10):
        a1=torch.randn(D,generator=g,device=DEV)
        a2=torch.randn(D,generator=g,device=DEV)
        nl.append(abs(float(F.cosine_similarity(a1,a2,dim=0))))
    nullc=sum(nl)/len(nl)
    ctrlcos=sum(float(F.cosine_similarity(vecs[CTRL],v,dim=0))
                for v in hv)/len(hv)
    npush=sum(1 for k in HELPERS if prof[k]['pushes_competitor'])
    pa=npush>=3
    pb=(mc>=0.30 and mc>=3*max(nullc,1e-6))
    pc=(not prof[CTRL]['pushes_competitor']) or ctrlcos<nullc
    out={'n_sites':len(sites),'profiles':prof,
         'mean_pairwise_cos_helpers':round(mc,3),
         'null_cos':round(nullc,4),
         'ctrl_cos_to_helpers':round(ctrlcos,3),
         'n_helpers_pushing_competitor':npush,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'helpers pushing competitor: {npush}/5 | mean pairwise '
          f'cos {mc:.3f} vs null {nullc:.4f} | ctrl cos {ctrlcos:.3f}')
    for nm,v in (('a','>=3 helpers push the competitor'),
                 ('b','helpers share a direction'),
                 ('c','control does neither')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
