"""PUNCT OVERCONF RECENTERED -- 459: the source scan (458) found
all five helping components push the wrong continuation over the
true punctuation target (a3 7.22 vs 4.67, a8 14.30 vs 7.01, m7
15.52 vs 11.94, a7 14.33 vs 13.09, a6 11.82 vs 11.43) while the
clean control a12 does not (9.55 vs 10.00). But its GEOMETRY leg
is contaminated: the helpers' mean pairwise cosine was 0.712
against a random null of 0.032 -- and the control sits at 0.650
with them. That is the stream-centre effect (449/450): every
component write at these depths is partly aligned with the layer-5
bias axis, so raw cosines between writes are inflated and cannot
discriminate.
Redo the geometry with the bias axis projected out of every write,
and re-check the logit asymmetry under the same projection.
REGISTERED PREDICTIONS:
  (a) GEOMETRY BECOMES DISCRIMINATIVE: after recentering, the
      helpers' mean pairwise cosine is >= 0.30 AND exceeds the
      control's cosine to them by >= 0.15;
  (b) if (a) fails, geometry is uninformative for this question
      and the logit asymmetry stands alone -- recorded either way;
  (c) the logit asymmetry survives recentering: >= 3 helpers
      still push the competitor above the target, and a12 does
      not."""

import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'punct_overconf_recentered_results.json'
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
    # the layer-5 stream-centre axis (449), projected out below
    import subprocess
    are=__import__('sys').modules[
        type(m.transformer.h[0].attn).__module__].apply_rotary_emb
    capb={}
    hb=m.transformer.h[5].attn.register_forward_pre_hook(
        lambda mo_,a_: capb.__setitem__('X',a_[0]))
    bb0=rows[:4,:257].to(DEV); idx0=bb0[:,:-1].contiguous()
    xb=F.rms_norm(m.transformer.wte(idx0),(D,)); x0b=xb; v1b=None
    for blk in m.transformer.h: xb,v1b=blk(xb,v1b,x0b)
    hb.remove()
    atb=m.transformer.h[5].attn; Xb=capb['X']
    cosb,sinb=atb.rotary(atb.c_q(Xb).view(4,T,9,128))
    def rotb(w):
        return are(F.rms_norm(w(Xb).view(4,T,9,128),
                   (128,))[:,:,7][:,:,None],cosb,sinb)[:,:,0]
    qb,kb=rotb(atb.c_q),rotb(atb.c_k)
    q2b,k2b=rotb(atb.c_q2),rotb(atb.c_k2)
    patb=((torch.einsum('bqd,bkd->bqk',qb.float(),kb.float())/128)
          *(torch.einsum('bqd,bkd->bqk',q2b.float(),k2b.float())/128)) \
        *torch.tril(torch.ones(T,T,device=DEV))
    vb=atb.c_v(Xb).view(4,T,9,128)[:,:,7].float()*(1-atb.lamb)
    Wpb=atb.c_proj.weight.float()[:,7*128:8*128]
    bias=(torch.einsum('bqk,bkd->bqd',patb,vb)@Wpb.T).mean(dim=(0,1))
    U=bias/bias.norm().clamp_min(1e-6)
    print(f'stream-centre axis captured (norm {float(bias.norm()):.1f})',
          flush=True)
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
                    w=w-(w@U)*U          # project out the centre
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
    pa=(mc>=0.30 and (mc-ctrlcos)>=0.15)
    pb=True
    pc=(npush>=3 and not prof[CTRL]['pushes_competitor'])
    out={'n_sites':len(sites),'profiles':prof,
         'mean_pairwise_cos_helpers':round(mc,3),
         'null_cos':round(nullc,4),
         'ctrl_cos_to_helpers':round(ctrlcos,3),
         'n_helpers_pushing_competitor':npush,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'helpers pushing competitor: {npush}/5 | mean pairwise '
          f'cos {mc:.3f} vs null {nullc:.4f} | ctrl cos {ctrlcos:.3f}')
    for nm,v in (('a','recentered geometry discriminates'),
                 ('b','verdict recorded either way'),
                 ('c','logit asymmetry survives recentering')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
