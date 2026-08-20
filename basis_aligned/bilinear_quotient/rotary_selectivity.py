"""ROTARY SELECTIVITY -- is behaviour-specific attention positional
across the model, or was that one head?
Two circuits, measured by different methods, gave the same answer
this week. Head 13.8's discrimination between the opener its
bracket closes and a distractor survives removing the token
embedding from the key side (6.81 against 6.48 untouched) and
VANISHES when rotary position encoding is removed (1.08) -- so its
selection is positional and content contributes nothing (529). And
the digit subspace's channel content is supplied by digit source
positions at 14x the base rate, yet what those positions
contribute is composed identically to what any other position
contributes (528, 530) -- so its selectivity is in which positions
are attended to, not in what they carry.
If that is a property of the model rather than of two circuits, it
should hold for the other behaviour-leading heads the atlas found.
Rotary is removed for ONE head at a time -- its queries and keys
use unrotated vectors while every other head, and the rest of the
network, is untouched -- and each head is re-priced on its own
behaviour.
  head   behaviour            damage at target (from 497-524)
  13.8   closing brackets     +0.825
  12.6   line breaks          +0.068
  10.7   opening quotes       +0.056
   8.3   digits (subspace)    measured here as head damage
Position-matched controls throughout, and the concentration is
reported as the absolute PAIR (target, elsewhere) rather than a
quotient (the rule from 520).
REGISTERED PREDICTIONS, including a differentiated bet that can
fail in a specific direction:
  (0) THE ARM FIRES: disabling rotary for a head changes that
      head's target-position damage by a relative amount above
      1e-6, checked per head; an exactly-zero arm is void (446);
  (a) GENERAL: at least three of the four heads lose >= 50% of
      their behaviour-specific damage when their own rotary is
      removed;
  (b) THE BRACKET HEAD LOSES ALMOST EVERYTHING: 13.8 retains
      <= 20%. Its pointer is a distance and nothing else;
  (c) THE NEWLINE HEAD DOES NOT: 12.6 retains >= 50%. It responds
      to sentence-final punctuation and to whether the document is
      line-broken (501), neither of which is a distance, so if the
      positional story were universal it would fail HERE, and
      that is the point of including it. If 12.6 also collapses,
      (a) becomes a much stronger claim about the model and (c)
      is reported as a wrong prediction.
  CONTROL: disabling rotary for an UNRELATED head in the same
      layer changes the target behaviour's damage by < 20% of what
      disabling the leading head's rotary does.
  NULL: disabling a head's rotary must change whole-text
      cross-entropy measurably (> 1e-4 nats), or the intervention
      is not doing anything and its result is void."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; NH=9; NLID=198
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'rotary_selectivity_results.json'
NFRESH=128
# (layer, head, behaviour, control head in the same layer)
TARGETS=[(13,8,'close_bracket',3),(12,6,'newline',2),
         (10,7,'open_quote',1),(8,3,'digit',5)]
OPENS={'(':')','[':']','{':'}'}; CLOSES={v:k for k,v in OPENS.items()}

def classes(fresh):
    nxt=fresh[:,1:257]; R,Tn=nxt.shape
    M={k:torch.zeros(R,Tn,dtype=torch.bool)
       for k in ('close_bracket','newline','open_quote','digit')}
    for r in range(R):
        for q in range(Tn):
            s=cl.d1(int(nxt[r,q])); t=s.strip()
            if '\n' in s: M['newline'][r,q]=True
            if not t: continue
            if t[0].isdigit(): M['digit'][r,q]=True
            if t in CLOSES: M['close_bracket'][r,q]=True
            if t in ('"',"'",'``',"'") and (s.startswith(' ') or s==t):
                M['open_quote'][r,q]=True
    return M

@torch.no_grad()
def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    fresh=cl.fineweb_rows(NFRESH)
    M=classes(fresh)
    g=torch.Generator().manual_seed(29)
    CT={}
    for k,mask in M.items():
        c=torch.zeros_like(mask)
        for r in range(NFRESH):
            n=int(mask[r].sum())
            if n==0: continue
            pos=mask[r].nonzero().squeeze(1)
            c[r,(torch.randint(-6,7,(n,),generator=g)+pos)
              .clamp(0,T-1)]=True
        CT[k]=c
        print(f'{k}: {int(mask.sum())} targets',flush=True)

    def run(mode):
        """mode: None | ('mean',li,h) | ('norot',li,h)"""
        ce=torch.zeros(NFRESH,T)
        for i in range(0,NFRESH,4):
            bb=fresh[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=bb.shape[0]; hs=[]
            if mode is not None:
                kind,li,h=mode
                at=m.transformer.h[li].attn
                def fh(mo,args,o_,at=at,li=li,h=h,kind=kind):
                    y,v1r=o_; X=args[0]
                    v1b=args[1] if args[1] is not None else v1r
                    z,vm=cl.head_parts(li,X,v1b); z=z.clone()
                    if kind=='mean':
                        z[:,h]=z[:,h].mean(dim=(0,1),keepdim=True)
                    else:
                        cq,sq=at.rotary(at.c_q(X).view(B,T,NH,128))
                        def raw(W):
                            return F.rms_norm(
                                W(X).view(B,T,NH,128),
                                (128,))[:,:,h].float()
                        s1=torch.einsum('bqd,bkd->bqk',raw(at.c_q),
                                        raw(at.c_k))/128
                        s2=torch.einsum('bqd,bkd->bqk',raw(at.c_q2),
                                        raw(at.c_k2))/128
                        sc=(s1*s2)*torch.tril(
                            torch.ones(T,T,device=DEV))
                        z[:,h]=torch.einsum('bqk,bkd->bqd',sc,
                                            vm[:,:,h].float())
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ce[i:i+B]=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                      reduction='none').view(B,T).cpu()
            for h_ in hs: h_.remove()
        return ce

    base=run(None)
    res={}; dead=[]
    for (li,h,beh,ctrl) in TARGETS:
        mask=M[beh]; cmask=CT[beh]
        d_mean=run(('mean',li,h))-base
        d_nor=run(('norot',li,h))-base
        d_ctrl=run(('norot',li,ctrl))-base
        conc=lambda d:(float(d[mask].mean()),float(d[~mask].mean()))
        m_t,m_o=conc(d_mean); n_t,n_o=conc(d_nor)
        c_t,c_o=conc(d_ctrl)
        # behaviour-specific damage = target minus elsewhere
        spec_mean=m_t-m_o; spec_nor=n_t-n_o; spec_ctrl=c_t-c_o
        retain=spec_nor/spec_mean if abs(spec_mean)>1e-9 else float('nan')
        fired=abs(float((d_nor-d_mean)[mask].mean()))>1e-6 \
              or abs(spec_nor-spec_mean)>1e-6
        if not fired: dead.append(f'{li}.{h}')
        res[f'{li}.{h}']={
            'behaviour':beh,
            'delete_target':round(m_t,5),'delete_else':round(m_o,5),
            'delete_specific':round(spec_mean,5),
            'norot_target':round(n_t,5),'norot_else':round(n_o,5),
            'norot_specific':round(spec_nor,5),
            'retained':round(retain,3),
            'ctrl_head':f'{li}.{ctrl}',
            'ctrl_specific':round(spec_ctrl,5),
            'ctrl_frac':round(abs(spec_ctrl)/max(abs(spec_nor),1e-9),3),
            'ce_shift':round(float(d_nor.mean()),6)}
        r=res[f'{li}.{h}']
        print(f"{li}.{h} ({beh}): delete-specific "
              f"{spec_mean:+.5f} | rotary-off specific "
              f"{spec_nor:+.5f} -> retains {retain:+.3f} | control "
              f"head {li}.{ctrl} {spec_ctrl:+.5f}",flush=True)
        json.dump(res,open(OUT,'w'),indent=1)
    if dead:
        print(f'*** ARMS THAT NEVER FIRED: {dead} -- void ***')
    lost=[k for k,v in res.items() if v['retained']<=0.50]
    pa=len(lost)>=3
    br=res.get('13.8',{}).get('retained',1.0)
    nl=res.get('12.6',{}).get('retained',0.0)
    vb,_=cl.score_bar('b',0.20-br,1e-9)
    vc,_=cl.score_bar('c',nl,0.50)
    ctrl_ok=all(v['ctrl_frac']<0.20 for v in res.values())
    nul=all(abs(v['ce_shift'])>1e-4 for v in res.values())
    print(f"\n(a) >=3 of 4 heads retain <=50%: {sorted(lost)} -> "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) bracket head 13.8 retains {br:+.3f} <= 0.20")
    print(f"(c) newline head 12.6 retains {nl:+.3f} >= 0.50")
    print(f"CONTROL (unrelated same-layer head < 20% of the "
          f"effect): {'ok' if ctrl_ok else 'VIOLATED'}")
    print(f"NULL (whole-text CE moves > 1e-4 for every head): "
          f"{'ok' if nul else 'VIOLATED'}")
    out={'heads':res,'lost':sorted(lost),'arms_never_fired':dead,
         'pred_a':bool(pa),'pred_b':vb=='HELD','pred_c':vc=='HELD',
         'control_ok':bool(ctrl_ok),'null_ok':bool(nul),
         'runtime_s':time.time()-t0}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
