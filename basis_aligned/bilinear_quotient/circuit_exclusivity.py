"""Stage G rung 2 for the four certified function circuits: SITE-SPECIFICITY
by positional ablation. For each circuit (owners = its top-2 components),
three arms on window C (rows 120-300), per-token CE: owners ablated
EVERYWHERE, ablated ON-SLICE only, ablated OFF-SLICE only. This tests the
causal statement "these components' function on these sites IS this
function": on-slice-only ablation should reproduce the slice damage of
full ablation; off-slice-only ablation should barely touch the slice; and
the off-slice arm's total cost measures how much ELSE the owners do
(their 'moonlighting' share, reported).

Circuits (from supervised_circuits, section 238):
  digit:   attn8 + mlp15    bclose: attn13 + attn4
  subword: mlp16 + mlp15    name:   attn1  + attn0

REGISTERED PREDICTIONS: (a) on-slice-only recovers >= 60% of full-ablation
slice damage for >= 3/4 circuits (damage is site-local); (b) off-slice-only
costs the slice <= 40% of full-ablation slice damage for >= 3/4 (little
spillover); (c) moonlighting: for subword (owners mlp16/15, big generic
components) off-slice total cost exceeds on-slice total cost (the owners
do much else); for bclose (attn13+attn4) registered the reverse as a
long-shot -- bracket-closing as a large share of those heads' job."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean
import tiktoken
enc=tiktoken.get_encoding('gpt2')
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_exclusivity_results.json'
R0,R1=120,300
CIRCUITS={'digit':('attn8','mlp15'),'bclose':('attn13','attn4'),
          'subword':('mlp16','mlp15'),'name':('attn1','attn0')}

def slicemask(name):
    M=torch.zeros(R1-R0,256,dtype=torch.bool)
    for r in range(R0,R1):
        toks=FW[r,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; tg=enc.decode([t]); s=tg.strip()
            if name=='digit': v=s.isdigit() and not tg.startswith(' ')
            elif name=='bclose':
                v=s in (')',']') and any(b in enc.decode(
                    toks[max(0,pos-60):pos+1]) for b in ('(','['))
            elif name=='subword': v=(not tg.startswith(' ')) and s.isalpha()
            else:
                pv=enc.decode([toks[pos]])
                v=(tg.startswith(' ') and s[:1].isupper() and
                   (pv.strip()[:1].isupper() if pv.strip() else False))
            M[r-R0,pos]=v
    return M

@torch.no_grad()
def prep_component(cname):
    typ=cname[:4]; li=int(cname[4:])
    if typ=='mlp ' or typ=='mlp'+cname[4]: pass
    if cname.startswith('mlp'):
        li=int(cname[3:])
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        return ('mlp',li,orth(Vh[:8].T),Yb.float())
    li=int(cname[4:])
    return ('attn',li,attn_mean(li),None)

@torch.no_grad()
def pertok(comps=None, mask=None):
    """comps: list of prepped components to ablate; mask (rows,256) bool on
    DEV or None=everywhere."""
    hs=[]
    cur={'b0':0}
    if comps:
        for spec in comps:
            if spec[0]=='mlp':
                _,li,Q,mu=spec
                def mk(li=li,Q=Q,mu=mu):
                    def hook(mod,i_,o_):
                        B,T,_=o_.shape
                        c=o_.float().reshape(-1,D)@Q
                        delta=((c-(mu@Q))@Q.T).view(B,T,D)
                        if mask is not None:
                            mm=mask[cur['b0']:cur['b0']+B,:T].to(DEV)
                            delta=delta*mm[:,:,None]
                        return o_-delta.to(o_.dtype)
                    return hook
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(mk()))
            else:
                _,li,mu,_=spec
                def mka(li=li,mu=mu):
                    def hook(mod,i_,o_):
                        out=o_[0] if isinstance(o_,tuple) else o_
                        B,T,_=out.shape
                        rep=mu[None,None,:].to(out.dtype).expand_as(out)
                        if mask is not None:
                            mm=mask[cur['b0']:cur['b0']+B,:T].to(DEV)
                            new=torch.where(mm[:,:,None],rep,out)
                        else: new=rep
                        if isinstance(o_,tuple): return (new,)+o_[1:]
                        return new
                    return hook
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(mka()))
    ces=[]
    for i in range(R0,R1,4):
        cur['b0']=i-R0
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    for h in hs: h.remove()
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    base=pertok()
    # sanity: attention module hook path must actually modify the forward
    test=prep_component('attn1')
    d_test=float((pertok(comps=[test])-base).mean())
    print(f'sanity attn-hook full ablation attn1 net {d_test:+.4f} '
          f'(atlas ~+0.05 expected scale)',flush=True)
    assert abs(d_test)>1e-3, 'attention hook did not modify forward!'
    out={}
    ok_a=0; ok_b=0
    for name,(c1,c2) in CIRCUITS.items():
        comps=[prep_component(c1),prep_component(c2)]
        Mmask=slicemask(name).to(DEV)
        flat=Mmask.reshape(-1)
        full=pertok(comps=comps)-base
        on=pertok(comps=comps,mask=Mmask)-base
        off=pertok(comps=comps,mask=~Mmask)-base
        sl_full=float(full[flat].mean()); sl_on=float(on[flat].mean())
        sl_off=float(off[flat].mean())
        tot_on=float(on.mean()); tot_off=float(off.mean())
        ra=sl_on/max(sl_full,1e-6); rb=sl_off/max(sl_full,1e-6)
        ok_a+= (ra>=0.60); ok_b+=(rb<=0.40)
        out[name]={'slice_damage_full':round(sl_full,4),
                   'slice_damage_onslice':round(sl_on,4),
                   'slice_damage_offslice':round(sl_off,4),
                   'total_onslice':round(tot_on,4),
                   'total_offslice':round(tot_off,4),
                   'recover_frac':round(ra,2),'spill_frac':round(rb,2)}
        print(f'{name:8s}: slice dmg full {sl_full:+.3f} | on-only '
              f'{sl_on:+.3f} ({ra:.0%}) | off-only {sl_off:+.3f} '
              f'({rb:.0%}) | moonlight tot off {tot_off:+.4f} vs on '
              f'{tot_on:+.4f}',flush=True)
    pa=ok_a>=3; pb=ok_b>=3
    pc_sub=out['subword']['total_offslice']>out['subword']['total_onslice']
    pc_bc=out['bclose']['total_offslice']<out['bclose']['total_onslice']
    res={'circuits':out,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c_subword_moonlights':bool(pc_sub),
         'longshot_bclose_dedicated':bool(pc_bc)}
    print(f"\n(a) site-local >=3/4: {'HELD' if pa else 'FAILED'} ({ok_a})")
    print(f"(b) low spillover >=3/4: {'HELD' if pb else 'FAILED'} ({ok_b})")
    print(f"(c) subword owners moonlight: {'HELD' if pc_sub else 'FAILED'}")
    print(f"    long-shot bclose dedicated: {'YES' if pc_bc else 'no'}")
    res['runtime_s']=time.time()-t0
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
