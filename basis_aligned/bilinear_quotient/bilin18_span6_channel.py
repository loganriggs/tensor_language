"""The certifying instrument section 213 demanded. Two fixes over the
confounded consumer scan: (1) controls are MAGNITUDE-MATCHED -- random 8-dim
spans with the per-batch deleted perturbation rescaled to the span deletion's
Frobenius norm, so every arm injects equal-energy noise at L6 and only
CONTENT differs; (2) a freeze-the-middle discriminator -- MLPs 8-15 pinned to
their clean outputs during the damaged pass, so the perturbation reaches the
output end only via the residual stream and attention, killing MLP
compounding. Response measured at L16 and L17 MLP outputs.

REGISTERED PREDICTIONS: (a) honest controls, no freezing: L17 span/control
ratio >= 1.5 (the section-213 peak survives magnitude matching); (b) under
freezing, L17 ratio stays >= 1.5 (a DIRECT residual/attention channel from
span 6:1-8 to the output reader -- certifies the private-channel story at
transport level); alternative: ratio <= 1.2 under freezing = the peak was MLP
compounding, story killed. (c) sanity: frozen clean pass reproduces clean
L16/L17 outputs to < 1e-4 relative."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_span6_channel_results.json')
BATCHES=[(i,FW[i:i+6,:513].to(DEV)) for i in range(384,420,6)]

@torch.no_grad()
def run(arm, freeze, clean_mid=None, clean_tail=None, scale_ref=None):
    """arm: None (clean) or (Q, cbar). Returns (mid_caps, tail_caps, dnorms)."""
    mid={li:{} for li in range(8,16)}; tail={li:{} for li in (16,17)}
    dn={}
    hs=[]
    cur={'b':None}
    if arm is not None:
        Q,cbar=arm
        def dmg(mod,i_,o_):
            c=o_.float().reshape(-1,D)@Q
            delta=((c-cbar)@Q.T)
            if scale_ref is not None:
                s=scale_ref[cur['b']]/float(delta.norm().clamp_min(1e-9))
                delta=delta*s
            dn[cur['b']]=float(delta.norm())
            return o_-delta.to(o_.dtype).view_as(o_)
        hs.append(m.transformer.h[6].mlp.register_forward_hook(dmg))
    for li in range(8,16):
        def mk(li=li):
            def h(mod,i_,o_):
                if freeze: return clean_mid[li][cur['b']].view_as(o_)
                mid[li][cur['b']]=o_.detach().clone()
                return None
            return h
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for li in (16,17):
        def mk(li=li):
            return lambda mo_,i_,o_: tail[li].__setitem__(
                cur['b'], o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i,b in BATCHES:
        cur['b']=i
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    return mid, tail, dn

def resp(tail, clean_tail, li):
    num=sum(float((tail[li][i]-clean_tail[li][i]).norm()**2) for i,_ in BATCHES)
    den=sum(float(clean_tail[li][i].norm()**2) for i,_ in BATCHES)
    return (num/den)**0.5

@torch.no_grad()
def main():
    t0=time.time()
    Y=grab(6,0,120); mu=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-mu).float(), full_matrices=False)
    Qs=orth(Vh[:8].T); arm_span=(Qs, mu.float()@Qs)
    clean_mid, clean_tail,_=run(None, False)
    _,ct2,_=run(None, True, clean_mid=clean_mid)
    sane=max(resp(ct2,clean_tail,li) for li in (16,17))
    print(f'(c) frozen clean drift: {sane:.2e}',flush=True)
    g=torch.Generator(device=DEV).manual_seed(0)
    arms_rnd=[]
    for s_ in range(3):
        R=orth(torch.randn(D,8,device=DEV,generator=g))
        arms_rnd.append((R, mu.float()@R))
    res={}
    for tag,freeze in (('open',False),('frozen',True)):
        _,tsp,dn_span=run(arm_span, freeze,
                          clean_mid=clean_mid if freeze else None)
        rs={li:resp(tsp,clean_tail,li) for li in (16,17)}
        rr={16:[],17:[]}
        for arm in arms_rnd:
            _,tr,_=run(arm, freeze, clean_mid=clean_mid if freeze else None,
                       scale_ref=dn_span)
            for li in (16,17): rr[li].append(resp(tr,clean_tail,li))
        res[tag]={li:(rs[li], sorted(rr[li])[1],
                      rs[li]/max(sorted(rr[li])[1],1e-9)) for li in (16,17)}
        for li in (16,17):
            a,b,c=res[tag][li]
            print(f'{tag:6s} L{li}: span {a:.4f} ctrl {b:.4f} ratio {c:.2f}',
                  flush=True)
    pa=res['open'][17][2]>=1.5
    pb=res['frozen'][17][2]>=1.5
    killed=res['frozen'][17][2]<=1.2
    out={'results':{t:{str(li):v for li,v in d.items()}
                    for t,d in res.items()},
         'sanity':sane,'pred_a':bool(pa),'pred_b':bool(pb),
         'alt_killed':bool(killed)}
    print(f"\n(a) open L17 >=1.5 honest: {'HELD' if pa else 'FAILED'}")
    print(f"(b) frozen L17 >=1.5 direct channel: {'HELD' if pb else 'FAILED'}"
          f"{' (<=1.2: compounding, story killed)' if killed else ''}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
