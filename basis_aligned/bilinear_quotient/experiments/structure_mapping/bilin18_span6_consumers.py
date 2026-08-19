"""Who reads span 6:1-8? Section 212 localized the sharing anomaly to L6's
top-8 output span: written loudly (most of the layer's output variance),
deletion-improves at content level, and the only code no reader shares a
vocabulary over. Causal consumer scan: delete the span in a live forward,
measure each downstream MLP's output response ||delta out|| / ||out||, against
matched controls (3 random 8-dim spans drawn from L6's own output
distribution, components 9+ mixed).

REGISTERED PREDICTIONS: (a) if NO downstream layer responds above 1.5x the
random-span control, the span is unread cargo -- written, never consumed,
consistent with pure overshoot; (b) if any layer responds above 2x control,
it is L17 -- the solitary reader (section 210) and the private span would be
one private channel; (c) control sanity: the immediate residual carrier L7
responds to any 8-dim deletion (ratio near 1 there is expected, it is not a
consumer signature)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_span6_consumers_results.json')

@torch.no_grad()
def responses(Q, cbar):
    caps={li:[] for li in range(7,18)}
    def dmg(mod,i_,o_):
        c=o_.float().reshape(-1,D)@Q
        return o_-((c-cbar)@Q.T).to(o_.dtype).view_as(o_)
    hs=[m.transformer.h[6].mlp.register_forward_hook(dmg)]
    for li in range(7,18):
        def mk(li=li):
            return lambda mo_,i_,o_: caps[li].append(
                o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(384,420,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    return {li:torch.cat(v) for li,v in caps.items()}

@torch.no_grad()
def main():
    t0=time.time()
    Y=grab(6,0,120); mu=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-mu).float(), full_matrices=False)
    Qs=orth(Vh[:8].T); cbar_s=mu.float()@Qs
    clean=responses(torch.zeros(D,8,device=DEV), torch.zeros(8,device=DEV))
    span=responses(Qs, cbar_s)
    g=torch.Generator(device=DEV).manual_seed(0)
    rnds=[]
    for s_ in range(3):
        mix=orth((Vh[8:].T@torch.randn(Vh.shape[0]-8,8,device=DEV,generator=g)))
        rnds.append(responses(mix, mu.float()@mix))
    ratios={}
    for li in range(7,18):
        d_span=float((span[li]-clean[li]).norm())/float(clean[li].norm())
        d_rnd=sorted(float((r[li]-clean[li]).norm())/float(clean[li].norm())
                     for r in rnds)[1]
        ratios[li]=(d_span,d_rnd,d_span/max(d_rnd,1e-9))
        print(f'L{li:2d}: span {d_span:.4f} rnd {d_rnd:.4f} '
              f'ratio {d_span/max(d_rnd,1e-9):.2f}',flush=True)
    big={li:r[2] for li,r in ratios.items() if r[2]>=2.0 and li>7}
    pa=all(r[2]<1.5 for li,r in ratios.items() if li>7)
    pb=(list(big)==[17]) if big else None
    out={'ratios':{str(k):v for k,v in ratios.items()},
         'pred_a_unread':bool(pa),
         'pred_b_l17':None if pb is None else bool(pb)}
    if pa: print('\n(a) HELD: span 6:1-8 is unread cargo (all ratios <1.5)')
    elif big: print(f"\n(a) failed; consumers >=2x: {sorted(big)} -- "
                    f"(b) L17-only: {'HELD' if pb else 'FAILED'}")
    else: print('\n(a) failed softly: responses in [1.5,2), no clear consumer')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
