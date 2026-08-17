"""What does the dissident DO? L11 is load-bearing (+0.033 nats, section 87), speaks
a foreign code (section 84), and is consumed diffusely (section 88). Two candidate
function classes distinguishable by the SHAPE of its ablation damage:
  calibration-like: damage spread evenly over tokens; logit entropy shifts.
  content-like: damage concentrated on a small token subset.

Mean-ablate L11's MLP output, measure per-token CE deltas on held-out rows.
REGISTERED PREDICTIONS: (a) diffuse damage -- the top decile of tokens by CE-delta
carries < 35% of total damage (uniform would be ~10%, content-features in past runs
carried > 50%); (b) entropy signature -- mean next-token entropy changes by >= 2x
the change under an energy-matched random shift at the same site. Either failing
still classifies: (a) failing = content-like after all; both failing = neither
story, report as open."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l11_function_results.json')

@torch.no_grad()
def run(rows, ablate=None):
    hs=[]
    if ablate is not None:
        Q,shift=ablate
        def ab(mod,i_,o_):
            if Q is None:
                return o_+shift.to(o_.dtype)
            c=o_.float()@Q
            return (o_-((c-shift)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[11].mlp.register_forward_hook(ab))
    ces=[]; ents=[]
    for i in range(0,len(rows),4):
        b=rows[i:i+4].to(DEV)
        idx,tg=b[:,:-1].contiguous(), b[:,1:].contiguous()
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h:
            x,v1=blk(x,v1,x0)
        x=F.rms_norm(x,(D,))
        logits=m.lm_head(x); logits=30*torch.tanh(logits/30); logits=logits.float()
        ce=F.cross_entropy(logits.view(-1,logits.size(-1)),tg.reshape(-1),
                           reduction='none')
        p=F.softmax(logits,dim=-1)
        ent=-(p*p.clamp_min(1e-12).log()).sum(-1)
        ces.append(ce); ents.append(ent.reshape(-1))
    for h in hs: h.remove()
    return torch.cat(ces), torch.cat(ents)

def main():
    t0=time.time()
    rows=FW[300:380,:257]
    acc=[]; fwd(FW[300:336,:513].to(DEV), collect=11, acc=acc)
    Y=acc[0]; mu=Y.mean(0); en=float((Y-mu).pow(2).sum(1).mean())
    ce0,en0=run(rows)
    I=torch.eye(D,device=DEV)
    ce1,en1=run(rows, ablate=(I, mu@I))
    g=torch.Generator(device=DEV).manual_seed(0)
    rvec=torch.randn(D,device=DEV,generator=g); rvec=rvec/rvec.norm()*en**0.5
    ce2,en2=run(rows, ablate=(None, rvec))
    d=(ce1-ce0)
    tot=float(d.clamp_min(0).sum())
    k=max(1,int(0.10*d.numel()))
    top=float(d.clamp_min(0).topk(k).values.sum())
    frac=top/max(tot,1e-9)
    dent=abs(float((en1-en0).mean())); dent_r=abs(float((en2-en0).mean()))
    ratio=dent/max(dent_r,1e-9)
    pa=frac<0.35; pb=ratio>=2
    out={'mean_damage':float(d.mean()),'top_decile_fraction':frac,
         'entropy_shift':dent,'entropy_shift_random':dent_r,
         'entropy_ratio':ratio,'pred_a_diffuse':bool(pa),
         'pred_b_entropy':bool(pb)}
    print(f'mean CE damage {float(d.mean()):+.4f} | top-decile share {frac:.2f}')
    print(f'entropy shift {dent:.4f} vs random-shift {dent_r:.4f} ({ratio:.1f}x)')
    print(f"(a) diffuse damage (<0.35): {'HELD' if pa else 'FAILED'}")
    print(f"(b) entropy signature (>=2x): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
