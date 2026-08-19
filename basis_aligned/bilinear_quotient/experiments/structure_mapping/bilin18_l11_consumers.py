"""Who reads the dissident? L11 is load-bearing (+0.033 nats, section 87) but its
functional code is outside the shared vocabulary (section 84). If specific readers
consume it, mean-ablating L11's MLP output should move their coefficients far more
than others'; if its function is diffuse (e.g. norm/calibration), movement should
be spread thin with no leader.

REGISTERED PREDICTIONS: (a) concentration -- the top consumer layer's median
coefficient movement is >= 3x the median across layers 12-17 (a dedicated
consumer exists); (b) adjacency -- the top consumer is L12 or L13 (coherence
length one layer, section 59, should apply to removal as to injection). Null
control: same measurement with a RANDOM 1152-dim mean-ablation of matched energy
at L11's position (movement then reflects generic perturbation, not L11 content);
bar: real/random ratio >= 2 at the top consumer."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l11_consumers_results.json')

@torch.no_grad()
def coeffs_all(rows, ablate=None):
    """Per-layer MLP-output projections on each layer's own top-8 output PCs."""
    outs={}
    hs=[]
    for li in range(12,18):
        def mk(li=li):
            def hook(mod,i_,o_):
                outs.setdefault(li,[]).append(o_.detach().reshape(-1,D).float())
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    if ablate is not None:
        Q,cbar=ablate
        def ab(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[11].mlp.register_forward_hook(ab))
    for i in range(0,len(rows),6):
        b=rows[i:i+6].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    return {li:torch.cat(v) for li,v in outs.items()}

def main():
    t0=time.time()
    rows=FW[300:336,:257]
    base=coeffs_all(rows)
    pcs={}
    for li,Y in base.items():
        _,_,Vh=torch.linalg.svd((Y-Y.mean(0)), full_matrices=False)
        pcs[li]=orth(Vh[:8].T)
    sig={li: (base[li]@pcs[li]).std(0) for li in base}
    # L11 mean stats for the ablations
    acc=[]; fwd(FW[300:336,:513].to(DEV), collect=11, acc=acc)
    Y11=acc[0]; mu=Y11.mean(0); en=float((Y11-mu).pow(2).sum(1).mean())
    I=torch.eye(D,device=DEV)
    abl=coeffs_all(rows, ablate=(I, mu@I))
    g=torch.Generator(device=DEV).manual_seed(0)
    # random control: shift L11 output by a random vector of matched energy
    rvec=torch.randn(D,device=DEV,generator=g); rvec=rvec/rvec.norm()*en**0.5
    Qr=orth(rvec[:,None])
    rnd=coeffs_all(rows, ablate=(Qr,(mu@Qr)+en**0.5))
    res={}
    for li in base:
        dv=((abl[li]-base[li])@pcs[li]).abs().mean(0)/sig[li]
        dr=((rnd[li]-base[li])@pcs[li]).abs().mean(0)/sig[li]
        res[li]={'real':float(dv.median()),'random':float(dr.median())}
        print(f'L{li}: real {res[li]["real"]:.3f}s | random-shift {res[li]["random"]:.3f}s',
              flush=True)
    meds={li:res[li]['real'] for li in res}
    top=max(meds,key=meds.get)
    overall=sorted(meds.values())[len(meds)//2]
    pa=meds[top]>=3*overall
    pb=top in (12,13)
    ratio=res[top]['real']/max(res[top]['random'],1e-9)
    pc=ratio>=2
    out={'per_layer':{str(k):v for k,v in res.items()},'top':top,
         'pred_a_concentration':bool(pa),'pred_b_adjacent':bool(pb),
         'ctrl_real_over_random':ratio,'ctrl_held':bool(pc)}
    print(f'\ntop consumer: L{top} ({meds[top]:.3f}s vs cross-layer median {overall:.3f}s)')
    print(f"(a) concentration >=3x: {'HELD' if pa else 'FAILED'}")
    print(f"(b) adjacent consumer: {'HELD' if pb else 'FAILED'}")
    print(f"random-shift control ratio {ratio:.1f}x: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
