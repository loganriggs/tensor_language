"""Control for bilin18_whitened.py: does the rank result survive using output
directions the model actually uses?

bilin18_whitened.py scored interaction forms for RANDOM output directions d. A random d
mixes all 4608 neurons in the layer, so a high-rank answer is close to guaranteed and
the comparison across depth could be an artefact. The directions that matter are the
ones the layer's output actually occupies. This reruns the same measurement with d
taken from the top principal components of the MLP's own output, and (as a second,
tougher basis) the read-out directions of the block that follows.

The result under test is bilin18_whitened.py's headline: in the Lambda-weighted metric
layer 17 needs rank 4 for 90% of the function while layers 9 and 13 still sit at
15-18% FVU at rank 128. If that ordering is an artefact of random d, it disappears here.
"""
import json, sys, time, torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
from tier2_model import load_elriggs
from bilin18_identifiable import mlp_inputs, form_for_direction
from bilin18_whitened import sqrtm_psd, fvu, truncate

DEV='cuda'; N_FIT=6000; N_TEST=6000; LAYERS=(0,1,5,9,13,17); KS=(1,2,4,8,16,32,64,128); N_DIRS=8

@torch.no_grad()
def mlp_out_pcs(model, tokens, layers, k):
    """Top-k principal directions of each MLP's output."""
    store={}; hooks=[]
    def mk(li):
        def hook(mod,inp,outp): store.setdefault(li,[]).append(outp.detach().reshape(-1,outp.shape[-1]).float())
        return hook
    for li in layers: hooks.append(model.transformer.h[li].mlp.register_forward_hook(mk(li)))
    for i in range(0,64,4):
        b=tokens[i:i+4].to(DEV); model(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hooks: h.remove()
    out={}
    for li in layers:
        Y=torch.cat(store[li],0); Y=Y-Y.mean(0,keepdim=True)
        _,_,V=torch.linalg.svd(Y, full_matrices=False)
        out[li]=V[:k].double()
    return out

def main():
    t0=time.time()
    model,cfg=load_elriggs('bilin18', device=DEV)
    tokens=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens.pt')
    X=mlp_inputs(model, tokens, LAYERS, N_FIT+N_TEST)
    PC=mlp_out_pcs(model, tokens, LAYERS, N_DIRS)
    out={'ks':list(KS),'layers':{},'basis':'top principal directions of the MLP output'}
    print('== same measurement, but for output directions the MLP actually uses ==')
    print('   d = top-8 PCs of the MLP output (bilin18_whitened.py used random d)\n')
    print(f"  {'layer':>5}  {'rank':>9}  "+''.join(f"{'k='+str(k):>9}" for k in KS))
    for li in LAYERS:
        Xa=X[li].to(DEV); Xf,Xt=Xa[:N_FIT],Xa[N_FIT:N_FIT+N_TEST]
        S=Xf.T@Xf/Xf.shape[0]; Sh,Sih=sqrtm_psd(S)
        mlp=model.transformer.h[li].mlp
        raw={k:[] for k in KS}; wht={k:[] for k in KS}
        for j in range(N_DIRS):
            d=PC[li][j].to(DEV).float(); M=form_for_direction(mlp, d/d.norm()); Mw=Sh@M@Sh
            for k in KS:
                raw[k].append(fvu(M,truncate(M,k),Xt))
                wht[k].append(fvu(M,Sih@truncate(Mw,k)@Sih,Xt))
        r=[sum(raw[k])/N_DIRS for k in KS]; w=[sum(wht[k])/N_DIRS for k in KS]
        def k90(c):
            for k,v in zip(KS,c):
                if v<=0.10: return k
            return None
        out['layers'][li]={'raw_fvu':r,'whitened_fvu':w,'k_for_90pct_raw':k90(r),
                           'k_for_90pct_whitened':k90(w),'gap_at_k16':r[KS.index(16)]/max(w[KS.index(16)],1e-30)}
        print(f"  {li:>5}  {'raw':>9}  "+''.join(f"{v:>9.3f}" for v in r))
        print(f"  {'':>5}  {'whitened':>9}  "+''.join(f"{v:>9.3f}" for v in w))
        print(f"  {'':>5}  {'->':>9}  rank for 90%: raw {k90(r)}, whitened {k90(w)}   "
              f"(gap at k=16: {out['layers'][li]['gap_at_k16']:.1f}x)\n", flush=True)
    out['runtime_s']=time.time()-t0
    p='/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_whitened_dirs_results.json'
    json.dump(out, open(p,'w'), indent=1); print(f'wrote {p} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
