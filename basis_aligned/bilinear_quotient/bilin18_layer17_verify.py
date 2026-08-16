"""Do layer 17's features actually detect the tokens the unembedding says they do?

bilin18_layer17_readout.py named each feature direction w by the tokens whose
unembedding rows it points along. That is the standard shortcut and it is weak
evidence: it assumes the residual direction the MLP READS is aligned with the one the
unembedding WRITES. This tests it without that assumption -- run the corpus, record
(w . x)^2 at every position, and ask which tokens actually sit where the feature fires.

FIRST ATTEMPT FAILED FOR LACK OF POWER, and is recorded here rather than quietly
replaced. It used the 32x513 eval set, where only 48 tokens clear 30 occurrences; a
20-name list against a 10-token excitation list then has CHANCE overlap 4.2/10, and the
measured 2.7/10 sat below chance. That run says nothing either way. Two fixes:

  data       512x513 = 262,656 positions, 1,029 tokens clearing 30 occurrences (16x more)
  statistic  Spearman correlation across ALL qualifying tokens between the
             unembedding-derived score |wte . w| and the measured mean excitation,
             against a permutation null -- instead of an overlap between two lists of
             mismatched length against a pool of 48

A feature could key on the CURRENT token (whose residual stream this is) or on what is
about to be predicted, so both are scored. If the correlation is not clearly above the
permutation null, the readout's token names are decoration and get struck.
"""
import json, sys, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl'); sys.path.insert(0,'/workspace/tensor_language')
import tiktoken
from tier2_model import load_elriggs
from bilin18_identifiable import form_for_direction
from bilin18_whitened import sqrtm_psd
from bilin18_layer17 import out_pcs, LAYER, DEV

enc=tiktoken.get_encoding('gpt2'); MIN_COUNT=30; N_PERM=200

@torch.no_grad()
def inputs_with_tokens(model, tokens, li, nb):
    store=[]; cur=[]; nxt=[]
    h=model.transformer.h[li].mlp.register_forward_hook(
        lambda m,i,o: store.append(i[0].detach().reshape(-1,i[0].shape[-1]).float()))
    for i in range(0,nb,4):
        b=tokens[i:i+4].to(DEV); model(b[:,:-1].contiguous(), b[:,1:].contiguous())
        cur.append(b[:,:-1].reshape(-1)); nxt.append(b[:,1:].reshape(-1))
    h.remove()
    return torch.cat(store,0), torch.cat(cur,0), torch.cat(nxt,0)

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

def main():
    model,cfg=load_elriggs('bilin18', device=DEV)
    tokens=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens_large.pt')
    X,cur,nxt=inputs_with_tokens(model,tokens,LAYER,len(tokens))
    uniq,counts=cur.unique(return_counts=True); keep=uniq[counts>=MIN_COUNT]
    print(f'{X.shape[0]:,} positions | {keep.numel()} tokens clearing {MIN_COUNT} '
          f'occurrences (the failed first attempt had 48)\n')
    P,mu,evr=out_pcs(model,tokens[:32],LAYER,8)
    Xs=X[:20000].double(); S=Xs.T@Xs/Xs.shape[0]; Sh,Sih=sqrtm_psd(S)
    wte=model.transformer.wte.weight.detach().float()
    mlp=model.transformer.h[LAYER].mlp
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'n_positions':int(X.shape[0]),'n_tokens':int(keep.numel()),'features':[]}
    print(f"  {'feature':>12}  {'rho(current tok)':>17}  {'rho(next tok)':>14}  "
          f"{'perm null 95th':>15}  verdict")
    for p in range(3):
        d=P[p].float(); M=form_for_direction(mlp,d/d.norm()); Mw=Sh@M@Sh
        ev,U=torch.linalg.eigh(Mw); idx=ev.abs().argsort(descending=True)[:2]
        W=(Sih@U[:,idx]).float(); W=W/W.norm(dim=0,keepdim=True)
        for j in range(2):
            w=W[:,j]; a=(X@w)**2
            name=(wte[keep]@w).abs()
            exc_c=torch.stack([a[cur==t].mean() for t in keep])
            exc_n=torch.stack([a[nxt==t].mean() if (nxt==t).any() else a.mean() for t in keep])
            rc, rn = spearman(name,exc_c), spearman(name,exc_n)
            null=sorted(abs(spearman(name[torch.randperm(keep.numel(),generator=g,device=DEV)],exc_c))
                        for _ in range(N_PERM))
            p95=null[int(0.95*N_PERM)]
            ok=max(abs(rc),abs(rn))>p95
            top=keep[exc_c.argsort(descending=True)[:8]]
            out['features'].append({'output_dir':p+1,'feature':j+1,'rho_current':rc,
                'rho_next':rn,'perm_null_p95':p95,'above_null':bool(ok),
                'top_tokens_by_excitation':[enc.decode([t]) for t in top.tolist()]})
            print(f"  {'dir %d feat %d'%(p+1,j+1):>12}  {rc:>17.3f}  {rn:>14.3f}  "
                  f"{p95:>15.3f}  {'SUPPORTED' if ok else 'not supported'}")
    n_ok=sum(f['above_null'] for f in out['features'])
    out['n_supported']=n_ok; out['n_features']=len(out['features'])
    print(f"\n{n_ok}/{len(out['features'])} feature namings beat their permutation null\n")
    for f in out['features']:
        print(f"  dir {f['output_dir']} feat {f['feature']} actually fires on: "
              f"{f['top_tokens_by_excitation']}")
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_layer17_verify.json','w'),indent=1)
    print('\nwrote bilin18_layer17_verify.json')

if __name__=='__main__': main()
