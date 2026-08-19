"""Completes bilin18_layer17.py: the forms say WHEN layer 17's MLP fires; this says
WHAT it does when it fires.

bilin18_layer17.py established that 4 output directions x rank-2 forms reproduce the
layer to within 0.7% of its total effect on cross-entropy, and named the input
directions of the leading form. The sentence is only finished once the OUTPUT
directions are named too: writing along P_p adds to the residual stream, which the
final norm and unembedding then read, so each P_p has a token profile of its own.

Reported per output direction: the tokens it promotes and demotes, its share of the
MLP's output variance, and -- the part that matters -- the eigen-decomposition of its
own form, so each row reads as "when <these squared features> are large, promote
<these tokens> and demote <those>"."""
import json, sys, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl'); sys.path.insert(0,'/workspace/tensor_language')
import tiktoken
from tier2_model import load_elriggs
from bilin18_identifiable import mlp_inputs, form_for_direction
from bilin18_whitened import sqrtm_psd
from bilin18_layer17 import out_pcs, LAYER, DEV, N_FIT

enc=tiktoken.get_encoding('gpt2')
def names(v, wte, n=10, sign=+1):
    a=(wte@v)*sign
    return [enc.decode([t]) for t in a.argsort(descending=True)[:n].tolist()]

def main():
    model,cfg=load_elriggs('bilin18', device=DEV)
    tokens=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens.pt')
    P,mu,evr=out_pcs(model, tokens, LAYER, 8)
    X=mlp_inputs(model,tokens,(LAYER,),N_FIT)[LAYER].to(DEV)
    S=X.T@X/X.shape[0]; Sh,Sih=sqrtm_psd(S)
    # the final norm is applied before the unembedding; it is a scale, so direction survives
    wte=model.transformer.wte.weight.detach().float()
    mlp=model.transformer.h[LAYER].mlp
    out={'layer':LAYER,'directions':[]}
    print(f'== layer {LAYER}: the four output directions that carry 90% of the MLP ==\n')
    for p in range(4):
        d=P[p].float(); share=float(evr[p])
        M=form_for_direction(mlp,d/d.norm()); Mw=Sh@M@Sh
        ev,U=torch.linalg.eigh(Mw); idx=ev.abs().argsort(descending=True)[:2]
        W=(Sih@U[:,idx]).float(); W=W/W.norm(dim=0,keepdim=True); lam=ev[idx]
        tot=float(ev.abs().sum())
        rec={'output_variance_share':share,
             'promotes':names(d,wte),'demotes':names(d,wte,sign=-1),
             'features':[{'eigenvalue':float(lam[j]),'share':float(lam[j].abs())/tot,
                          'aligned_tokens':names(W[:,j],wte,8)} for j in range(2)]}
        out['directions'].append(rec)
        print(f"OUTPUT DIRECTION {p+1}  ({100*share:.1f}% of the MLP's output variance)")
        print(f"  writing along it PROMOTES: {rec['promotes']}")
        print(f"  and DEMOTES:               {rec['demotes']}")
        for j,f in enumerate(rec['features']):
            print(f"    feature {j+1}: lambda {f['eigenvalue']:+.2e} ({100*f['share']:.0f}% of the "
                  f"form) -> {'ADDS to' if f['eigenvalue']>0 else 'SUBTRACTS from'} that direction")
            print(f"       squared projection onto: {f['aligned_tokens']}")
        print()
    json.dump(out, open('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_layer17_readout.json','w'), indent=1)
    print('wrote bilin18_layer17_readout.json')

if __name__=='__main__': main()
