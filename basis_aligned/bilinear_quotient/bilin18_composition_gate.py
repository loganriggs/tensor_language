"""The composition gate: do the verified surrogates survive being installed TOGETHER?

Every compression in this program was verified with the rest of the model intact, and
this model is superadditive under ablation (§10/§27), so joint replacement could fail
while each part passes alone. There is also one known interaction in the set: the
L16->L17 syntax-bus edge (§24) means L17's replacement was fit on inputs produced by an
INTACT layer 16.

The three verified whole-unit surrogates installed together:
    L1  leader-direction surrogate  (a(u.x)^2+b along d0; §19)
    L16 whole-layer replacement     (R=4 output dirs x rank-2 whitened forms; §10)
    L17 whole-layer replacement     (R=4 x rank-2; §8/§10)

REGISTERED PREDICTION: joint damage <= 1.3x the sum of individual damages. The three
sites are far apart and only one tested edge (16->17) links them, so mild interaction
at most; if the bus edge matters, the joint number exceeds the sum mainly through the
17-given-replaced-16 term, which is separable by also measuring {16,17} alone.

Arms: base / each alone / {16,17} / all three. 7 evaluations.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, eval_ce
from bilin18_joint_removal import orth
from bilin18_identifiable import form_for_direction
from bilin18_whitened import sqrtm_psd, truncate
from bilin18_layer17 import Truncated, out_pcs
from bilin18_identifiable import mlp_inputs

DEV='cuda'
model, cfg = load_elriggs('bilin18', device=DEV)
tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/bilin18_eval_tokens.pt')
d = cfg['n_embd']
base = eval_ce(model, tokens, batch=4)
print(f'base CE {base:.4f}')

def whole_layer_repl(li, R=4, k=2):
    mlp = model.transformer.h[li].mlp
    V, mu, ev = out_pcs(model, tokens, li, 512)
    P = V[:R]
    X = mlp_inputs(model, tokens, (li,), 6000)[li].to(DEV)
    S = X.T @ X / X.shape[0]
    Sh, Sih = sqrtm_psd(S)
    bias = mlp.Down_bias.detach().float()
    forms = torch.stack([form_for_direction(mlp, P[p]) for p in range(R)])
    Fw = torch.stack([Sih @ truncate(Sh @ forms[p] @ Sh, k) @ Sih for p in range(R)])
    return Truncated(P.float(), Fw.float(), (mu - bias).float(), bias.float()).to(DEV)

def leader_surrogate(li=1):
    # rebuild the section-19 surrogate for L1's leader
    from bilin18_source_folding import forward_tracked
    from bilin18_joint_removal import fwd, FW
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs); _,_,Vh=torch.linalg.svd((Y-Y.mean(0)).float(), full_matrices=False)
    d0=orth(Vh[:32].T)[:,0].float()
    Xl=[]
    for i in range(0,96,6):
        _,xh,_=forward_tracked(FW[i:i+6,:513].to(DEV)); Xl.append(xh)
    Xh=torch.cat(Xl)
    M=form_for_direction(model.transformer.h[1].mlp, d0).float()
    c=torch.einsum('ni,ij,nj->n',Xh,M,Xh)
    S=(Xh.T@Xh/Xh.shape[0]).double(); ev,U=torch.linalg.eigh(S)
    kd=ev>1e-8*ev.max()
    Sih=(U[:,kd]*ev[kd].rsqrt())@U[:,kd].T; Shh=(U[:,kd]*ev[kd].sqrt())@U[:,kd].T
    Mw=Shh@M.double()@Shh; ew,Uw=torch.linalg.eigh(Mw)
    u=(Sih@Uw[:,ew.abs().argmax()]).float(); u=u/u.norm()
    p2=(Xh@u)**2
    co=torch.linalg.lstsq(torch.stack([p2,torch.ones_like(p2)],1),c[:,None]).solution.squeeze()
    a_,b_=float(co[0]),float(co[1])
    mlp1=model.transformer.h[1].mlp; orig=mlp1.forward
    def new_forward(x):
        mo=orig(x)
        cb=mo.float()@d0
        chat=a_*(x.float()@u)**2+b_
        return mo+((chat-cb)[...,None]*d0).to(mo.dtype)
    return new_forward

def main():
    t0=time.time()
    repl={16: whole_layer_repl(16), 17: whole_layer_repl(17)}
    l1f = leader_surrogate()
    orig={li: model.transformer.h[li].mlp.forward for li in (1,16,17)}
    def ce_with(sites):
        if 1 in sites: model.transformer.h[1].mlp.forward = l1f
        for li in (16,17):
            if li in sites: model.transformer.h[li].mlp.forward = repl[li].forward
        try: return eval_ce(model, tokens, batch=4) - base
        finally:
            for li in (1,16,17): model.transformer.h[li].mlp.forward = orig[li]
    out={'base_ce': base, 'arms': {}}
    arms=[('L1 leader',[1]),('L16',[16]),('L17',[17]),('L16+L17',[16,17]),
          ('all three',[1,16,17])]
    res={}
    for tag,S in arms:
        v=ce_with(S); res[tag]=v
        out['arms'][tag]=v
        print(f'  {tag:12s} +{v:.4f}', flush=True)
    ssum=res['L1 leader']+res['L16']+res['L17']
    ratio=res['all three']/max(ssum,1e-9)
    pair_int=res['L16+L17']-res['L16']-res['L17']
    out['sum_individual']=ssum; out['joint_over_sum']=ratio
    out['pair_16_17_interaction']=pair_int
    print(f'\n  sum of individuals +{ssum:.4f} | all three +{res["all three"]:.4f} '
          f'| ratio {ratio:.2f}x')
    print(f'  16-17 interaction term: {pair_int:+.4f} '
          f'(the bus-edge contribution to non-additivity)')
    held = ratio <= 1.3
    out['prediction_held']=bool(held)
    print(f'  registered prediction (joint <= 1.3x sum): {"HELD" if held else "FAILED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out, open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_composition_gate_results.json','w'), indent=1)
    print(f'\nwrote bilin18_composition_gate_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
