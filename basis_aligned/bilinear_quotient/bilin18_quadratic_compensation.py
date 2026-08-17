"""Is L17's nonlinear residue a damage absorber? Section 102 sidebar: under
linearized L17, ablating L16's span costs +0.312 vs +0.211 with the real L17 --
the quadratic residue reduced upstream damage by a third. If that is gain-control
(renormalizing its input distribution) it should generalize across damage types.

Damages at L16's write: (i) top-8 PCA span ablation; (ii) random-8 span ablation
x2 seeds; (iii) additive noise at matched energy. Each evaluated under real vs
linearized L17. REGISTERED PREDICTIONS: (a) generality -- real-L17 damage <=
0.8x linearized-L17 damage for at least 3 of 4 damage types (the residue
compensates broadly, not just for the specific span); (b) the compensation is
NOT explained by the stand-in's imperfection: an L16-untouched control shows
|CE(linearized) - CE(real)| <= 0.11 base gap, and (a)'s comparison uses damage
DELTAS from each arm's own base."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_quadratic_compensation_results.json')
LIN17={}; DMG={}

@torch.no_grad()
def ce_eval():
    hs=[]
    if LIN17:
        blk=m.transformer.h[17]; state={}
        hs.append(blk.register_forward_pre_hook(
            lambda mod,inp: state.__setitem__('x',inp[0].detach())))
        def mlp_hook(mod,i_,o_):
            x=state['x'].reshape(-1,D).float()
            return ((x-LIN17['bx'])@LIN17['W']+LIN17['by']).to(o_.dtype).view_as(o_)
        hs.append(blk.mlp.register_forward_hook(mlp_hook))
    if DMG:
        kind=DMG['kind']
        def d_hook(mod,i_,o_):
            if kind=='span':
                Q,cbar=DMG['Q'],DMG['cbar']
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            if kind=='noise':
                return o_+DMG['vec'].to(o_.dtype)
        hs.append(m.transformer.h[16].mlp.register_forward_hook(d_hook))
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    global LIN17,DMG
    t0=time.time()
    accs=[]
    for i in range(0,36,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs); Ybar=Y.mean(0)
    _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
    Qp=orth(Vh[:8].T)
    en8=float(((Y-Ybar)@Qp).pow(2).sum(1).mean())
    g=torch.Generator(device=DEV).manual_seed(0)
    dmgs=[('pca8',{'kind':'span','Q':Qp,'cbar':Ybar@Qp})]
    for s_ in (1,2):
        Qr=orth(torch.randn(D,8,device=DEV,
                            generator=torch.Generator(device=DEV).manual_seed(s_)))
        dmgs.append((f'rand8_{s_}',{'kind':'span','Q':Qr,'cbar':Ybar@Qr}))
    nv=torch.randn(D,device=DEV,generator=g); nv=nv/nv.norm()*en8**0.5
    dmgs.append(('noise',{'kind':'noise','vec':nv}))
    ins=[]; mos=[]
    for i in range(0,60,6):
        a_,b_=fwd_all(FW[i:i+6,:257].to(DEV))
        ins.append(a_[17]); mos.append(b_[17])
    X=torch.cat(ins); Ym=torch.cat(mos)
    bx=X.mean(0); by=Ym.mean(0)
    Xc=X-bx; Yc=Ym-by
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                         Xc.T@Yc/Xc.shape[0])
    out={'damage':{}}
    for tag,lin in (('real',{}),('linearized',{'W':W,'bx':bx,'by':by})):
        LIN17=lin; DMG={}
        base=ce_eval()
        out[tag+'_base']=base
        for name,d in dmgs:
            DMG=d
            delta=ce_eval()-base
            out['damage'].setdefault(name,{})[tag]=delta
            print(f'{tag:10s} {name:8s}: +{delta:.4f}',flush=True)
        DMG={}
    LIN17={}
    comp=sum(1 for name in out['damage']
             if out['damage'][name]['real']<=0.8*out['damage'][name]['linearized'])
    gap=abs(out['linearized_base']-out['real_base'])
    pa=comp>=3; pb=gap<=0.11
    out['n_compensated']=comp; out['base_gap']=gap
    out['pred_a']=bool(pa); out['ctrl_b']=bool(pb)
    print(f"\n(a) broad compensation (>=3/4 at <=0.8x): {'HELD' if pa else 'FAILED'} ({comp}/4)")
    print(f"control base gap <=0.11: {'HELD' if pb else 'VIOLATED'} ({gap:.3f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
