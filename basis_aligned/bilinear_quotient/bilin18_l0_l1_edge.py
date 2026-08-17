"""Completing the front of the graph: is there an L0-leader -> L1-leader edge?

§17's writer decomposition says L1's leader is 76% attn1 x attn1 with emb x mlp0 at
0.5% -- so MLP0's direct quadratic contribution to the L1 leader is tiny ON
DISTRIBUTION. The causal-abstraction question is interventional: does steering L0's
leader (the rho-0.95 punctuation axis) move L1's leader coefficient? REGISTERED
PREDICTION (an absence claim, falsifiable): steering the L0 leader by +/-2 sigma moves
c1 by LESS than 0.1 sigma_c1 -- the two front leaders are causally near-independent,
and the graph's front is parallel rather than chained. Control: the same steering must
move L0's own downstream (its solo deletion costs 0.0153, so the steering is potent --
verified by measuring Delta CE of the steering itself)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
STEER=None   # (d0_layer0, delta)

@torch.no_grad()
def run(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    c1=None; ce=None
    for li in range(len(m.transformer.h)):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==0 and STEER is not None:
            dv,delta=STEER
            mo=mo+(delta*dv).to(mo.dtype)
        if li==1:
            c1=torch.einsum('bti,ij,btj->bt',xhat.float(),RUN['M1'],xhat.float())
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    V_=logits.shape[-1]
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V_).float(),
                       idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return c1,ce

RUN={}
def main():
    global STEER
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    import json as js
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d0L0=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    s0=float((((Y0-Y0.mean(0)).float())@d0L0).std())
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh1=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    d0L1=orth(Vh1[:32].T)[:,0].float()
    RUN['M1']=form_for_direction(m.transformer.h[1].mlp,d0L1).float()
    rows=FW[300:324,:257].to(DEV)
    STEER=None; c1b,ceb=run(rows)
    s1=float(c1b.std())
    out={'sigma_c1':s1,'arms':{}}
    for sgn in (+1,-1):
        STEER=(d0L0,sgn*2*s0)
        c1p,cep=run(rows)
        STEER=None
        dz=float((c1p-c1b).mean())/s1
        dce=float((cep-ceb).mean())
        out['arms'][sgn]={'dc1_sigma':dz,'dce':dce}
        print(f'{"+"if sgn>0 else "-"}2s steering of L0 leader: '
              f'Delta c1 = {dz:+.3f} sigma | Delta CE = {dce:+.4f}',flush=True)
    mx=max(abs(v['dc1_sigma']) for v in out['arms'].values())
    potent=any(abs(v['dce'])>0.003 for v in out['arms'].values())
    out['pred_absence_held']=bool(mx<0.1); out['steering_potent']=bool(potent)
    print(f"\nabsence prediction (|Delta c1| < 0.1 sigma): "
          f"{'HELD' if out['pred_absence_held'] else 'FAILED'} (max {mx:.3f})")
    print(f"steering potency control (|Delta CE| > 0.003): "
          f"{'OK' if potent else 'WEAK -- absence claim not licensed'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_l0_l1_edge_results.json','w'),indent=1)
    print(f'wrote bilin18_l0_l1_edge_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
