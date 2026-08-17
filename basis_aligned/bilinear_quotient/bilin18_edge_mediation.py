"""Is the new L0->L1 edge mediated by attention block 1?

§44.3 found steering L0's leader moves L1's leader at unit gain despite a 0.5%
proximate-writer share; the proposed route is L0-write -> residual -> attn1 -> z.
REGISTERED PREDICTION: freezing attn1's context (all heads patched to their unsteered
values) while steering L0's leader kills >= 80% of the Delta c1; freezing only head 4
kills >= 60% (head 4 computes z per §21). Control: freezing attn0's context instead
(upstream of the steering injection point) must NOT reduce Delta c1."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
CFG={'steer':None,'freeze1':None,'freeze0':None,'cap1':None,'cap0':None}
RUN={}

@torch.no_grad()
def run(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    c1=None; caps={}
    for li in range(3):
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
        ctx=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        caps[li]=ctx.detach().clone()
        if li==1 and CFG['freeze1'] is not None:
            hs=CFG['freeze1']
            ref=CFG['cap1']
            for h in hs: ctx[:,:,h,:]=ref[:,:,h,:]
        if li==0 and CFG['freeze0'] is not None:
            ctx=CFG['cap0'].clone()
        x=x+a.c_proj(ctx.reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==0 and CFG['steer'] is not None:
            dv,delta=CFG['steer']; mo=mo+(delta*dv).to(mo.dtype)
        if li==1:
            c1=torch.einsum('bti,ij,btj->bt',xhat.float(),RUN['M1'],xhat.float())
        x=x+mo
    return c1,caps

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d0L0=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    s0=float((((Y0-Y0.mean(0)).float())@d0L0).std())
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh1=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    RUN['M1']=form_for_direction(m.transformer.h[1].mlp,
                                orth(Vh1[:32].T)[:,0].float()).float()
    rows=FW[300:324,:257].to(DEV)
    CFG.update({'steer':None,'freeze1':None,'freeze0':None})
    c1b,caps_b=run(rows); s1=float(c1b.std())
    CFG['cap1']=caps_b[1]; CFG['cap0']=caps_b[0]
    def arm(steer,freeze1=None,freeze0=None):
        CFG.update({'steer':steer,'freeze1':freeze1,'freeze0':freeze0})
        c1p,_=run(rows)
        CFG.update({'steer':None,'freeze1':None,'freeze0':None})
        return float((c1p-c1b).mean())/s1
    full=arm((d0L0,2*s0))
    fr_all=arm((d0L0,2*s0),freeze1=list(range(NH)))
    fr_h4=arm((d0L0,2*s0),freeze1=[4])
    fr_a0=arm((d0L0,2*s0),freeze0=True)
    out={'dc1_steer':full,'dc1_freeze_attn1':fr_all,'dc1_freeze_head4':fr_h4,
         'dc1_freeze_attn0':fr_a0}
    k_all=1-fr_all/max(full,1e-9); k_h4=1-fr_h4/max(full,1e-9)
    k_a0=1-fr_a0/max(full,1e-9)
    out['killed_attn1']=k_all; out['killed_head4']=k_h4; out['killed_attn0']=k_a0
    print(f'steer alone:            Delta c1 = {full:+.3f} sigma')
    print(f'  + freeze attn1 (all): {fr_all:+.3f}  ({100*k_all:.0f}% killed)')
    print(f'  + freeze head 4 only: {fr_h4:+.3f}  ({100*k_h4:.0f}% killed)')
    print(f'  + freeze attn0 (ctrl):{fr_a0:+.3f}  ({100*k_a0:.0f}% killed)')
    pa=k_all>=0.8; pb=k_h4>=0.6; pc=k_a0<0.3
    out['pred_attn1']=bool(pa); out['pred_head4']=bool(pb); out['ctrl_ok']=bool(pc)
    print(f"\nattn1 mediation >=80%: {'HELD' if pa else 'FAILED'} | head4 >=60%: "
          f"{'HELD' if pb else 'FAILED'} | attn0 control <30%: "
          f"{'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_edge_mediation_results.json','w'),indent=1)
    print(f'wrote bilin18_edge_mediation_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
