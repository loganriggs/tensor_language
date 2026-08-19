"""How does head 1 transport the L0 signal: through its values, or its pattern?

§46: freezing head 1's full context kills 96% of the L0->L1 steered effect. A head's
context is pattern x values; a static L0-leader offset in the stream could enter either
(values: the offset is copied; pattern: the offset changes q/k geometry). REGISTERED
PREDICTION: value-mediated -- freezing only head 1's VALUE vectors (v for head 1 to
unsteered values) kills >= 70% of the effect, freezing only its PATTERN kills < 30%.
Control: same two freezes on head 6 (a 0%-mediation head), both predicted < 10%."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
CFG={'steer':None,'v_freeze':None,'p_freeze':None,'cap_v':None,'cap_p':None}
RUN={}

@torch.no_grad()
def run(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    c1=None; caps={}
    for li in range(2):
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
        if li==1:
            caps['v']=v.detach().clone(); caps['p']=pat.detach().clone()
            if CFG['v_freeze'] is not None:
                h=CFG['v_freeze']; v=v.clone(); v[:,:,h,:]=CFG['cap_v'][:,:,h,:]
            if CFG['p_freeze'] is not None:
                h=CFG['p_freeze']; pat=pat.clone(); pat[:,h]=CFG['cap_p'][:,h]
        ctx=torch.einsum('bhqk,bkhd->bqhd',pat,v)
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
    CFG.update({'steer':None,'v_freeze':None,'p_freeze':None})
    c1b,caps=run(rows); s1=float(c1b.std())
    CFG['cap_v']=caps['v']; CFG['cap_p']=caps['p']
    def arm(v_freeze=None,p_freeze=None):
        CFG.update({'steer':(d0L0,2*s0),'v_freeze':v_freeze,'p_freeze':p_freeze})
        c1p,_=run(rows)
        CFG.update({'steer':None,'v_freeze':None,'p_freeze':None})
        return float((c1p-c1b).mean())/s1
    full=arm()
    out={'full':full,'arms':{}}
    print(f'steer alone: {full:+.3f} sigma')
    for tag,vf,pf in (('h1 values',1,None),('h1 pattern',None,1),
                      ('h6 values',6,None),('h6 pattern',None,6)):
        d=arm(vf,pf); k=1-d/max(full,1e-9)
        out['arms'][tag]={'dc1':d,'killed':k}
        print(f'  freeze {tag:>10}: {d:+.3f}  ({100*k:.0f}% killed)',flush=True)
    kv=out['arms']['h1 values']['killed']; kp=out['arms']['h1 pattern']['killed']
    pa=kv>=0.70 and kp<0.30
    pc=abs(out['arms']['h6 values']['killed'])<0.10 and \
       abs(out['arms']['h6 pattern']['killed'])<0.10
    out['pred_value_mediated']=bool(pa); out['ctrl_ok']=bool(pc)
    print(f"\nvalue-mediated (v>=70%, p<30%): {'HELD' if pa else 'FAILED'} | "
          f"h6 control: {'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_head1_route_results.json','w'),indent=1)
    print(f'wrote bilin18_head1_route_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
