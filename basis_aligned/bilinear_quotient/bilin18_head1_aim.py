"""Where does head 1 re-aim under L0-leader steering?

§47: the L0->L1 edge is pattern-dominant through head 1 -- the steering changes where
head 1 attends. The L0 leader is the punctuation-vs-content axis (rho 0.95, §20).
REGISTERED PREDICTION: under +2 sigma steering, head 1's absolute pattern mass shifts
TOWARD punctuation/layout keys -- the mean |pattern| on punctuation-class key tokens
rises by at least 2x the rise on matched content-class keys. Control: head 6 (inert in
the mediation sweep) shows < 1.3x class asymmetry."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
import tiktoken
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
enc=tiktoken.get_encoding('gpt2')
PUNCT=set()
for s_ in ['.',',','!','?',';',':',')','(','"',"'",'\n','-','/','#','<','>']:
    t=enc.encode(s_)
    if len(t)==1: PUNCT.add(t[0])
CONTENT=set()
for s_ in [' time',' people',' work',' world',' day',' man',' life',' way',
           ' house',' water',' school',' state',' family',' student',' group']:
    t=enc.encode(s_)
    if len(t)==1: CONTENT.add(t[0])
STEER=None

@torch.no_grad()
def run(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
            return pat
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==0 and STEER is not None:
            dv,delta=STEER; mo=mo+(delta*dv).to(mo.dtype)
        x=x+mo

def main():
    global STEER
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
    rows=FW[300:324,:257].to(DEV)
    ids=rows
    isp=torch.zeros_like(ids,dtype=torch.bool)
    for t in PUNCT: isp|=(ids==t)
    isc=torch.zeros_like(ids,dtype=torch.bool)
    for t in CONTENT: isc|=(ids==t)
    print(f'punct keys: {int(isp.sum())} | content keys: {int(isc.sum())}')
    STEER=None; pat_b=run(rows)
    STEER=(d0L0,2*s0); pat_s=run(rows); STEER=None
    out={'heads':{}}
    for h in (1,6):
        db=pat_s[:,h].abs()-pat_b[:,h].abs()      # (B,Tq,Tk)
        keymass=db.sum(1)                          # (B,Tk) change in mass per key
        mp=float(keymass[isp].mean()); mc=float(keymass[isc].mean())
        ratio=mp/max(abs(mc),1e-9)
        out['heads'][h]={'d_mass_punct':mp,'d_mass_content':mc,'ratio':ratio}
        print(f'head {h}: delta|pattern| mass on punct keys {mp:+.4f} vs content '
              f'{mc:+.4f} (ratio {ratio:+.1f})',flush=True)
    r1=out['heads'][1]['ratio']; r6=out['heads'][6]['ratio']
    pa=r1>=2.0; pc=abs(r6)<1.3
    out['pred_held']=bool(pa); out['ctrl_ok']=bool(pc)
    print(f"\nprediction (head-1 punct/content shift >= 2x): "
          f"{'HELD' if pa else 'FAILED'} | head-6 control: "
          f"{'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_head1_aim_results.json','w'),indent=1)
    print(f'wrote bilin18_head1_aim_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
