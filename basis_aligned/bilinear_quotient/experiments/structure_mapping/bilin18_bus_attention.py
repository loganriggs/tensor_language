"""Completing the bus origin: section 94 ruled out every MLP write 5-15 as the
syntax bus's supplier (all at the ~0.2s diffuse floor). The untested route is
ATTENTION: L16's attention step reads the whole sequence and may assemble the bus
content that its MLP then writes. Transplant L16's attention output (the residual
delta added by attn at L16) from different documents; measure movement of L16's
MLP write along the bus span vs the same random-span control as section 94.

REGISTERED PREDICTIONS: (a) the attention transplant moves the bus coordinates
>= 3x the section-94 floor (>= 0.6s, vs sources' 0.16-0.32s) -- the bus content
arrives through attention, not the stream; (b) specificity: bus movement / random-
span movement >= 1.5 (unlike every MLP source, which sat at ~1.0). Both failing
would mean the bus content is computed by L16's MLP from diffuse stream state --
also an answer."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH, HD, D = 9, 128, 1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_bus_attention_results.json')

ATT_PATCH={}   # {'src': tensor} -> replace L16 attn contribution

@torch.no_grad()
def fwd16(idx, capture_attn=False):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    mo16=None; attn16=None
    for li in range(17):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn; hcur=F.rms_norm(x,(D,))
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
        att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        if li==16:
            if capture_attn: attn16=att.detach()
            if 'src' in ATT_PATCH: att=ATT_PATCH['src'].to(att.dtype)
        x=x+att
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==16: mo16=mo.detach().reshape(-1,D).float()
        x=x+mo
    return mo16, attn16

def main():
    t0=time.time()
    base_rows=FW[300:324,:257].to(DEV); src_rows=FW[400:424,:257].to(DEV)
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=16, acc=acc); accs.append(acc[0])
    Y16=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y16-Y16.mean(0)).float(), full_matrices=False)
    BUS=orth(Vh[:8].T)
    g=torch.Generator(device=DEV).manual_seed(0)
    RND=orth(torch.randn(D,8,device=DEV,generator=g))
    b16,_=fwd16(base_rows)
    sig_bus=(b16@BUS).std(0); sig_rnd=(b16@RND).std(0)
    _,attn_src=fwd16(src_rows, capture_attn=True)
    ATT_PATCH['src']=attn_src
    p16,_=fwd16(base_rows)
    del ATT_PATCH['src']
    mv_bus=float((((p16-b16)@BUS)/sig_bus).abs().mean())
    mv_rnd=float((((p16-b16)@RND)/sig_rnd).abs().mean())
    pa=mv_bus>=0.6; pb=mv_bus/max(mv_rnd,1e-9)>=1.5
    out={'bus_move':mv_bus,'random_move':mv_rnd,
         'specificity':mv_bus/max(mv_rnd,1e-9),
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'L16 attention transplant: bus move {mv_bus:.3f}s | random-span '
          f'{mv_rnd:.3f}s | specificity {mv_bus/max(mv_rnd,1e-9):.2f}')
    print(f"(a) above the diffuse floor (>=0.6s): {'HELD' if pa else 'FAILED'}")
    print(f"(b) bus-specific (>=1.5x random): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
