"""Spectral localization of L1's functional nonlinearity. Section 109: the top-48
span carries 12% of linearization cost at 59% of energy; the complement 53%.
Where in the spectrum does the cost live? Linearize the top-k span for k in
(48,128,256,512) plus complement-of-512. REGISTERED: (a) cost(top-k) <= 30% of
full through k=256; (b) monotone in k; (c) random-k controls track energy share
(within 3x). Original section-109 docstring:

Are L1's two crowns the same fact? L1 is the source of the 80-word functional
vocabulary (all readers' quadratics consume its top-48 output-PCA coordinates,
section 58) and the model's most functionally nonlinear layer (linearization
+0.282, section 107). If the functional nonlinearity IS the vocabulary channel,
linearizing only the top-48 span's component of L1's write should carry most of
the full cost.

Partial stand-in: mo' = mo_real + P(mo_lin - mo_real), P = projection onto the
span (or its complement). REGISTERED PREDICTIONS: (a) span-only >= 60% of the
full +0.282; (b) complement-only <= 40%; (c) additivity: span + complement within
15% of full. Control: a random-48 span's partial cost should sit near its
energy share, well below the principal span's."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
import bilin18_pipe_refit as PR
from tier2_model import rope_tables, apply_rot
NH,HD,D,K=9,128,1152,48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_spectral_nonlinearity_results.json')

@torch.no_grad()
def fwd_partial(idx, lin, Q=None, mode='full'):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(18):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        xin=x
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
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if li==1 and mode!='none':
            xi=xin.reshape(-1,D).float()
            mol=((xi-lin['bx'])@lin['W']+lin['by'])
            mof=mo.float().reshape(-1,D)
            diff=mol-mof
            if mode=='full':
                mo=mol.to(mo.dtype).view_as(mo)
            elif mode=='span':
                mo=(mof+(diff@Q)@Q.T).to(mo.dtype).view_as(mo)
            elif mode=='comp':
                mo=(mof+diff-(diff@Q)@Q.T).to(mo.dtype).view_as(mo)
        x=x+mo
    lg=m.lm_head(F.rms_norm(x,(D,)))
    return (30*torch.tanh(lg/30)).float()

@torch.no_grad()
def ce(lin,Q,mode):
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        lg=fwd_partial(b[:,:-1].contiguous(),lin,Q,mode)
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y=torch.cat(accs); Yc=Y-Y.mean(0)
    _,_,Vh=torch.linalg.svd(Yc.float(), full_matrices=False)
    Q=orth(Vh[:K].T)
    espan=float((Yc@Q).pow(2).sum())/float(Yc.pow(2).sum())
    g=torch.Generator(device=DEV).manual_seed(0)
    Qr=orth(torch.randn(D,K,device=DEV,generator=g))
    PR.LINS={}
    lin=PR.fit_layer(1)
    base=ce(lin,None,'none')
    full=ce(lin,None,'full')-base
    print(f'base {base:.4f} | full +{full:.4f}')
    out={'base':base,'full':full,'spans':{}}
    costs={}
    for k in (48,128,256,512):
        Qk=orth(Vh[:k].T)
        ek=float((Yc@Qk).pow(2).sum())/float(Yc.pow(2).sum())
        ck=ce(lin,Qk,'span')-base
        gk=torch.Generator(device=DEV).manual_seed(k)
        Qrk=orth(torch.randn(D,k,device=DEV,generator=gk))
        erk=float((Yc@Qrk).pow(2).sum())/float(Yc.pow(2).sum())
        crk=ce(lin,Qrk,'span')-base
        costs[k]=ck
        out['spans'][str(k)]={'cost':ck,'energy':ek,'rand_cost':crk,
                              'rand_energy':erk}
        print(f'top-{k:3d}: +{ck:.4f} ({ek:.0%} energy) | random-{k}: +{crk:.4f} '
              f'({erk:.0%} energy)',flush=True)
    c512=ce(lin,orth(Vh[:512].T),'comp')-base
    out['comp_512']=c512
    print(f'complement-of-512: +{c512:.4f}')
    pa=all(costs[k]<=0.3*full for k in (48,128,256))
    pb=all(costs[a]<=costs[b]+0.005 for a,b in ((48,128),(128,256),(256,512)))
    pc=all(out['spans'][str(k)]['rand_cost']<=
           3*out['spans'][str(k)]['rand_energy']*full+0.01 for k in (48,128,256,512))
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['ctrl_c']=bool(pc)
    print(f"(a) <=30% of full through k=256: {'HELD' if pa else 'FAILED'}")
    print(f"(b) monotone in k: {'HELD' if pb else 'FAILED'}")
    print(f"(c) random controls track energy: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
