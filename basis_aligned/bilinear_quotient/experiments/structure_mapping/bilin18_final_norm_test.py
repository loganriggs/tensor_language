"""Final-norm closure. Section 115: with linear L17, attention compensates
(freezing it doubles the excess), so the surviving +0.030 should live in the
final rms_norm -- the only nonlinearity left. Arms: linear-L17 with free
attention, final rms_norm's per-token scale FROZEN to the clean run's (the
normalized direction stays live; only the 1/rms gain is clamped). REGISTERED:
(a) freezing the final gain kills >= 60% of the +0.030 excess; (b) control:
freeze under no damage <= 0.005. Original locator docstring:

Where does the interaction excess that SURVIVES linear L17 live? Section 106:
with L17's MLP replaced by its linear stand-in, the 16->17 excess drops 0.143 ->
0.030. The surviving +0.030 must flow through a nonlinearity that is not L17's
MLP: either L17's ATTENTION (its pattern is a product of scores reading L16's
damaged write) or the final rms_norm before the unembedding.

Arms (span-ablation composition as section 106, all with linear L17 MLP):
(i) free L17 attention (reproduces +0.030); (ii) L17 attention output CLAMPED to
its no-damage values (computed in lockstep on the same input) -- if the excess
dies, attention carries it. REGISTERED PREDICTIONS: (a) clamping L17's attention
kills >= 60% of the surviving excess; (b) control: clamping under NO damage is
free (<= 0.005); alternative to (a): the excess survives attention-clamping ->
it lives in the final norm (also an answer)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
import bilin18_pipe_refit as PR
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_final_norm_results.json')

@torch.no_grad()
def fwd_arm(idx, spans, span_lis, lin17, clamp17):
    """Dual forward: clean reference and damaged run; damaged uses linear L17 MLP
    and (optionally) the clean run's L17 attention output."""
    B,T=idx.shape
    cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    def emb(): x=F.rms_norm(m.transformer.wte(idx),(D,)); return x,x,None
    xc,x0c,v1c=emb(); xh,x0h,v1h=emb()
    for li in range(18):
        blk=m.transformer.h[li]; a=blk.attn; mlp=blk.mlp
        def attn_step(x,x0,v1):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x; hcur=F.rms_norm(x,(D,))
            def qk(l):
                z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
                return apply_rot(z,cosb,sinb)
            v=a.c_v(hcur).view(B,T,NH,HD)
            v1n=v if v1 is None else v1
            v=(1-a.lamb)*v+a.lamb*v1n.view_as(v)
            q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
            s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            at=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            return xin,at,v1n
        xinc,atc,v1c=attn_step(xc,x0c,v1c)
        xinh,ath,v1h=attn_step(xh,x0h,v1h)
        # clamp17 repurposed: no attention clamping in this variant
        xc2=xinc+atc; xh2=xinh+ath
        def mlp_out(x2,xin,hybrid):
            xhat=F.rms_norm(x2,(D,))
            if li==17 and hybrid and lin17 is not None:
                xi=xin.reshape(-1,D).float()
                return ((xi-lin17['bx'])@lin17['W']+lin17['by']).to(x2.dtype).view_as(x2)
            return mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        moc=mlp_out(xc2,xinc,False); moh=mlp_out(xh2,xinh,True)
        if li in span_lis:
            Q,cbar=spans[li]
            c=moh.float().reshape(-1,D)@Q
            moh=moh-((c-cbar)@Q.T).to(moh.dtype).view_as(moh)
        xc=xc2+moc; xh=xh2+moh
    if clamp17:   # freeze final per-token gain to the clean run's
        rms_c=xc.float().pow(2).mean(-1,keepdim=True).sqrt()
        xn=(xh.float()/rms_c.clamp_min(1e-8)).to(xh.dtype)
    else:
        xn=F.rms_norm(xh,(D,))
    lg=m.lm_head(xn)
    return (30*torch.tanh(lg/30)).float()

@torch.no_grad()
def ce(spans,span_lis,lin17,clamp17):
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        lg=fwd_arm(b[:,:-1].contiguous(),spans,span_lis,lin17,clamp17)
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    spans={}
    for li in (16,17):
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Y=torch.cat(accs); Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Ybar).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Ybar@orth(Vh[:8].T)) if False else (orth(Vh[:8].T),None)
        Q=spans[li][0]; spans[li]=(Q,Ybar@Q)
    PR.LINS={}
    lin17=PR.fit_layer(17)
    out={}
    for tag,clamp in (('free',False),('norm_frozen',True)):
        b_=ce(spans,[],lin17,clamp)
        d16=ce(spans,[16],lin17,clamp)-b_
        d17=ce(spans,[17],lin17,clamp)-b_
        joint=ce(spans,[16,17],lin17,clamp)-b_
        exc=joint-d16-d17
        out[tag]={'base':b_,'d16':d16,'d17':d17,'excess':exc}
        print(f'{tag:8s}: base {b_:.4f} | d16 {d16:+.4f} | d17 {d17:+.4f} | '
              f'excess {exc:+.4f}',flush=True)
    kill=1-out['norm_frozen']['excess']/out['free']['excess'] \
         if out['free']['excess']>1e-6 else float('nan')
    ctrl=abs(out['norm_frozen']['base']-out['free']['base'])
    pa=kill>=0.6; pb=ctrl<=0.005
    out['kill']=kill; out['ctrl_gap']=ctrl
    out['pred_a']=bool(pa); out['ctrl_b']=bool(pb)
    print(f"\n(a) final norm carries the excess (kill >=60%): "
          f"{'HELD' if pa else 'FAILED'} ({kill if kill==kill else 0:.0%})")
    print(f"(b) clamp free under no damage (<=0.005): "
          f"{'HELD' if pb else 'VIOLATED'} ({ctrl:.4f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
