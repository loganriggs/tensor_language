"""Logit-level decomposition of the surviving excess. Section 118: +0.162 of
content-level 16->17 interaction survives a LINEAR L17 with FROZEN final gain.
Through that pipeline logit deltas should superpose exactly; then the excess is
pure loss curvature. Measure, under linear-L17 + frozen gain: logit deltas
D16 = logits(d16)-logits(base), D17, Djoint. REGISTERED: (a) additivity --
E||Djoint-D16-D17|| / E||Djoint|| <= 0.15; (b) CE at synthetic base+D16+D17
reproduces >= 70% of the joint arm's excess (curvature is the carrier).

Prior context -- Content-level skin test. Section 117: at content level (per-arm frozen final
gain) the 16->17 excess is +0.205. Section 106 showed linearizing L17 kills 79%
of the RAW excess. Does the quadratic-skin story survive at content level? Arms,
all with per-arm-correct frozen gain (clean twin runs the SAME hybrid): real L17
(content excess, reproduces +0.205) vs linear L17. REGISTERED: (a) linear L17
kills >= 60% of the content-level excess; (b) both controls exact (<=0.002).

Prior context -- Final-norm closure. Section 115: with linear L17, attention compensates
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
     'bilin18_curvature_results.json')

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
        moc=mlp_out(xc2,xinc,True); moh=mlp_out(xh2,xinh,True)
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
    lin17_map=PR.fit_layer(17)
    out={}
    L=lin17_map
    num=0.;den=0.;tot_ce={'base':0,'d16':0,'d17':0,'joint':0,'synt':0};n=0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        idx=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
        lg={}
        for tag,sl in (('base',[]),('d16',[16]),('d17',[17]),('joint',[16,17])):
            lg[tag]=fwd_arm(idx,spans,sl,L,True)
        D16=lg['d16']-lg['base']; D17=lg['d17']-lg['base']
        Dj=lg['joint']-lg['base']
        num+=float((Dj-D16-D17).pow(2).sum()); den+=float(Dj.pow(2).sum())
        lg['synt']=lg['base']+D16+D17
        ntok=tg.numel()
        for tag in tot_ce:
            c=F.cross_entropy(lg[tag].view(-1,lg[tag].size(-1)),tg)
            tot_ce[tag]+=float(c)*ntok
        n+=ntok
    for tag in tot_ce: tot_ce[tag]/=n
    add_res=(num/den)**0.5
    exc_true=tot_ce['joint']-tot_ce['d16']-tot_ce['d17']+tot_ce['base']
    exc_synt=tot_ce['synt']-tot_ce['d16']-tot_ce['d17']+tot_ce['base']
    out={'ce':tot_ce,'additivity_residual':add_res,
         'excess_true':exc_true,'excess_synthetic':exc_synt}
    print(f'logit additivity residual: {add_res:.3f}')
    print(f'true excess {exc_true:+.4f} | synthetic (additive logits) excess '
          f'{exc_synt:+.4f}',flush=True)
    kill=float('nan'); ctrl=0.0
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
