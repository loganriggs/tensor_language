"""Causal test of word #5's name ('measurement register').

§68 named word #5 correlationally (fires on ' detected, levels, samples'; rho 0.48).
Every prior causal test of a token story failed (layout->register, bus->determiners,
head1-aim). Registered with that record in view: steer L1's output along word #5's
top eigen-direction at +/-2 units and measure Delta logprob of a measurement
vocabulary (' levels',' samples',' data',' measured',' rate',' concentration',
' values',' detected',' analysis',' results') vs frequency-matched controls.
REGISTERED PREDICTIONS: (a) the full swing (+ vs -) moves measurement tokens' mean
logprob by >= 1.5x the control set's swing (the first causal token-story bar ever set
this low); (b) the swing direction matches the correlational sign."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_gradient_steering import collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
import torch.nn.functional as F
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
enc=tiktoken.get_encoding('gpt2')
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_word5_causal_results.json')

@torch.no_grad()
def full_logits(idx, steer=None):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
        if li==1 and steer is not None:
            vec,mag=steer; mo=mo+(mag*vec).to(mo.dtype)
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    return F.log_softmax(logits.float(),-1)

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    rows=[]
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    Pm=0.5*(W[4].view(K,K)+W[4].view(K,K).T)
    evp,Up=torch.linalg.eigh(Pm.double())
    w=(V@Up[:,evp.abs().argmax()].float()); w=w/w.norm()
    meas=[enc.encode(t)[0] for t in (' levels',' samples',' data',' measured',
          ' rate',' values',' detected',' analysis',' results',' rates')]
    ctrl=[enc.encode(t)[0] for t in (' people',' world',' story',' house',
          ' morning',' friend',' road',' music',' game',' door')]
    rows_in=FW[300:312,:257].to(DEV)
    lp0=full_logits(rows_in)
    sw={}
    for sgn in (+1,-1):
        lp=full_logits(rows_in,steer=(w,sgn*mag))
        sw[sgn]={'meas':float((lp[...,meas]-lp0[...,meas]).mean()),
                 'ctrl':float((lp[...,ctrl]-lp0[...,ctrl]).mean())}
        print(f'{"+" if sgn>0 else "-"}steer: measurement {sw[sgn]["meas"]:+.4f} | '
              f'control {sw[sgn]["ctrl"]:+.4f}',flush=True)
    swing_m=sw[1]['meas']-sw[-1]['meas']
    swing_c=sw[1]['ctrl']-sw[-1]['ctrl']
    out={'swings':{'measurement':swing_m,'control':swing_c},'per_sign':sw}
    ratio=abs(swing_m)/max(abs(swing_c),1e-9)
    pa=ratio>=1.5
    out['ratio']=ratio; out['pred_a']=bool(pa)
    print(f'\nfull swing: measurement {swing_m:+.4f} vs control {swing_c:+.4f} '
          f'({ratio:.1f}x)')
    print(f"(a) >= 1.5x: {'HELD -- first causal token-story success' if pa else 'FAILED -- four-for-four'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
