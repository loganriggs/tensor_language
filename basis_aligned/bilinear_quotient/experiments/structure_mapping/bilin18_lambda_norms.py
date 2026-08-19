"""Relative norms of the lambda-mixed terms (user request). Each block computes
x = lambda0*x_prev + lambda1*x0 (x0 = token embeddings), then adds attention and
MLP writes. The lambda table alone (section 111) shows lambda1 ~ 8 everywhere and
lambda0 ~ 0.01/0.07 at L1/L5. Here: the actual RMS norms of each term on data,
per layer -- how dominant is the embedding re-injection really?

REGISTERED PREDICTIONS: (a) at L1 and L5, ||lambda1*x0|| >= 2x ||lambda0*x_prev||
(true stream resets); (b) from L7 on, the stream term dominates the embedding
term (>= 2x) -- re-injection matters early, stream wins late; (c) the MLP-write
share of the post-block norm declines with depth (the dilution law from the
input side)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_lambda_norms_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    stats={li:{'prev':0.,'emb':0.,'attn':0.,'mlp':0.,'post':0.,'n':0}
           for li in range(18)}
    for i in range(0,24,6):
        idx=FW[i:i+6,:257].to(DEV)
        B,T=idx.shape
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        for li in range(18):
            blk=m.transformer.h[li]
            tp=blk.lambdas[0]*x; te=blk.lambdas[1]*x0
            x=tp+te
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
            at=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            x=x+at
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            x=x+mo
            s=stats[li]
            s['prev']+=float(tp.float().pow(2).mean()); s['emb']+=float(te.float().pow(2).mean())
            s['attn']+=float(at.float().pow(2).mean()); s['mlp']+=float(mo.float().pow(2).mean())
            s['post']+=float(x.float().pow(2).mean()); s['n']+=1
    print(f"{'L':>3} {'|l0*x_prev|':>11} {'|l1*x0|':>9} {'|attn|':>8} {'|mlp|':>8} {'|post|':>8}")
    for li in range(18):
        s=stats[li]; n=s['n']
        for k in ('prev','emb','attn','mlp','post'): s[k]=(s[k]/n)**0.5
        print(f"{li:>3} {s['prev']:>11.2f} {s['emb']:>9.2f} {s['attn']:>8.2f} "
              f"{s['mlp']:>8.2f} {s['post']:>8.2f}",flush=True)
    pa=all(stats[li]['emb']>=2*stats[li]['prev'] for li in (1,5))
    pb=all(stats[li]['prev']>=2*stats[li]['emb'] for li in range(7,18))
    shares=[stats[li]['mlp']/stats[li]['post'] for li in range(5,18)]
    inv=sum(1 for i in range(len(shares)-1) if shares[i+1]>shares[i])
    pc=inv<=3
    out={str(li):{k:stats[li][k] for k in ('prev','emb','attn','mlp','post')}
         for li in range(18)}
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) L1/L5 embedding-dominated (>=2x): {'HELD' if pa else 'FAILED'}")
    print(f"(b) stream dominates from L7 (>=2x): {'HELD' if pb else 'FAILED'}")
    print(f"(c) mlp share declines with depth (<=3 inversions): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
