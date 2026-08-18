"""Stage F prep: CONFIRMATION atlas -- rows 120-300 (45k tokens), never
fingerprinted, for confirming wave-2 provisional certifications on data
neither wave optimized against. Same construction as Stage A.

Original Stage A docstring follows:
 All 36 component fingerprints
(MLP top-8 span deletion; attention mean-ablation) over corpus rows
300-512 -- 212 rows x 256 targets = 54,272 tokens, 3.3x the original
atlas, none previously fingerprinted. Saved split into DISCOVERY (even
rows) and REPLICATION (odd rows) halves; span fits from rows 0-120
(disjoint from evaluation). Registered: (a) runtime < 45 min; (b) sanity:
per-component net damage sign matches the original atlas for >= 30/36
components (same instrument, bigger window)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_fingerprints import attn_mean, NH, HD
from tier2_model import rope_tables, apply_rot
D=1152; R0,R1=120,300
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_atlas_third_results.json'

@torch.no_grad()
def pertok(mlp_span=None, attn_layer=None):
    hs=[]
    if mlp_span is not None:
        li,(Q,cbar)=mlp_span
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[li].mlp.register_forward_hook(hook))
    ces=[]
    for i in range(R0,R1,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        if attn_layer is None:
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        else:
            ali,amu=attn_layer
            cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
            cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
            mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for li2 in range(18):
                blk=m.transformer.h[li2]; a=blk.attn
                x=blk.lambdas[0]*x+blk.lambdas[1]*x0
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
                att=a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v)
                             .reshape(B,T,-1))
                if li2==ali:
                    att=amu[None,None,:].to(att.dtype).expand_as(att)
                x=x+att
                xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
                x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none'))
    for h in hs: h.remove()
    return torch.cat(ces)

@torch.no_grad()
def main():
    t0=time.time()
    ce0=pertok()
    fps={}
    for li in range(18):
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        Q=orth(Vh[:8].T)
        fps[f'mlp{li}']=(pertok(mlp_span=(li,(Q,Yb@Q)))-ce0).cpu()
        print(f'mlp{li} {time.time()-t0:.0f}s',flush=True)
    for li in range(18):
        mu=attn_mean(li)
        fps[f'attn{li}']=(pertok(attn_layer=(li,mu))-ce0).cpu()
        print(f'attn{li} {time.time()-t0:.0f}s',flush=True)
    rows=torch.arange(R0,R1)
    rowmask_even=((rows-R0)%2==0)
    tokrow=torch.repeat_interleave(torch.arange(len(rows)),256)
    even=rowmask_even[tokrow]
    torch.save({'base':ce0.cpu(),'fingerprints':fps,'rows':(R0,R1),
                'even_mask':even},PT+'circuit_atlas_third.pt')
    d0=torch.load(PT+'bilin18_fingerprint_atlas.pt',weights_only=False)
    agree=sum(1 for k in fps
              if (fps[k].mean()>0)==(d0['fingerprints'][k].float().mean()>0))
    pa=(time.time()-t0)<2700; pb=agree>=30
    out={'n_tokens':int(ce0.numel()),'sign_agree':agree,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\ntokens {ce0.numel()} | sign agreement {agree}/36")
    print(f"(a) <45min: {'HELD' if pa else 'FAILED'}")
    print(f"(b) signs agree >=30/36: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
