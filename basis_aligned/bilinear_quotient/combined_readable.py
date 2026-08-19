"""COMBINED READABLE MODEL -- put both k-dials together: top-512
quadratic units in ALL 18 MLPs + 4-read code on the 120 cheapest
heads (the negative-cost set from 379). The whole model in sparse-
enumerable form: at every position, the computation is a list of
<=4 reads per coded head and <=512 active quadratic units per MLP.
REGISTERED PREDICTIONS:
  (a) all-18-MLP top-512 alone <= +0.15 grid;
  (b) COMBINED (MLPs + 120 coded heads) <= +0.25 grid;
  (c) combined FRESH <= +0.35;
  (d) IOI margin >= 70% under combined."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'combined_readable_results.json'
MIDS=list(range(18))

@torch.no_grad()
def main():
    t0=time.time()
    def hooks(k,random_units=False):
        hs=[]
        for li in MIDS:
            mlp=m.transformer.h[li].mlp
            def fh(mo,i_,o_,mlp=mlp,k=k,random_units=random_units):
                x=i_[0]
                h=mlp.Left(x)*mlp.Right(x)
                if random_units:
                    g=torch.Generator().manual_seed(4)
                    idx=torch.randperm(h.shape[-1],generator=g)[:k] \
                        .to(DEV)
                    msk=torch.zeros(h.shape[-1],device=DEV)
                    msk[idx]=1.0
                    h=h*msk
                else:
                    _,idx=h.abs().topk(k,dim=-1)
                    msk=torch.zeros_like(h).scatter(-1,idx,1.0)
                    h=h*msk
                return mlp.Down(h)+mlp.Down_bias
            hs.append(mlp.register_forward_hook(fh))
        return hs
    import json as j9
    order=[tuple(map(int,k.split('.'))) for k in
           j9.load(open(PT+'head_code_frontier_results.json'))
           ['cheapest_order']][:120]
    import sys as s_
    def head_hooks():
        byl={}
        for li,hd in order: byl.setdefault(li,[]).append(hd)
        hs=[]
        for li,hds in byl.items():
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hds=hds,at=at):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                are=s_.modules[type(at).__module__].apply_rotary_emb
                Bb,Tq=X.shape[0],X.shape[1]
                v=at.c_v(X).view(Bb,Tq,9,128)
                vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v)
                                          if v1 is not None else v)
                cos,sin=at.rotary(at.c_q(X).view(Bb,Tq,9,128))
                qf=F.rms_norm(at.c_q(X).view(Bb,Tq,9,128),(128,))
                kf=F.rms_norm(at.c_k(X).view(Bb,Tq,9,128),(128,))
                qf,kf=are(qf,cos,sin),are(kf,cos,sin)
                q2=F.rms_norm(at.c_q2(X).view(Bb,Tq,9,128),(128,))
                k2=F.rms_norm(at.c_k2(X).view(Bb,Tq,9,128),(128,))
                q2,k2=are(q2,cos,sin),are(k2,cos,sin)
                sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                kf.float())/128
                s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                k2.float())/128
                pat=(sc*s2)*torch.tril(torch.ones(Tq,Tq,device=DEV))
                z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                for hd in hds:
                    p1=pat[:,hd]
                    _,idx=p1.abs().topk(4,dim=-1)
                    msk=torch.zeros_like(p1).scatter(-1,idx,1.0)
                    z[:,hd]=torch.einsum('bqk,bkd->bqd',p1*msk,
                                         vm[:,:,hd].float())
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,Tq,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
        return hs
    base=cl.ce_sweep([])
    dm=cl.ce_sweep(hooks(512))
    am=round(float((dm-base).mean()),4)
    print(f'all-18 MLP top-512: {am:+.4f}',flush=True)
    dc=cl.ce_sweep(hooks(512)+head_hooks())
    cm=round(float((dc-base).mean()),4)
    print(f'combined: {cm:+.4f}',flush=True)
    FRESH=cl.fresh_rows(120)
    bF=cl.ce_sweep([],tok=FRESH)
    cF=cl.ce_sweep(hooks(512)+head_hooks(),tok=FRESH)
    fm=round(float((cF-bF).mean()),4)
    pr=cl.ioi_prompts()
    def margin(hooks9):
        ms=[]
        for txt,ti,ts in pr:
            ids=torch.tensor(cl.enc().encode(txt))[None,:].to(DEV)
            x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F.rms_norm(x,(D,)))/30)).float()[0,-1]
            ms.append(float(lg[ti]-lg[ts]))
        for h in hooks9: h.remove()
        return sum(ms)/len(ms)
    mb=margin([]); mc=margin(hooks(512)+head_hooks())
    pa=am<=0.15; pb=cm<=0.25; pc_=fm<=0.35; pd=mc>=0.7*mb
    out={'mlp18_top512':am,'combined':cm,'combined_fresh':fm,
         'ioi_real':round(mb,3),'ioi_combined':round(mc,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':bool(pd)}
    print(f'fresh {fm:+.4f} | IOI {mc:+.3f}/{mb:+.3f}')
    print(f"(a) MLPs <=+0.15: {'HELD' if pa else 'FAILED'}")
    print(f"(b) combined <=+0.25: {'HELD' if pb else 'FAILED'}")
    print(f"(c) fresh <=+0.35: {'HELD' if pc_ else 'FAILED'}")
    print(f"(d) IOI >=70%: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
