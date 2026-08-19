"""TOPK4 STACK -- 376 closed induction with 4-read code at 98.8%.
Apply the same 4-read code to ALL 162 heads at once: the whole
attention stack on four lines of readable code.
REGISTERED PREDICTIONS:
  (a) census-grid CE cost <= +0.10 total (all 162 heads);
  (b) FRESH cost <= +0.15 (travels);
  (c) IOI margin >= 85% retained;
  (d) r.3.0 member cost <= 15% of deleting its heads (the novel
      circuit's gap closes under 4 reads too)."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'topk4_stack_results.json'

byl_ref=[{}]

def code_hooks():
    import sys as s_
    hs=[]
    for li,hds in byl_ref[0].items():
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
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
            s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
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

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    byl_ref[0]={li:list(range(9)) for li in range(18)}
    base=cl.ce_sweep([])
    code=cl.ce_sweep(code_hooks())
    cc=float((code-base).mean())
    print(f'grid all-162 top-4: {cc:+.4f}',flush=True)
    FRESH=cl.fresh_rows(120)
    bF=cl.ce_sweep([],tok=FRESH)
    cF=cl.ce_sweep(code_hooks(),tok=FRESH)
    fr=float((cF-bF).mean())
    pr=cl.ioi_prompts()
    def margin(hooks):
        ms=[]
        for txt,ti,ts in pr:
            ids=torch.tensor(cl.enc().encode(txt))[None,:].to(DEV)
            x=F.rms_norm(m.transformer.wte(ids),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(
                F.rms_norm(x,(D,)))/30)).float()[0,-1]
            ms.append(float(lg[ti]-lg[ts]))
        for h in hooks: h.remove()
        return sum(ms)/len(ms)
    mb=margin([]); mc=margin(code_hooks())
    lf=cl.leaf('r.3.0'); mem=lf['member']
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    r30=float((code-base)[mm].abs().mean())
    byl_ref[0]={16:[8,2]}
    dele=cl.ce_sweep(cl.mean_head_hooks(16,[8,2]))
    r30d=float((dele-base)[mm].abs().mean())
    pa=cc<=0.10; pb=fr<=0.15; pc_=mc>=0.85*mb
    pd=r30<=0.15*max(r30d,1e-3)
    out={'grid':round(cc,4),'fresh':round(fr,4),
         'ioi_real':round(mb,3),'ioi_code':round(mc,3),
         'r30_member_code':round(r30,4),'r30_member_delete':round(r30d,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':bool(pd)}
    print(f'grid {cc:+.4f} | fresh {fr:+.4f} | IOI {mc:+.3f}/{mb:+.3f}'
          f' | r30 {r30:.4f} vs del {r30d:.4f}')
    print(f"(a) grid <=+0.10: {'HELD' if pa else 'FAILED'}")
    print(f"(b) fresh <=+0.15: {'HELD' if pb else 'FAILED'}")
    print(f"(c) IOI >=85%: {'HELD' if pc_ else 'FAILED'}")
    print(f"(d) r.3.0 <=15% of deletion: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
