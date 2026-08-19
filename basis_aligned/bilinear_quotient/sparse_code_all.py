"""SPARSE CODE ALL -- cash 370 into coverage: replace ALL 69
sparse-reader heads (head_read_census top1_share >= 0.4) with the
one-line code z_h(q) = pat(q,k*) * vm(k*), everywhere. 43% of the
attention stack running on readable one-read code.
REGISTERED PREDICTIONS:
  (a) census-grid CE cost <= +0.15 total;
  (b) FRESH data (120 never-seen rows, Ledger 22): cost <= 1.5x the
      fit-window cost;
  (c) IOI margin >= 70% retained;
  (d) CONTROL: the same substitution on 69 DIFFUSE heads (lowest
      top-1 share) costs >= 3x."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sparse_code_all_results.json'

def head_sets():
    import json as j9
    prof=j9.load(open(PT+'head_read_census_results.json'))['profiles']
    items=[(k,v['top1_share']) for k,v in prof.items()]
    sparse=[tuple(map(int,k.split('.'))) for k,v in items if v>=0.4]
    diffuse=[tuple(map(int,k.split('.'))) for k,v in
             sorted(items,key=lambda kv:kv[1])[:len(sparse)]]
    return sparse,diffuse

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    SPARSE,DIFFUSE=head_sets()
    print(f'sparse {len(SPARSE)} | diffuse control {len(DIFFUSE)}',
          flush=True)
    def mkbyl(hs9):
        byl={}
        for li,hd in hs9: byl.setdefault(li,[]).append(hd)
        return byl
    byl=mkbyl(SPARSE)
    def code_hooks(shuffle=False):
        hs=[]
        for li,hds in byl.items():
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hds=hds,at=at,shuffle=shuffle):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                import sys as s_
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
                    ks=p1.abs().argmax(-1)
                    if shuffle:
                        g=torch.Generator().manual_seed(9)
                        qq=torch.arange(Tq)
                        ks=torch.stack([(ks[b].cpu()*0+
                            (qq[torch.randperm(Tq,generator=g)]
                             .clamp(max=1)*0+ks[b].cpu()
                             [torch.randperm(Tq,generator=g)]))
                            for b in range(Bb)]).to(DEV)
                        ks=torch.minimum(ks,torch.arange(
                            Tq,device=DEV)[None,:])
                    w=p1.gather(-1,ks[...,None]).squeeze(-1)
                    vv=vm[:,:,hd].float().gather(
                        1,ks[...,None].expand(-1,-1,128))
                    z[:,hd]=w[...,None]*vv
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,Tq,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
        return hs
    def mean9():
        hs=[]
        for li,hds in byl.items(): hs+=cl.mean_head_hooks(li,hds)
        return hs
    base=cl.ce_sweep([])
    code=cl.ce_sweep(code_hooks())
    cc=float((code-base).mean())
    byl=mkbyl(DIFFUSE)
    diff=cl.ce_sweep(code_hooks())
    dd=float((diff-base).mean())
    byl=mkbyl(SPARSE)
    FRESH=cl.fresh_rows(120)
    baseF=cl.ce_sweep([],tok=FRESH)
    codeF=cl.ce_sweep(code_hooks(),tok=FRESH)
    cf=float((codeF-baseF).mean())
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
    pa=cc<=0.15
    pb=cf<=1.5*max(cc,1e-3)
    pc_=mc>=0.7*mb
    pd=dd>=3*max(cc,1e-3)
    out={'n_sparse':len(SPARSE),'grid_cost':round(cc,4),
         'fresh_cost':round(cf,4),'diffuse_cost':round(dd,4),
         'ioi_real':round(mb,3),'ioi_code':round(mc,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':bool(pd)}
    print(f'grid {cc:+.4f} | fresh {cf:+.4f} | diffuse-ctl {dd:+.4f} '
          f'| IOI {mc:+.3f}/{mb:+.3f}')
    print(f"(a) grid <=+0.15: {'HELD' if pa else 'FAILED'}")
    print(f"(b) fresh <=1.5x grid: {'HELD' if pb else 'FAILED'}")
    print(f"(c) IOI >=70%: {'HELD' if pc_ else 'FAILED'}")
    print(f"(d) diffuse >=3x: {'HELD' if pd else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
