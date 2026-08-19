"""SPARSE READ CODE -- 369's next rung: the minimal faithful code
for the induction heads must include the pattern computation (legal:
double-QK scores are closed-form in the stream). Code: each head's
output = ONLY its top-1 read, z_h(q) = pat(q,k*) * vm(k*), with
pat from the real QK weights and k* = argmax|pat|. Diagnostics say
top-1 carries 46-65% of variance; does it carry the FUNCTION?
REGISTERED PREDICTIONS:
  (a) match-position CE cost of all-9 sparse coding <= 25% of the
      cost of fully mean-ablating the same 9 heads;
  (b) IOI margin >=60% retained under all-9 sparse coding;
  (c) CONTROL: shuffled top-key (k* permuted among a row's
      positions) costs >=3x the true sparse code at match
      positions."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sparse_read_code_results.json'
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()
    byl={}
    for li,hd in IND: byl.setdefault(li,[]).append(hd)
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
    def match_mask():
        M=torch.zeros(212,T,dtype=torch.bool)
        for r in range(212):
            toks=ROWS[r,:T].tolist(); last={}
            for q in range(T):
                t=toks[q]
                if t in last and last[t]+1<q: M[r,q]=True
                last[t]=q
        return M.reshape(-1)
    mk=match_mask()
    base=cl.ce_sweep([])
    code=cl.ce_sweep(code_hooks())
    dead=cl.ce_sweep(mean9())
    shuf=cl.ce_sweep(code_hooks(shuffle=True))
    cc=float((code-base)[mk].mean()); dd=float((dead-base)[mk].mean())
    ss=float((shuf-base)[mk].mean())
    pa=cc<=0.25*max(dd,1e-3)
    pc_=ss>=3*max(cc,1e-3)
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
    pb=mc>=0.6*mb
    out={'match_cost_code':round(cc,4),'match_cost_dead':round(dd,4),
         'match_cost_shuffled':round(ss,4),
         'ioi_margin_real':round(mb,3),'ioi_margin_code':round(mc,3),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_)}
    print(f'match CE: code {cc:+.4f} | dead {dd:+.4f} | shuffled '
          f'{ss:+.4f}')
    print(f'IOI margin: real {mb:+.3f} | code {mc:+.3f}')
    print(f"(a) code <=25% of dead: {'HELD' if pa else 'FAILED'}")
    print(f"(b) IOI >=60% retained: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled >=3x code: {'HELD' if pc_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
