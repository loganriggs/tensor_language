"""CIRCUIT CODE r.3.0 -- novel-circuit computational attempt v2.
Lessons applied: context-free tables capture 8.5% (368); one-read
pattern-as-code carries 77% for induction heads (370). r.3.0's
machinery is attention-side: heads (16,8),(16,2) + a6/a7 bundles.
THE CODE: replace heads 16.8 and 16.2 with the one-read code
z_h(q) = pat(q,k*) * vm(k*) (pattern from real weights). If the
circuit's named heads work by sparse coincidence reads, the code
must replicate behavior ON THE CIRCUIT'S MEMBERS.
REGISTERED PREDICTIONS:
  (a) member CE cost of the code <=30% of deleting both heads;
  (b) member-position contribution correlation (real vs code head
      output effect on member CE) proxied by: |code dCE - 0| vs
      |deletion dCE|: median |code dCE| <= 0.3 * median |deletion
      dCE| on members;
  (c) CONTROL: shuffled-key code costs >=2.5x the true code on
      members;
  (d) off-member cost reported (spillover)."""
import json, time, torch
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'circuit_code_r30_results.json'
TAG='r.3.0'; HEADS=[(16,8),(16,2)]

@torch.no_grad()
def main():
    t0=time.time()
    import sys as s_
    lf=cl.leaf(TAG); mem=lf['member']
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    bv=cl.base_ce()
    def hooks(mode):
        hs=[]
        byl={}
        for li,hd in HEADS: byl.setdefault(li,[]).append(hd)
        for li,hds in byl.items():
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hds=hds,at=at,mode=mode):
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
                    if mode=='delete':
                        z[:,hd]=z[:,hd].mean(1,keepdim=True); continue
                    p1=pat[:,hd]
                    ks=p1.abs().argmax(-1)
                    if mode=='shuffled':
                        g=torch.Generator().manual_seed(9)
                        for b in range(Bb):
                            pi=torch.randperm(Tq,generator=g).to(DEV)
                            ks[b]=torch.minimum(ks[b][pi],
                                torch.arange(Tq,device=DEV))
                    w=p1.gather(-1,ks[...,None]).squeeze(-1)
                    vv=vm[:,:,hd].float().gather(
                        1,ks[...,None].expand(-1,-1,128))
                    z[:,hd]=w[...,None]*vv
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,Tq,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
        return hs
    dcode=cl.ce_sweep(hooks('code'))-bv
    ddel=cl.ce_sweep(hooks('delete'))-bv
    dshuf=cl.ce_sweep(hooks('shuffled'))-bv
    cc=float(dcode[mm].abs().mean()); dd=float(ddel[mm].abs().mean())
    ss=float(dshuf[mm].abs().mean())
    off=float(dcode[~mm].abs().mean())
    mc=float(dcode[mm].abs().median()); md=float(ddel[mm].abs().median())
    pa=cc<=0.3*max(dd,1e-3)
    pb=mc<=0.3*max(md,1e-3)
    pc_=ss>=2.5*max(cc,1e-3)
    out={'tag':TAG,'heads':[f'{a9}.{b9}' for a9,b9 in HEADS],
         'member_abs_code':round(cc,4),'member_abs_delete':round(dd,4),
         'member_abs_shuffled':round(ss,4),'offmember_abs':round(off,4),
         'median_code':round(mc,4),'median_delete':round(md,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc_),
         'pred_d':True}
    print(f'members |dCE|: code {cc:.4f} | delete {dd:.4f} | '
          f'shuffled {ss:.4f} | off-member {off:.4f}')
    print(f"(a) code <=30% delete: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median <=30%: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled >=2.5x: {'HELD' if pc_ else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
