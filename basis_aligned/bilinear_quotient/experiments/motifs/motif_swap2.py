"""MOTIF PATTERN SWAP v2 -- v1 failed both bars AND its random-head
control improved CE, indicting the instrument (no reconstruction null).
v2: (i) RECONSTRUCTION NULL -- replay every motif layer through the
recompute path with its REAL patterns; must cost ~0 or all arms are
void; (ii) per-layer recon offsets subtracted from all marginals;
(iii) PER-HEAD GREEDY -- swap each of the 74 motif heads alone, adopt
those costing <= +0.01 corrected, evaluate the adopted set.
REGISTERED PREDICTIONS:
  (a) |global recon null| <= 0.02 (else instrument invalid, arms void);
  (b) >=30 of 74 heads accept their motif sentence at <= +0.01
      (recon-corrected);
  (c) adopted-set total <= +0.10 (recon-corrected)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'motif_swap2_results.json'
CA,CB=300,512; R0,R1=120,300

@torch.no_grad()
def main():
    t0=time.time()
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    prevh={}; selfh={}
    for li,hd,mo,fr in mt:
        if mo=='prev': prevh.setdefault(li,[]).append(hd)
        if mo=='self': selfh.setdefault(li,[]).append(hd)
    nprev=sum(len(v) for v in prevh.values())
    nself=sum(len(v) for v in selfh.values())
    print(f'prev heads {nprev} in {sorted(prevh)} | '
          f'self heads {nself} in {sorted(selfh)}',flush=True)
    g=torch.Generator().manual_seed(0)
    allh=[(li,hd) for li in range(18) for hd in range(9)]
    nonprev=[(li,hd) for li,hd in allh
             if hd not in prevh.get(li,[])]
    ctrl=[nonprev[i] for i in torch.randperm(len(nonprev),
          generator=g)[:nprev].tolist()]
    ctrlh={}
    for li,hd in ctrl: ctrlh.setdefault(li,[]).append(hd)
    mod=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod.apply_rotary_emb
    T=256
    def head_z(at,X,v1):
        B=X.shape[0]
        q=at.c_q(X).view(B,T,9,128); k=at.c_k(X).view(B,T,9,128)
        q2=at.c_q2(X).view(B,T,9,128); k2=at.c_k2(X).view(B,T,9,128)
        v=at.c_v(X).view(B,T,9,128)
        if v1 is None: v1=v
        vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
        cos,sin=at.rotary(q)
        q=F.rms_norm(q,(128,)); k=F.rms_norm(k,(128,))
        q,k=are(q,cos,sin),are(k,cos,sin)
        q2=F.rms_norm(q2,(128,)); k2=F.rms_norm(k2,(128,))
        q2,k2=are(q2,cos,sin),are(k2,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',q.float(),k.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
        pat=sc*sc2
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        pat=pat*mask
        z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
        return z,vm.float()
    # fit alphas on window A: capture attn inputs + v1
    swaps=set(list(prevh)+list(selfh)+list(ctrlh))
    caps={li:{'x':[],'v1':[]} for li in swaps}
    hs=[]
    for li in swaps:
        def mk(li=li):
            def h(mo_,args):
                caps[li]['x'].append(args[0].detach())
                caps[li]['v1'].append(args[1].detach()
                                      if args[1] is not None else None)
            return h
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    for i in range(CA,CA+32,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    ALPHA={}
    for li in swaps:
        num=torch.zeros(9,device=DEV); den=torch.zeros(9,device=DEV)
        nums=torch.zeros(9,device=DEV); dens=torch.zeros(9,device=DEV)
        for X,v1 in zip(caps[li]['x'],caps[li]['v1']):
            z,vm=head_z(m.transformer.h[li].attn,X,v1)
            vprev=torch.zeros_like(vm); vprev[:,1:]=vm[:,:-1]
            vp=vprev.permute(0,2,1,3)      # (B,H,T,dh)
            vs=vm.permute(0,2,1,3)
            num+=(z*vp).sum((0,2,3)); den+=(vp*vp).sum((0,2,3))
            nums+=(z*vs).sum((0,2,3)); dens+=(vs*vs).sum((0,2,3))
        ALPHA[li]=(num/den.clamp_min(1e-9),
                   nums/dens.clamp_min(1e-9))
        caps[li]=None
    def install(cfg):
        # cfg: {li: (mode, [heads])} mode 'prev' or 'self'
        hs=[]
        for li,(mode,heads) in cfg.items():
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            def h(mo_,args,out,li=li,at=at,heads=heads,mode=mode,
                  ap=ap,asf=asf):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_z(at,X,v1)
                if mode=='prev':
                    vt=torch.zeros_like(vm); vt[:,1:]=vm[:,:-1]
                    al=ap
                else:
                    vt=vm; al=asf
                vt=vt.permute(0,2,1,3)
                for hd in heads:
                    z[:,hd]=al[hd]*vt[:,hd]
                B=X.shape[0]
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X.dtype))
                return (ynew,v1r)
            hs.append(at.register_forward_hook(h))
        return hs
    def evalCE(hooks):
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hooks: h.remove()
        return float(torch.cat(ces).mean())
    base=evalCE([])
    def install_cfg(cfg):
        hs=[]
        for li,dd in cfg.items():
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            ph=dd.get('prev',[]); sh=dd.get('self',[])
            def h(mo_,args,out,at=at,ph=ph,sh=sh,ap=ap,asf=asf):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_z(at,X,v1)
                vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
                vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
                for hd in ph: z[:,hd]=ap[hd]*vp[:,hd]
                for hd in sh: z[:,hd]=asf[hd]*vs[:,hd]
                B=X.shape[0]
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X.dtype))
                return (ynew,v1r)
            hs.append(at.register_forward_hook(h))
        return hs
    layers=sorted(set(list(prevh)+list(selfh)))
    recon=evalCE(install_cfg({li:{} for li in layers}))-base
    print(f'GLOBAL RECON NULL: {recon:+.4f}',flush=True)
    roff={}
    for li in layers:
        roff[li]=evalCE(install_cfg({li:{}}))-base
    print('per-layer recon offsets:',
          {li:round(v,4) for li,v in roff.items()},flush=True)
    per={}; adopted={}
    nad=0
    for mode,fam in (('prev',prevh),('self',selfh)):
        for li,hh in fam.items():
            for hd in hh:
                c=evalCE(install_cfg({li:{mode:[hd]}}))-base-roff[li]
                per[f'L{li}h{hd}.{mode}']=round(c,4)
                if c<=0.01:
                    adopted.setdefault(li,{}).setdefault(mode,[])                        .append(hd)
                    nad+=1
    print('per-head corrected marginals:',per,flush=True)
    tot=(evalCE(install_cfg(adopted))-base
         -sum(roff[li] for li in adopted)) if adopted else 0.0
    pa=abs(recon)<=0.02; pb=nad>=30; pc=tot<=0.10
    out={'base':round(base,4),'recon':round(recon,4),
         'recon_offsets':{li:round(v,4) for li,v in roff.items()},
         'per_head':per,'n_adopted':nad,
         'adopted':{li:d for li,d in adopted.items()},
         'adopted_total_corrected':round(tot,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f'adopted {nad}/74 | adopted-set corrected total {tot:+.4f}')
    print(f"(a) |recon| <= 0.02: {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=30/74 free swaps: {'HELD' if pb else 'FAILED'}")
    print(f"(c) adopted set <= +0.10: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
