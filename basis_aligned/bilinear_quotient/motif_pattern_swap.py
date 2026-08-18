"""MOTIF PATTERN SWAP -- exploit the census (280): replace the pattern
side of every prev-motif head (27 heads, 11 layers) with the LITERAL
one-hot previous-token pattern scaled by one fitted gain per head, and
every self-motif head (47 heads) with one-hot self. The head still uses
its real (lambda-mixed) values and c_proj -- only WHERE it looks is
replaced by the motif sentence. If this is cheap, the pattern side of 74
of 162 heads compresses to two sentences plus 74 numbers.
Gains: alpha_h = <z_h, v_target>/<v_target, v_target> averaged over the
fit window (window A), computed per head from the real pattern's output.
Arms (eval on window C, clean model otherwise):
  prev-swap (27 heads) / self-swap (47) / both (74)
  control: the same one-hot-prev swap applied to 27 random NON-prev
  heads (matched count), fixed seed.
REGISTERED PREDICTIONS:
  (a) both-swap CE cost <= +0.15 (74 pattern-halves for two sentences);
  (b) control >= 3x the prev-swap cost;
  (c) per-arm costs reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'motif_pattern_swap_results.json'
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
    arms={}
    arms['prev']=evalCE(install({li:('prev',hh)
                                 for li,hh in prevh.items()}))-base
    arms['self']=evalCE(install({li:('self',hh)
                                 for li,hh in selfh.items()}))-base
    def install_both():
        cfg={}
        hs=[]
        layers=set(list(prevh)+list(selfh))
        for li in layers:
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            ph=prevh.get(li,[]); sh=selfh.get(li,[])
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
    arms['both']=evalCE(install_both())-base
    arms['control']=evalCE(install({li:('prev',hh)
                                    for li,hh in ctrlh.items()}))-base
    for k,v in arms.items(): print(f'{k:8s} {v:+.4f}',flush=True)
    pa=arms['both']<=0.15
    pb=arms['control']>=3*max(arms['prev'],1e-3)
    out={'base':round(base,4),
         'arms':{k:round(v,4) for k,v in arms.items()},
         'n_prev':nprev,'n_self':nself,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) both-swap <= +0.15: {'HELD' if pa else 'FAILED'}")
    print(f"(b) control >= 3x prev: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
