"""BIAS LINEARIZES MLP5 (v2) -- 447: v1 was VOID (writeup 447):
it subtracted the RAW constant from mlp5's RMS-NORMALISED input
(norm ~6800 against ~34) and omitted the layer's Down_bias, so
every arm exploded to 14-16 nats and the exactness check failed
at 1.6e-2. Both fixed here: the constant's share of the
normalised input is computed per position from the true raw
residual (captured from the block input, the attention output and
the lambda mix), and Down_bias is included.
The claim under test is exact algebra for a non-gated bilinear
layer (config.gated is False for this model):
    Down[(L(x+c)) * (R(x+c))] + b
  = Down[Lx*Rx] + Down[Lx*Rc] + Down[Lc*Rx] + Down[Lc*Rc] + b
The two cross terms are LINEAR in x, so a constant offset opens a
linear pathway through an otherwise purely quadratic layer. That
would explain why a network with no activation functions needs a
large learned constant at all.
REGISTERED PREDICTIONS:
  (a) EXACTNESS first (this gates the rest): the four terms plus
      Down_bias reconstruct mlp5's real output with relative
      error < 1e-3; if this FAILS the run is void again and no
      arm is interpreted;
  (b) LINEARIZATION: removing the two cross terms costs >= 0.5 x
      the full bias deletion (0.9154), i.e. most of the bias's
      value is the linear pathway;
  (c) the pure constant term Down[Lc*Rc] costs < 0.20."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_linearizes_results.json'
NR=32
ARMS=['cross','const_term','both']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    mlp=m.transformer.h[LJ].mlp
    L=mlp.Left.weight.float(); R=mlp.Right.weight.float()
    Dw=mlp.Down.weight.float(); Db=mlp.Down_bias.detach().float()
    lam=m.transformer.h[LJ].lambdas.detach().float()
    acc={a:[0.0,0] for a in ARMS}; relerr=[]; FIRED={}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4; cap={}
        hs=[m.transformer.h[LJ].register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('xb',
                    a_[0].detach().float())),
            m.transformer.h[LJ].attn.register_forward_hook(
                lambda mo,i_,o_: cap.__setitem__('aout',
                    o_[0].detach().float())),
            m.transformer.h[LJ].attn.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0])),
            mlp.register_forward_hook(
                lambda mo,i_,o_: cap.update(
                    {'min':i_[0].detach().float(),
                     'mout':o_.detach().float()}))]
        E=F.rms_norm(m.transformer.wte(idx),(D,))
        x=E; x0=E; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        # the head's constant, correctly scaled (443 lesson)
        at=m.transformer.h[LJ].attn; X=cap['X']
        cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
        def rot(w):
            return are(F.rms_norm(w(X).view(B,T,9,128),
                       (128,))[:,:,HD][:,:,None],cos,sin)[:,:,0]
        qf,kf=rot(at.c_q),rot(at.c_k); q2,k2=rot(at.c_q2),rot(at.c_k2)
        pat=((torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())/128)
             *(torch.einsum('bqd,bkd->bqk',q2.float(),k2.float())/128)) \
            *torch.tril(torch.ones(T,T,device=DEV))
        v=at.c_v(X).view(B,T,9,128)[:,:,HD].float()*(1-at.lamb)
        Wp=at.c_proj.weight.float()[:,HD*128:(HD+1)*128]
        const=(torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T).mean(dim=(0,1))
        # raw residual entering mlp5, and the constant's share of
        # the NORMALISED input
        xraw=lam[0]*cap['xb']+lam[1]*E.float()+cap['aout']
        scale=(D**0.5)/xraw.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        cn=const[None,None,:]*scale
        xin=cap['min']; xr=xin-cn
        lq,rq=xr@L.T,xr@R.T; lc,rc=cn@L.T,cn@R.T
        quad=(lq*rq)@Dw.T; cross=((lq*rc)+(lc*rq))@Dw.T
        ct=(lc*rc)@Dw.T
        recon=quad+cross+ct+Db
        relerr.append(float((recon-cap['mout']).norm()
                            /cap['mout'].norm().clamp_min(1e-6)))
        def run(arm):
            hlist=[]
            if arm is not None:
                def mh(mo,inp,o_,arm=arm):
                    xi=inp[0].float()
                    xr2=xi-cn
                    l2,r2=xr2@L.T,xr2@R.T
                    lc2,rc2=cn@L.T,cn@R.T
                    q2_=(l2*r2)@Dw.T
                    cr=((l2*rc2)+(lc2*r2))@Dw.T
                    ct2=(lc2*rc2)@Dw.T
                    full=q2_+cr+ct2+Db
                    out={'cross':full-cr,'const_term':full-ct2,
                         'both':full-cr-ct2}[arm]
                    return out.to(o_.dtype)
                hlist.append(mlp.register_forward_hook(mh))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx
            v1b=None
            for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h_ in hlist: h_.remove()
            return ce
        base=run(None)
        for a in ARMS:
            c=run(a); acc[a][0]+=c-base; acc[a][1]+=1
            if abs(c-base)>1e-9: FIRED[a]=FIRED.get(a,0)+1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()}
    re_=sum(relerr)/len(relerr)
    dead=[a for a in ARMS if not FIRED.get(a)]
    if dead:
        print(f'*** ARMS THAT NEVER FIRED: {dead} -- VOID ***')
    pa=re_<1e-3
    pb=CE['cross']>=0.5*0.9154
    pc=CE['const_term']<0.20
    out={'dce':CE,'reconstruction_rel_error':float(f'{re_:.3e}'),
         'arms_never_fired':dead,'pred_a':bool(pa),
         'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('dCE by removed term:',CE)
    print(f'reconstruction relative error {re_:.3e}')
    for nm,v in (('a','EXACTNESS <1e-3 (gates the rest)'),
                 ('b','cross terms >=0.5x bias value'),
                 ('c','pure constant term <0.20')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    if not pa: print('EXACTNESS FAILED -> arms not interpreted')
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
