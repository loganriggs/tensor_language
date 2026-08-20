"""BIAS LINEARIZES MLP5 -- 446: the constant head 5.7 broadcasts
must reach mlp5 (removing it from everything AFTER layer 5 costs
0.24 nats against 0.92 for full deletion -- mlp5 delivers ~74% of
its value). Why would an MLP need a constant input offset? In a
BILINEAR layer the answer is exact algebra:
    Down[(L(x+c)) * (R(x+c))]
  = Down[Lx*Rx] + Down[Lx*Rc] + Down[Lc*Rx] + Down[Lc*Rc]
The two CROSS terms are LINEAR in x. So a constant offset turns
part of a purely quadratic layer into a linear operator -- the
bias may exist to give the model a linear pathway through mlp5.
Test by decomposing mlp5's output into the four exact terms and
pricing each: run the model with each term individually removed
from mlp5's output (c = the head's mean write, x = the rest of
mlp5's normalised input).
REGISTERED PREDICTIONS:
  (a) LINEARIZATION: deleting the two cross terms costs >= 0.5 x
      the full bias deletion (0.92), i.e. most of the bias's value
      is the linear pathway it opens, not the constant it adds;
  (b) the pure constant term Down[Lc*Rc] costs < 0.20;
  (c) EXACTNESS sanity: the four reconstructed terms sum to
      mlp5's actual output with relative error < 1e-3."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_linearizes_results.json'
NR=32
ARMS=['none','cross','const_term','quad_only','all_bias']

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    mlp=m.transformer.h[LJ].mlp
    L=mlp.Left.weight.float(); R=mlp.Right.weight.float()
    Dw=mlp.Down.weight.float()
    acc={a:[0.0,0] for a in ARMS}; relerr=[]
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B=4; cap={}
        h=m.transformer.h[LJ].attn.register_forward_pre_hook(
            lambda mo_,args: cap.__setitem__('X',args[0]))
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        h.remove()
        # the head's constant (its mean write), correctly scaled
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
        def run(arm):
            hs=[]
            if arm!='none':
                def mh(mo,inp,o_,arm=arm):
                    xin=inp[0].float()          # normalised input
                    # split xin = xr + cn, where cn is the bias's
                    # share of the normalised input
                    sc=(xin.norm(dim=-1,keepdim=True)
                        /max(float(const.norm()),1e-6))
                    cn=const.to(xin.dtype)[None,None,:] \
                        *(xin.norm(dim=-1,keepdim=True)
                          /xin.norm(dim=-1,keepdim=True))
                    cn=const[None,None,:].expand_as(xin)*0 \
                        +const[None,None,:]
                    xr=xin-cn
                    lq=xr@L.T; rq=xr@R.T
                    lc=cn@L.T; rc=cn@R.T
                    quad=(lq*rq)@Dw.T
                    cross=((lq*rc)+(lc*rq))@Dw.T
                    ct=(lc*rc)@Dw.T
                    full=quad+cross+ct
                    if arm=='exact': return full.to(o_.dtype)
                    if arm=='cross': out=full-cross
                    elif arm=='const_term': out=full-ct
                    elif arm=='quad_only': out=quad
                    elif arm=='all_bias': out=quad
                    else: out=full
                    return out.to(o_.dtype)
                hs.append(mlp.register_forward_hook(mh))
            xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx
            v1b=None
            for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(xx,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h_ in hs: h_.remove()
            return ce
        base=run('none')
        # exactness check
        capm={}
        hh=mlp.register_forward_hook(
            lambda mo,inp,o_: capm.update(
                {'in':inp[0].detach().float(),
                 'out':o_.detach().float()}))
        xx=F.rms_norm(m.transformer.wte(idx),(D,)); x0b=xx; v1b=None
        for blk in m.transformer.h: xx,v1b=blk(xx,v1b,x0b)
        hh.remove()
        xin=capm['in']; cn=const[None,None,:].expand_as(xin)
        xr=xin-cn
        lq,rq=xr@L.T,xr@R.T; lc,rc=cn@L.T,cn@R.T
        recon=((lq*rq)+(lq*rc)+(lc*rq)+(lc*rc))@Dw.T
        relerr.append(float((recon-capm['out']).norm()
                            /capm['out'].norm().clamp_min(1e-6)))
        for a in ARMS:
            if a=='none': continue
            acc[a][0]+=run(a)-base; acc[a][1]+=1
        print(f'batch {i} done',flush=True)
    CE={a:round(v[0]/max(v[1],1),4) for a,v in acc.items()
        if a!='none'}
    re_=sum(relerr)/len(relerr)
    pa=CE['cross']>=0.5*0.9154
    pb=CE['const_term']<0.20
    pc=re_<1e-3
    out={'dce':CE,'reconstruction_rel_error':round(re_,6),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print('dCE by removed term:',CE)
    print(f'reconstruction relative error {re_:.2e}')
    for nm,v in (('a','cross terms carry >=0.5x the bias value'),
                 ('b','pure constant term <0.20'),
                 ('c','exact decomposition (<1e-3)')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
