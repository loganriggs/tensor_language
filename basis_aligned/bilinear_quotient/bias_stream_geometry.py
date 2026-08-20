"""BIAS STREAM GEOMETRY -- 449: the bias is a SPECIFIC direction,
not a scale device (448): replacing head 5.7's constant with a
random vector of identical norm costs 6.30 nats -- seven times
worse than deleting the head outright (0.92) -- while halving or
doubling the true constant costs only 0.046 and 0.139. Magnitude
is loosely tuned; direction is essential. And its value is
non-additive (447): local ablations find a quarter of it.
That profile fits one picture: the bias is the residual stream's
own central direction, the offset every later layer's rms_norm
and every later weight matrix is calibrated around. Test it
geometrically, no ablation needed.
Measure, at each layer 5-17: the cosine between the head's
constant and (i) the mean residual direction, (ii) the top
principal direction of the residual, plus the share of residual
norm the constant accounts for. Null: random vectors.
REGISTERED PREDICTIONS:
  (a) CENTRAL DIRECTION: |cos| between the constant and the mean
      residual direction is >= 0.5 at a majority of layers 6-17;
  (b) NULL: that exceeds the random-vector null (~0.03) by at
      least 10x;
  (c) report the constant's share of the residual norm per layer
      (is the stream mostly this one vector?)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bias_stream_geometry_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    accs={li:{'mean':torch.zeros(D),'n':0,'norm':0.0,
              'samples':[]} for li in range(5,18)}
    consts=[]
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hs=[m.transformer.h[LJ].attn.register_forward_pre_hook(
                lambda mo_,a_: cap.__setitem__('X',a_[0]))]
        for li in range(5,18):
            hs.append(m.transformer.h[li].register_forward_pre_hook(
                (lambda li: lambda mo_,a_: cap.__setitem__(
                    f'r{li}',a_[0].detach().float()))(li)))
        E=F.rms_norm(m.transformer.wte(idx),(D,))
        x=E; x0=E; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        cap['r18']=x.detach().float()
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
        consts.append((torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T)
                      .mean(dim=(0,1)).cpu())
        for li in range(5,18):
            r=cap[f'r{li}'].reshape(-1,D)
            a=accs[li]
            a['mean']+=r.sum(0).cpu(); a['n']+=r.shape[0]
            a['norm']+=float(r.norm(dim=-1).sum())
            if len(a['samples'])<3:
                a['samples'].append(r[::37].cpu())
        print(f'batch {i} done',flush=True)
    const=torch.stack(consts).mean(0)
    cu=const/const.norm().clamp_min(1e-6)
    g=torch.Generator().manual_seed(5)
    rnds=[torch.randn(D,generator=g) for _ in range(5)]
    rnds=[r/r.norm() for r in rnds]
    rows=[]
    for li in range(5,18):
        a=accs[li]
        mu=a['mean']/max(a['n'],1)
        mun=mu/mu.norm().clamp_min(1e-6)
        cos_mean=float(cu@mun)
        S=torch.cat(a['samples'])
        Sc=S-S.mean(0,keepdim=True)
        try:
            _,_,V=torch.linalg.svd(Sc[:2000].float(),
                                   full_matrices=False)
            cos_pc1=float(abs(cu@V[0]))
        except Exception:
            cos_pc1=float('nan')
        avg_norm=a['norm']/max(a['n'],1)
        share=float(const.norm())/max(avg_norm,1e-6)
        nullc=sum(abs(float(r@mun)) for r in rnds)/len(rnds)
        rows.append({'layer':li,'cos_mean_dir':round(cos_mean,4),
                     'cos_pc1':round(cos_pc1,4),
                     'residual_norm':round(avg_norm,1),
                     'const_norm_share':round(share,4),
                     'null_cos':round(nullc,4)})
        print(f"L{li}: cos(mean dir) {cos_mean:+.3f} | cos(PC1) "
              f"{cos_pc1:.3f} | ||resid|| {avg_norm:.0f} | "
              f"const/resid {share:.3f} | null {nullc:.3f}",
              flush=True)
    hi=[r for r in rows if r['layer']>=6]
    frac=sum(1 for r in hi if abs(r['cos_mean_dir'])>=0.5)/len(hi)
    nullm=sum(r['null_cos'] for r in hi)/len(hi)
    meanc=sum(abs(r['cos_mean_dir']) for r in hi)/len(hi)
    pa=frac>=0.5
    pb=meanc>=10*max(nullm,1e-6)
    out={'layers':rows,'const_norm':round(float(const.norm()),1),
         'frac_layers_cos_ge_0.5':round(frac,3),
         'mean_abs_cos':round(meanc,4),'mean_null':round(nullm,4),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f"majority-aligned: {frac:.2f} | mean |cos| {meanc:.3f} "
          f"vs null {nullm:.4f}")
    for nm,v in (('a','|cos| >=0.5 at a majority of layers 6-17'),
                 ('b','>=10x the random null'),
                 ('c','norm shares reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
