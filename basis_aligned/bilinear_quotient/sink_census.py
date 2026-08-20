"""SINK CENSUS -- 451: head 5.7 reads position 0 for 99.8% of
queries and its constant is the residual stream's centre (449).
Is it alone? Sweep all 162 heads for position-0 locking: the
fraction of query positions whose top read is position 0, plus
each head's deletion cost from the full map (429/head_cost_map).
This is an architecture-level census, not a per-leaf claim.
REGISTERED PREDICTIONS:
  (a) SINKS ARE A CLASS: at least 5 heads read position 0 for
      more than 50% of queries;
  (b) COST LINK: among heads with sink fraction > 0.5, the median
      deletion cost exceeds the median for non-sink heads;
  (c) 5.7 is the most extreme sink in the model (highest
      fraction, or within 0.01 of the highest)."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sink_census_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    hits={f'{lj}.{h}':[0,0] for lj in range(18) for h in range(9)}
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        B=4; cap={}
        hs=[m.transformer.h[lj].attn.register_forward_pre_hook(
            (lambda lj: lambda mo_,a_: cap.__setitem__(
                lj,a_[0]))(lj)) for lj in range(18)]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        tril=torch.tril(torch.ones(T,T,device=DEV))
        for lj in range(18):
            at=m.transformer.h[lj].attn; X=cap[lj]
            cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
            def rr(w):
                return are(F.rms_norm(w(X).view(B,T,9,128),
                           (128,)),cos,sin)
            qf,kf=rr(at.c_q),rr(at.c_k)
            q2,k2=rr(at.c_q2),rr(at.c_k2)
            sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                            kf.float())/128
            sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                             k2.float())/128
            pat=((sc*sc2)*tril).abs()
            am=pat[:,:,8:,:].argmax(-1)      # B,9,T-8
            for h9 in range(9):
                k=f'{lj}.{h9}'
                hits[k][0]+=int((am[:,h9]==0).sum())
                hits[k][1]+=am[:,h9].numel()
        print(f'batch {i} done',flush=True)
    frac={k:round(v[0]/max(v[1],1),4) for k,v in hits.items()}
    cost=json.load(open(PT+'head_cost_map_results.json'))['heads']
    sinks=[k for k,v in frac.items() if v>0.5]
    def med(xs):
        xs=sorted(xs); return xs[len(xs)//2] if xs else None
    ms=med([cost[k]['dce_all'] for k in sinks if k in cost])
    mn=med([cost[k]['dce_all'] for k in frac
            if frac[k]<=0.5 and k in cost])
    top=sorted(frac,key=frac.get,reverse=True)[:10]
    pa=len(sinks)>=5
    pb=(ms is not None and mn is not None and ms>mn)
    pc=(frac['5.7']>=frac[top[0]]-0.01)
    out={'sink_fraction':frac,'sinks_over_half':sorted(sinks),
         'n_sinks':len(sinks),'median_cost_sinks':ms,
         'median_cost_nonsinks':mn,
         'top10':[(k,frac[k],cost.get(k,{}).get('dce_all'))
                  for k in top],
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f'{len(sinks)} heads read position 0 >50% of the time')
    print('top 10 (head, sink fraction, deletion cost):')
    for k in top: print(f'   {k}: {frac[k]:.3f}  '
                        f'{cost.get(k,{}).get("dce_all")}')
    print(f'median deletion cost: sinks {ms} vs non-sinks {mn}')
    for nm,v in (('a','>=5 heads are sinks'),
                 ('b','sinks cost more to delete'),
                 ('c','5.7 is the most extreme')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
