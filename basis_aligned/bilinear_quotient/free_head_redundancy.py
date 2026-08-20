"""FREE HEAD REDUNDANCY -- 465: the complete cost map (429) found
39 of 162 heads cost nothing or help when deleted individually,
spread evenly across depth. The sink pair (454) then showed that
"free" can mean "covered by a partner": head 5.2 costs 0.015 alone
but deleting it with 5.7 costs 0.28 more than the sum. If that
generalises, the free set is not spare capacity -- it is a
redundancy pool, and its members look free only because their
partners are intact.
Arms (mean-ablation, the program's operator-C form):
  all 39 free heads jointly
  random subsets of 10 and 20 of them (seeded, 3 draws each)
  a control set of 39 heads drawn from the COSTLY half
REGISTERED PREDICTIONS:
  (a) REDUNDANCY: deleting all 39 jointly costs >= 0.30 nats,
      although their individual costs sum to <= 0;
  (b) SUPERLINEAR: cost(39) >= 3x the mean cost(20);
  (c) SANITY: the costly-set control costs more than the free
      set at the same size."""
import json, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'free_head_redundancy_results.json'
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    cost=json.load(open(PT+'head_cost_map_results.json'))['heads']
    free=sorted([k for k,v in cost.items() if v['dce_all']<=0])
    costly=sorted(cost,key=lambda k:-cost[k]['dce_all'])[:39]
    sum_free=sum(cost[k]['dce_all'] for k in free)
    print(f'{len(free)} free heads, individual sum '
          f'{sum_free:+.4f}',flush=True)
    ROWS=cl.rows()[:NR]
    are=__import__('sys').modules[
        type(m.transformer.h[0].attn).__module__].apply_rotary_emb
    def run(heads):
        byl={}
        for k in heads:
            l,h=k.split('.'); byl.setdefault(int(l),[]).append(int(h))
        tot=0.0; n=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            for lj,hds in byl.items():
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,hds=hds):
                    y,v1r=o_
                    X=args[0]
                    v1=args[1] if args[1] is not None else v1r
                    v=at.c_v(X).view(B,T,9,128)
                    vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
                    cos,sin=at.rotary(at.c_q(X).view(B,T,9,128))
                    def r2(w):
                        return are(F.rms_norm(
                            w(X).view(B,T,9,128),(128,)),cos,sin)
                    qf,kf=r2(at.c_q),r2(at.c_k)
                    q2,k2=r2(at.c_q2),r2(at.c_k2)
                    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),
                                    kf.float())/128
                    sc2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),
                                     k2.float())/128
                    pat=(sc*sc2)*torch.tril(
                        torch.ones(T,T,device=DEV))
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                    for h in hds: z[:,h]=0
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').mean().item()
            for h in hs: h.remove()
            tot+=ce; n+=1
        return tot/max(n,1)
    base=run([])
    res={'baseline':round(base,4),'n_free':len(free),
         'individual_sum_free':round(sum_free,4)}
    res['all_free']=round(run(free)-base,4)
    print(f"all {len(free)} free: {res['all_free']:+.4f}",flush=True)
    g=torch.Generator().manual_seed(23)
    for sz in (10,20):
        vals=[]
        for _ in range(3):
            pick=[free[i] for i in torch.randperm(
                len(free),generator=g)[:sz].tolist()]
            vals.append(run(pick)-base)
        res[f'subset_{sz}']=[round(v,4) for v in vals]
        res[f'mean_{sz}']=round(sum(vals)/len(vals),4)
        print(f'subset {sz}: {res[f"subset_{sz}"]}',flush=True)
    res['costly_39']=round(run(costly)-base,4)
    print(f"costly 39 control: {res['costly_39']:+.4f}",flush=True)
    pa=(res['all_free']>=0.30 and sum_free<=0)
    pb=res['all_free']>=3*max(res['mean_20'],1e-6)
    pc=res['costly_39']>res['all_free']
    res.update({'pred_a':bool(pa),'pred_b':bool(pb),
                'pred_c':bool(pc),'runtime_s':time.time()-t0})
    for nm,v in (('a','39 individually-free heads jointly cost >=0.30'),
                 ('b','cost grows superlinearly with subset size'),
                 ('c','costly-set control is worse')):
        print(f"({nm}) {v}: "
              f"{'HELD' if res['pred_'+nm] else 'FAILED'}")
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
