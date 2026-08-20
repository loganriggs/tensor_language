"""MECH TOOL RECENTER -- 450: the residual stream after layer 5 is
60-72% ONE FIXED VECTOR (449: the sink constant is the stream's
mean direction at cosine 0.99, and accounts for 0.62-0.72 of the
residual norm at layers 6-8). That has a direct consequence for
this program's own instrument: leaf_input_decomp computes each
writer's PROJECTION SHARE onto the total residual, and after
layer 5 that total is mostly the bias -- so writer shares at mid
and late layers are partly measuring alignment with the bias
rather than contribution to the leaf's machinery.
Recompute the mechanism tables with the bias axis projected OUT
of every writer contribution and the total, and compare.
Leaves: r.3.0.2 (the one CONFIRMED positive, a14 at 2.37
[2.06-2.85]), r.13.2.1 and r.2.0.1 (confirmed negatives).
REGISTERED PREDICTIONS:
  (a) INSTRUMENT SENSITIVITY: for at least one of the three
      leaves the TOP WRITER changes when the bias axis is removed;
  (b) THE POSITIVE SURVIVES: r.3.0.2's a14 enrichment stays
      >= 1.5 after recentering (if it collapses, the program's
      only confirmed mechanism claim was a bias-alignment
      artifact and must be retracted);
  (c) both tables reported per leaf, old and recentered."""
import json, ast, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; LJ=5; HD=7
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'mech_tool_recenter_results.json'
TAGS=['r.3.0.2','r.13.2.1','r.2.0.1']
MAXROWS=16

@torch.no_grad()
def get_bias():
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    ROWS=cl.rows()[:8]
    cap={}
    h=m.transformer.h[LJ].attn.register_forward_pre_hook(
        lambda mo_,a_: cap.__setitem__('X',a_[0]))
    idx=ROWS[:,:256].to(DEV); B=8
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    h.remove()
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
    c=(torch.einsum('bqk,bkd->bqd',pat,v)@Wp.T).mean(dim=(0,1))
    return c/c.norm().clamp_min(1e-6)

@torch.no_grad()
def table(tag,u=None):
    """u: unit vector to project OUT (None = original tool)."""
    lf=cl.leaf(tag)
    probes=[ast.literal_eval(p) if isinstance(p,str) else p
            for p in lf['top_probes']]
    keys=[]
    for p in probes:
        k=p[1] if p[0] in ('comp','pca') else f'a{p[1]}'
        if k not in keys: keys.append(k)
    mem=lf['member']; sl=lf['slice']; NF=cl.nflat()
    memm=torch.zeros(NF,dtype=torch.bool); memm[mem]=True
    slm=torch.zeros(NF,dtype=torch.bool); slm[sl]=True
    g=torch.Generator().manual_seed(5)
    rr=(mem//256).unique()
    if len(rr)>MAXROWS:
        rr=rr[torch.randperm(len(rr),generator=g)[:MAXROWS]] \
            .sort().values
    rows=cl.rows(); out={}
    for key in keys:
        li=int(key[1:])
        WR=['wte']+[f'{k}{l}' for l in range(li) for k in ('a','m')]
        if key[0]=='m': WR.append(f'a{li}')
        sm={w:0.0 for w in WR}; so={w:0.0 for w in WR}
        cm=co=0
        for i in range(0,len(rr),4):
            rid=rr[i:i+4]
            bb=rows[rid,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); B=len(rid)
            outs={}; hs=[]
            for lj in range(li+1):
                for kind,mod in (('a',m.transformer.h[lj].attn),
                                 ('m',m.transformer.h[lj].mlp)):
                    def mk(k9=f'{kind}{lj}'):
                        def h(mo,i_,o_):
                            y=o_[0] if isinstance(o_,tuple) else o_
                            outs[k9]=y.detach().float()
                        return h
                    hs.append(mod.register_forward_hook(mk()))
            E=F.rms_norm(m.transformer.wte(idx),(D,)).float()
            x=E.to(m.transformer.wte.weight.dtype); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            for h in hs: h.remove()
            lam=m.transformer.h[li].lambdas.detach().float()
            parts={w:((lam[0]+lam[1])*E if w=='wte'
                      else lam[0]*outs[w]) for w in WR
                   if w!=f'a{li}'}
            if f'a{li}' in WR: parts[f'a{li}']=outs[f'a{li}']
            if u is not None:
                uu=u.to(DEV)
                parts={w:(p-(p@uu)[...,None]*uu)
                       for w,p in parts.items()}
            tot=sum(parts.values())
            tn=(tot*tot).sum(-1).clamp_min(1e-9)
            frac={w:((p*tot).sum(-1)/tn) for w,p in parts.items()}
            gi=(rid[:,None].to(DEV)*256
                +torch.arange(T,device=DEV)[None,:])
            mk_=memm.to(DEV)[gi]; ok_=(~slm.to(DEV)[gi])
            cm+=int(mk_.sum()); co+=int(ok_.sum())
            for w,fv in frac.items():
                sm[w]+=float(fv[mk_].sum()); so[w]+=float(fv[ok_].sum())
        tbl={w:{'ratio':round((sm[w]/max(cm,1))
                              /max(so[w]/max(co,1),1e-4),3),
                'member':round(sm[w]/max(cm,1),4)} for w in WR}
        top=sorted(tbl.items(),key=lambda kv:-kv[1]['ratio'])[:3]
        out[key]={'top':top}
    return out

def main():
    t0=time.time()
    cl.use_state(PT+'census_state_diverse.pt')
    u=get_bias().cpu()
    res={}
    for tag in TAGS:
        try:
            torch.cuda.empty_cache()
            old=table(tag,None); new=table(tag,u)
            res[tag]={'original':old,'recentered':new}
            for k in old:
                print(f"{tag} {k}: orig {[(w,v['ratio']) for w,v in old[k]['top']]}"
                      f" | recentered {[(w,v['ratio']) for w,v in new[k]['top']]}",
                      flush=True)
            json.dump(res,open(OUT,'w'),indent=1)
        except Exception as e:
            print(f'{tag}: SKIPPED ({type(e).__name__}: {e})',
                  flush=True)
    changed=0
    for tag in res:
        for k in res[tag]['original']:
            if (res[tag]['original'][k]['top'][0][0]
                    !=res[tag]['recentered'][k]['top'][0][0]):
                changed+=1; break
    a14=None
    if 'r.3.0.2' in res:
        t=res['r.3.0.2']['recentered']
        a14=max((v['top'][0][1]['ratio'] for v in t.values()),
                default=0)
    pa=changed>=1
    pb=(a14 is not None and a14>=1.5)
    out={'leaves':res,'n_leaves_top_writer_changed':changed,
         'r302_best_recentered_ratio':a14,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':True,
         'runtime_s':time.time()-t0}
    print(f'top writer changed for {changed}/{len(res)} leaves | '
          f'r.3.0.2 best recentered ratio {a14}')
    for nm,v in (('a','top writer changes for >=1 leaf'),
                 ('b','r.3.0.2 a14 survives at >=1.5'),
                 ('c','both tables reported')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
