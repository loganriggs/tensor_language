"""INDUCTION WINDOW -- 485: a tension the sweeps have exposed and
that touches this program's flagship result. Deleting the nine
induction-band heads costs +0.601 at match positions (376). But
restricting their LAYERS to a 4-token read window costs almost
nothing at those same positions (484: layers 1,2,6,7,8 all
between -0.06 and -0.02, i.e. free or slightly helpful; layer 5's
+0.996 is the sink, since allowing position 0 drops it to +0.070).
Induction is defined by reading a DISTANT earlier occurrence. If
its heads cannot see past four tokens and nothing happens, then
either their value at match positions comes from LOCAL reads, or
the deletion cost measures something other than the match read.
Test the heads directly rather than their layers.
Arms, all scored at match positions:
  window_ind : the nine band heads restricted to 4 tokens
                (1.4, 2.5, 3.5, 3.8, 5.5, 6.5, 7.3, 8.3, 8.4)
  delete_ind : the same nine deleted (sanity against 376's 0.601)
  window_ctrl: nine RANDOM non-band heads restricted the same way
REGISTERED PREDICTIONS:
  (a) SANITY: deleting the nine costs >= 0.30 at match positions;
  (b) THE DECIDING BAR: windowing them costs >= 0.30 at match
      positions if their distant reads carry the function, or
      <= 0.10 if the function is local -- either outcome is a
      substantive finding about the induction claim and is
      recorded as such;
  (c) CONTROL: nine random non-band heads windowed cost less at
      match positions than the band heads windowed."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256; K=4
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'induction_window_results.json'
NR=16
IND=[(1,4),(2,5),(3,5),(3,8),(5,5),(6,5),(7,3),(8,3),(8,4)]

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    g=torch.Generator().manual_seed(17)
    ctrl=[]
    while len(ctrl)<9:
        lj=int(torch.randint(0,18,(1,),generator=g))
        hd=int(torch.randint(0,9,(1,),generator=g))
        if (lj,hd) not in IND and (lj,hd) not in ctrl:
            ctrl.append((lj,hd))
    print(f'control heads: {ctrl}',flush=True)
    def run(heads,mode):
        byl={}
        for lj,hd in heads: byl.setdefault(lj,[]).append(hd)
        tm=tn=0.0; nm_=nn_=0
        for i in range(0,NR,4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            B=4; hs=[]
            for lj,hds in byl.items():
                at=m.transformer.h[lj].attn
                def fh(mo_,args,o_,at=at,hds=hds,mode=mode):
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
                    tril=torch.tril(torch.ones(T,T,device=DEV))
                    ar=torch.arange(T,device=DEV)
                    win=tril*((ar[:,None]-ar[None,:])<K).float()
                    pat=(sc*sc2)*tril
                    patw=(sc*sc2)*win
                    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
                    zw=torch.einsum('bhqk,bkhd->bhqd',patw,
                                    vm.float())
                    for h in hds:
                        z[:,h]=0 if mode=='delete' else zw[:,h]
                    return (at.c_proj(z.transpose(1,2).contiguous()
                            .view(B,T,-1).to(X.dtype)),v1r)
                hs.append(at.register_forward_hook(fh))
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x
            v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))
                              /30)).float()
            ce=F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                               reduction='none').view(B,T).cpu()
            mk=torch.zeros(B,T,dtype=torch.bool)
            for b in range(B):
                toks=ROWS[i+b,:T].tolist(); last={}
                for q in range(T):
                    t=toks[q]
                    if t in last and last[t]+1<q and q>=8:
                        mk[b,q]=True
                    last[t]=q
            tm+=float(ce[mk].sum()); nm_+=int(mk.sum())
            tn+=float(ce[~mk].sum()); nn_+=int((~mk).sum())
            for h in hs: h.remove()
        return tm/max(nm_,1),tn/max(nn_,1)
    bm,bn=run([],'window')
    res={}
    for nm,(heads,mode) in {
            'window_ind':(IND,'window'),
            'delete_ind':(IND,'delete'),
            'window_ctrl':(ctrl,'window')}.items():
        pm,pn=run(heads,mode)
        res[nm]={'match':round(pm-bm,4),
                 'nonmatch':round(pn-bn,4)}
        print(f"{nm}: match {res[nm]['match']:+.4f} non-match "
              f"{res[nm]['nonmatch']:+.4f}",flush=True)
    pa=res['delete_ind']['match']>=0.30
    wi=res['window_ind']['match']
    pb=(wi>=0.30 or wi<=0.10)
    pc=res['window_ctrl']['match']<wi
    out={'baseline_match_ce':round(bm,4),'arms':res,
         'control_heads':ctrl,'verdict':(
             'distant reads carry it' if wi>=0.30
             else 'induction value at match is LOCAL'
             if wi<=0.10 else 'intermediate'),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc),
         'runtime_s':time.time()-t0}
    print(f"verdict: {out['verdict']}")
    for nm,v in (('a','deleting the nine costs >=0.30 at match'),
                 ('b','windowing them is decisive either way'),
                 ('c','control heads matter less')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
