"""R30 READ SEMANTICS -- 413: r.3.0's heads compare m0|m0 (411)
but the ladder instrument is content-weak at layer 16 (412/413).
Ask the mechanism question directly at the TOKEN level: where do
16.8/16.2's top reads LAND? If they score coincidences on the
identity code, top reads should sit on tokens IDENTICAL to (or
adjacent to) the query token far above chance. Measured at
r.3.0-style member positions (old-tree circuit; generic every-4th
position used as the base set, with the seen-before split).
For each top read: is t_key == t_query? == t_{query-1}? in the
query's history? offset histogram.
REGISTERED PREDICTIONS:
  (a) IDENTITY READS: >=40% of top reads at seen-before query
      positions have t_key == t_query, vs <10% for a
      frequency-matched random-read null;
  (b) at FRESH query positions (token's first occurrence,
      self-coincidence impossible beyond q itself), the modal
      read is LOCAL (|offset| <= 2) -- the heads fall back to
      neighborhood reads; report either way;
  (c) the two heads agree (same dominant read class in (a)/(b))."""
import json, sys, time, torch
import torch.nn.functional as F
import census_lib as cl
from bilin18_joint_removal import m, DEV
D=1152; T=256
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'r30_read_semantics_results.json'
HEADS=[(16,8),(16,2)]
NR=16

@torch.no_grad()
def main():
    t0=time.time()
    ROWS=cl.rows()[:NR]
    are=sys.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    st={f'{li}.{hd}':{'seen_same':0,'seen_prev':0,'seen_hist':0,
                      'seen_n':0,'fresh_local':0,'fresh_n':0,
                      'null_same':0,'null_n':0,'offsets':{}}
        for li,hd in HEADS}
    g=torch.Generator().manual_seed(7)
    for i in range(0,NR,4):
        bb=ROWS[i:i+4,:257].to(DEV); idx=bb[:,:-1].contiguous()
        cap={}
        hs=[m.transformer.h[li].attn.register_forward_pre_hook(
            (lambda li: lambda mo_,args: cap.__setitem__(
                li,args[0]))(li)) for li,_ in HEADS]
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
        for h in hs: h.remove()
        for li,hd in HEADS:
            at=m.transformer.h[li].attn
            X=cap[li]
            cos,sin=at.rotary(at.c_q(X).view(4,T,9,128))
            qf=F.rms_norm(at.c_q(X).view(4,T,9,128),(128,))[:,:,hd]
            kf=F.rms_norm(at.c_k(X).view(4,T,9,128),(128,))[:,:,hd]
            q2=F.rms_norm(at.c_q2(X).view(4,T,9,128),(128,))[:,:,hd]
            k2=F.rms_norm(at.c_k2(X).view(4,T,9,128),(128,))[:,:,hd]
            qf=are(qf[:,:,None],cos,sin)[:,:,0]
            kf=are(kf[:,:,None],cos,sin)[:,:,0]
            q2=are(q2[:,:,None],cos,sin)[:,:,0]
            k2=are(k2[:,:,None],cos,sin)[:,:,0]
            pat=(torch.einsum('bqd,bkd->bqk',qf.float(),kf.float())
                 *torch.einsum('bqd,bkd->bqk',q2.float(),
                               k2.float())) \
                *torch.tril(torch.ones(T,T,device=DEV))
            s=st[f'{li}.{hd}']
            for b in range(4):
                toks=ROWS[i+b,:T].tolist()
                seenpos={}
                for q in range(4,T,4):
                    hist=set(toks[:q])
                    isseen=toks[q] in hist
                    k=int(pat[b,q,:q].abs().argmax())
                    off=k-q
                    s['offsets'][off]=s['offsets'].get(off,0)+1
                    kr=int(torch.randint(0,q,(1,),generator=g))
                    s['null_same']+=int(toks[kr]==toks[q])
                    s['null_n']+=1
                    if isseen:
                        s['seen_n']+=1
                        s['seen_same']+=int(toks[k]==toks[q])
                        s['seen_prev']+=int(k>0 and
                                            toks[k-1]==toks[q])
                        s['seen_hist']+=int(toks[k] in hist)
                    else:
                        s['fresh_n']+=1
                        s['fresh_local']+=int(abs(off)<=2)
        print(f'batch {i} done',flush=True)
    outj={}
    for k9,s in st.items():
        outj[k9]={
            'seen_same':round(s['seen_same']/max(s['seen_n'],1),3),
            'seen_prev':round(s['seen_prev']/max(s['seen_n'],1),3),
            'seen_hist':round(s['seen_hist']/max(s['seen_n'],1),3),
            'fresh_local':round(s['fresh_local']
                                /max(s['fresh_n'],1),3),
            'null_same':round(s['null_same']/max(s['null_n'],1),3),
            'n_seen':s['seen_n'],'n_fresh':s['fresh_n'],
            'top_offsets':sorted(s['offsets'].items(),
                                 key=lambda kv:-kv[1])[:6]}
        print(f"{k9}: {outj[k9]}",flush=True)
    pa=all(outj[k]['seen_same']>=0.40 and outj[k]['null_same']<0.10
           for k in outj)
    pb=all(outj[k]['fresh_local']>=0.5 for k in outj)
    cls=lambda o:('same' if o['seen_same']>=0.4 else 'other',
                  'local' if o['fresh_local']>=0.5 else 'other')
    pc=len({cls(outj[k]) for k in outj})==1
    out={'heads':outj,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    for nm,v in (('a','seen-same >=40% vs null <10%'),
                 ('b','fresh reads local (report)'),
                 ('c','heads agree')):
        print(f"({nm}) {v}: "
              f"{'HELD' if out['pred_'+nm] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
