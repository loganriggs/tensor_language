"""LEVEL-2 OF THE HIERARCHY (rung 226): input-class-indexed selection at m2.

CONVENTION (S2135): per-position dCE = CE(m2 replaced, all else real) - CE(real model); LOWER IS BETTER.
S2323: raw token identity fades by block 2 (TOKID/STATIC 0.771). The user's hierarchy: a second, coarser
index COMPUTED FROM CONTEXT. Here: 10 input-classes of the CURRENT token, prefix-only (digit/bracket-
close/newline/sentence-end/comma/name/repeat/subword/seen-before/other - the census taxonomy's rules
applied to the input position; the census TARGET-class is NOT used, it would leak the label). Class rows
are fit on FW windows (disjoint from census). Selection stays a static tensor indexed by (token id, class).
ARMS (m2; anchors from the S2323 receipt: STATIC 0.2200, TOKID 0.1697, ORACLE 0.0353):
  CLSID = class-indexed table alone (10 rows, fit-data mean scores).
  HIER  = two-level blend: s = s_tok(id) + s_cls(class), top-1152 of the static sum.
REGISTERED PREDICTIONS:
  (a) CLASS BEATS FADED TOKEN AT DEPTH: census|dCE|(CLSID) <= 0.9 x TOKID.
  (b) THE HIERARCHY COMPOUNDS: HIER <= 0.8 x TOKID.
  (c) CONSISTENCY: ORACLE <= HIER <= STATIC; arms non-inert.
NULL: the 10-class index is too coarse to select units (CLSID >= 0.9 x STATIC and HIER >= 0.95 x TOKID).
PRICE: 10 class rows + the S2321 token table + 2 census passes. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['tokid_m2_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: class-indexed selection at m2'); raise SystemExit(0)
import torch
import torch.nn.functional as F
import tiktoken
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'clsid_m2_results.json'
MOD=2
enc=tiktoken.get_encoding('gpt2')

def clsrow(toks):
    out=[]; seen=set()
    for pos in range(256):
        t=toks[pos]; p=toks[pos-1] if pos>0 else t
        tg=enc.decode([t]); s=tg.strip(); pv=enc.decode([p])
        if s.isdigit() and not tg.startswith(' '): k=0
        elif s in (')',']') and any(b in enc.decode(toks[max(0,pos-60):pos]) for b in ('(','[')): k=1
        elif '\n' in tg: k=2
        elif tg in ('.','!','?'): k=3
        elif tg==',': k=4
        elif tg.startswith(' ') and s[:1].isupper() and (pv.strip()[:1].isupper() if pv.strip() else False): k=5
        elif pos>0 and t==p: k=6
        elif (not tg.startswith(' ')) and s.isalpha(): k=7
        elif t in seen: k=8
        else: k=9
        out.append(k); seen.add(t)
    return out

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    ANC=json.load(open(PT+'tokid_m2_results.json'))['arms']
    mlp=m.transformer.h[MOD].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0); K=1152
    W=m.transformer.wte.weight.detach().float(); Vw=W.shape[0]
    # level-1 token score table (embedding-derived, as S2321)
    STOK=torch.empty(Vw,4608,dtype=torch.float16,device=DEV)
    for v0 in range(0,Vw,4096):
        xe=F.rms_norm(W[v0:v0+4096],(D,))
        STOK[v0:v0+4096]=(((xe@L.T)*(xe@Rw.T)).abs()*dn).half()
    # level-2 class rows from fit windows
    cap={}
    hp=mlp.register_forward_pre_hook(lambda mo,i_: cap.__setitem__('x',i_[0].detach().float()))
    SC=torch.zeros(10,4608,device=DEV); CC=torch.zeros(10,device=DEV)
    for i in range(0,512,4):
        ix=FW[i:i+4,:256].to(DEV)
        cls=torch.tensor([clsrow(FW[r,:257].tolist()) for r in range(i,i+4)],device=DEV)
        with torch.no_grad():
            x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
            for li in range(MOD+1): x,v1=m.transformer.h[li](x,v1,x0)
            u=(cap['x']@L.T)*(cap['x']@Rw.T)
            s=(u.abs()*dn).reshape(-1,4608)
        fid=cls.reshape(-1)
        SC.index_add_(0,fid,s); CC.index_add_(0,fid,torch.ones_like(fid,dtype=torch.float))
    hp.remove()
    SCLS=SC/CC.clamp(min=1)[:,None]
    CIDX=SCLS.topk(K,dim=-1).indices
    print(f'tables built; class counts {CC.int().tolist()} ({time.time()-T00:.0f}s)',flush=True)
    CLSMAP=torch.stack([torch.tensor(clsrow(ROWS[r,:257].tolist())) for r in range(ROWS.shape[0])]).to(DEV)
    print(f'census classified ({time.time()-T00:.0f}s)',flush=True)
    CUR={}
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            b=ROWS[i:i+4,:257].to(DEV)
            ix=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            CUR['ids']=ix; CUR['cls']=CLSMAP[i:i+4]
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    def hk_cls(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=CIDX[CUR['cls']]
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    def hk_hier(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        sc=STOK[CUR['ids']].float()+SCLS[CUR['cls']]
        idx=sc.topk(K,dim=-1).indices
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    res={}
    for nm,hk in (('CLSID',hk_cls),('HIER',hk_hier)):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} inert')
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f}",flush=True)
    pa=res['CLSID']['aggabs']<=0.9*ANC['TOKID']['aggabs']
    pb=res['HIER']['aggabs']<=0.8*ANC['TOKID']['aggabs']
    pc=(ANC['ORACLE']['aggabs']<=res['HIER']['aggabs']<=ANC['STATIC']['aggabs'])
    out={'arms':res,'anchors':{k:v['aggabs'] for k,v in ANC.items()},
         'pred_a_class_beats_token':bool(pa),'pred_b_hier_compounds':bool(pb),'pred_c_consistency':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f"(a) CLSID {res['CLSID']['aggabs']:.4f} <= 0.9 x TOKID {ANC['TOKID']['aggabs']:.4f}: {'HELD' if pa else 'FAILED'}")
    print(f"(b) HIER {res['HIER']['aggabs']:.4f} <= 0.8 x TOKID: {'HELD' if pb else 'FAILED'}")
    print(f"(c) ORACLE <= HIER <= STATIC: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
