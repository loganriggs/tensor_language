"""TOKEN-INDEXED SPARSITY AT m0 (rung 222): the selection tensor replaces the dynamic gate.

CONVENTION (S2135): per-position dCE = CE(m0 replaced, all else real) - CE(real model); LOWER IS BETTER.
USER DIRECTIVE (2026-08-31): the compiled program should stay a TENSOR NETWORK - sparsity installed as
static structure (a sparse selection tensor, hierarchy, a DAG), not left as a runtime top-k nonlinearity;
and at layers fed by the embedding the conditioning variable is DISCRETE AND KNOWN: the token id itself.
Here the per-position active-unit set at m0 is precomputed PER VOCABULARY ID from the pure embedding
(x_e = rms_norm(wte(id)); score |u_e| * ||D_i||; top-1152 of 4608) - a static {0,1}^{V x 4608} selection
tensor with 1152 ones per row. No runtime scoring: the token id gathers its row.
ARMS (single-site m0):
  STATIC = one fixed top-1152 unit set (norm score) for all tokens.
  ORACLE = per-position dynamic top-1152 from the true activations (upper bound, S2304 instrument).
  TOKID  = the static token-indexed table.
REGISTERED PREDICTIONS:
  (a) TOKEN IDENTITY CARRIES THE SELECTION AT m0: census|dCE|(TOKID) <= 0.25 x STATIC.
  (b) NEAR-ORACLE: TOKID <= 2.0 x ORACLE.
  (c) CONSISTENCY: ORACLE <= STATIC; all arms non-inert.
NULL: context dominates even at block 0 (TOKID >= 0.8 x STATIC) - static tables need deeper conditioning.
PRICE (honest): the naive table is 50257 x 1152 int16 indices (~58M values); the known sparsification
paths (frequency-weighted rows, shared rows for rare tokens) are NOT claimed here. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['cp_topk1152_results.json','census_state_diverse.pt']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: token-indexed sparsity at m0'); raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'tokid_sparsity_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu(); CBASE=CN.base_ce().float().cpu()
    mlp=m.transformer.h[0].mlp
    L=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
    Dw=mlp.Down.weight.detach().float(); db=mlp.Down_bias.detach().float()
    dn=Dw.norm(dim=0); K=1152
    # token-indexed table from pure embeddings (wte may be padded past V)
    W=m.transformer.wte.weight.detach().float()
    Vw=W.shape[0]
    IDX=torch.empty(Vw,K,dtype=torch.int32,device=DEV)
    for v0 in range(0,Vw,4096):
        xe=F.rms_norm(W[v0:v0+4096],(D,))
        ue=(xe@L.T)*(xe@Rw.T)
        IDX[v0:v0+4096]=(ue.abs()*dn).topk(K,dim=-1).indices.int()
        del xe,ue
    print(f'token table built ({time.time()-T00:.0f}s)',flush=True)
    kp=(dn*L.norm(dim=1)*Rw.norm(dim=1)).argsort(descending=True)[:K]
    smask=torch.zeros(4608,device=DEV); smask[kp]=1.0
    CUR={}
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            b=ROWS[i:i+4,:257].to(DEV)
            ix=b[:,:-1].contiguous(); tg=b[:,1:].reshape(-1)
            CUR['ids']=ix
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(ix),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    def hk_static(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        return (((u*smask)@Dw.T)+db).to(o_.dtype)
    def hk_oracle(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=(u.abs()*dn).topk(K,dim=-1).indices
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    def hk_tokid(mo,i_,o_):
        x=i_[0].float()
        u=(x@L.T)*(x@Rw.T)
        idx=IDX[CUR['ids']].long()
        msk=torch.zeros_like(u).scatter_(-1,idx,1.0)
        return (((u*msk)@Dw.T)+db).to(o_.dtype)
    res={}
    for nm,hk in (('STATIC',hk_static),('ORACLE',hk_oracle),('TOKID',hk_tokid)):
        h=mlp.register_forward_hook(hk)
        cev=evalce()
        h.remove()
        d=cev-CBASE
        if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {nm} inert')
        res[nm]={'agg':round(float(d.mean()),4),'aggabs':round(float(d.abs().mean()),4)}
        print(f"  {nm}: agg {res[nm]['agg']:+.4f} | census|dCE| {res[nm]['aggabs']:.4f}",flush=True)
    ra=res['TOKID']['aggabs']/max(res['STATIC']['aggabs'],1e-9)
    rb=res['TOKID']['aggabs']/max(res['ORACLE']['aggabs'],1e-9)
    pa=ra<=0.25; pb=rb<=2.0; pc=res['ORACLE']['aggabs']<=res['STATIC']['aggabs']
    out={'arms':res,'ratio_tokid_static':round(ra,4),'ratio_tokid_oracle':round(rb,4),
         'pred_a_token_carries':bool(pa),'pred_b_near_oracle':bool(pb),'pred_c_consistency':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'ratios: TOKID/STATIC {ra:.3f} | TOKID/ORACLE {rb:.3f}')
    print(f"(a) TOKID <= 0.25 x STATIC ({ra:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) TOKID <= 2.0 x ORACLE ({rb:.3f}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) ORACLE <= STATIC: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

main()
