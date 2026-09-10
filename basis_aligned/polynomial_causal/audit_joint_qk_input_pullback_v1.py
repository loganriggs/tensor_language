"""Pull fixed joint QK V2 feature readers into a common raw-input quadratic space.

Pre-run bars are on the board: exact bridge 1e-10, all overlaps <=.10 versus
at least one head with both role overlaps >=.50. Descriptive geometry only.
"""
import json, hashlib, time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def gram(basis,q1,q2):
    width=q1.shape[0];w=basis.T.reshape(-1,width,width)
    a,b,c=q1@q1.T,q2@q2.T,q1@q2.T
    mapped=.5*(a@w@b+c@w.transpose(-1,-2)@c)
    g=basis.T@mapped.flatten(1).T
    return (g+g.T)/2

def whiten(g):
    e,v=torch.linalg.eigh(g);keep=e>e[-1]*1e-10
    return v[:,keep]/e[keep].sqrt(),int(keep.sum())

def compare(g,na):
    ga,gb,cross=g[:na,:na],g[na:,na:],g[:na,na:]
    wa,ra=whiten(ga);wb,rb=whiten(gb)
    cos=torch.linalg.svdvals(wa.T@cross@wb)
    return dict(rank_A=ra,rank_B=rb,normalized_overlap=float(cos.square().sum()/min(ra,rb)),
        largest_principal_cosine=float(cos[0]),principal_cosines=cos.tolist())

def main():
    tic=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(9114951)
    q1,q2=torch.randn(4,9,dtype=torch.float64),torch.randn(4,9,dtype=torch.float64)
    basis=torch.randn(16,7,dtype=torch.float64)
    w=basis.T.reshape(-1,4,4);raw=q1.T@w@q2;h=(raw+raw.transpose(-1,-2))/2
    direct=h.flatten(1)@h.flatten(1).T;implicit=gram(basis,q1,q2)
    fixture=float((direct-implicit).norm()/direct.norm());assert fixture<=1e-10
    ap=P/'CORRELATIVE_JOINT_QK_SUBSPACES_V2_BASES.pt';receipt=json.loads((P/'CORRELATIVE_JOINT_QK_SUBSPACES_V2_RESULT.json').read_text())
    assert hashlib.sha256(ap.read_bytes()).hexdigest()==receipt['artifact_sha256']
    spaces=torch.load(ap,map_location='cpu',weights_only=True);sd=torch.load(CK,map_location='cpu',mmap=True,weights_only=True)
    reports={};native_bridge=0.
    for unit,aroles in spaces['A'].items():
        _,ll,_,hh=unit.split(':');layer,head=int(ll),int(hh);reports[unit]={}
        for role,k in [('query','q'),('key','k')]:
            q1=sd[f'transformer.h.{layer}.attn.c_{k}.weight'][head*128:(head+1)*128].double()
            q2=sd[f'transformer.h.{layer}.attn.c_{k}2.weight'][head*128:(head+1)*128].double()
            ba,bb=aroles[role],spaces['B'][unit][role];both=torch.cat([ba,bb],1)
            g=gram(both,q1,q2);r=compare(g,ba.shape[1])
            r['product_coordinate_overlap']=receipt['overlap'][unit][role]['normalized_overlap']
            r['overlap_change']=r['normalized_overlap']-r['product_coordinate_overlap']
            # Direct trained-weight bridge for first head, both roles and two families.
            if len(reports)==1:
                selected=torch.stack([ba[:,0],bb[:,0]],1)
                raw=q1.T@selected.T.reshape(2,128,128)@q2;h=(raw+raw.transpose(-1,-2))/2
                dg=h.flatten(1)@h.flatten(1).T;ig=gram(selected,q1,q2)
                native_bridge=max(native_bridge,float((dg-ig).norm()/dg.norm()))
            reports[unit][role]=r
    valid=fixture<=1e-10 and native_bridge<=1e-10 and all(0<=r['normalized_overlap']<=1+1e-8 and r['largest_principal_cosine']<=1+1e-8 for roles in reports.values() for r in roles.values())
    out=dict(schema='joint.qk.input.pullback.v1',predictions={'pred_a_instrument':valid,
        'pred_b_all_input_spans_mostly_disjoint':bool(valid and all(r['normalized_overlap']<=.10 for roles in reports.values() for r in roles.values())),
        'pred_c_shared_head_in_both_roles':bool(valid and any(all(r['normalized_overlap']>=.50 for r in roles.values()) for roles in reports.values()))},
        reports=reports,fixture_relative_error=fixture,native_bridge_relative_error=native_bridge,native_forwards=0,
        cpu_seconds=time.perf_counter()-tic,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fixed learned subspaces pulled through both actual score matrices. Per-head, per-role raw quadratic numerator metric. Common normalization denominators within each comparison preserve exact function identities, but angles are not activation-weighted or behavioral. No refit, raw-state realizability, selective causal, or OOD claim.')
    target=P/'JOINT_QK_INPUT_PULLBACK_V1_AUDIT.json'
    with target.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k!='reports'},indent=2))
    for role in ['query','key']:
        vals=[r[role]['normalized_overlap'] for r in reports.values()]
        print(role,dict(min=min(vals),max=max(vals),mean=sum(vals)/len(vals)))

if __name__=='__main__':main()
