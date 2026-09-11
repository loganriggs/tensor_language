"""A square pullback identity<=1e-10 B quotation path matches old150>=.95 bothjointseeds.
All256oldreaders and both fitted/native writers checked; no behavioral novelty claim.
"""
import json,hashlib
from pathlib import Path
import torch
from sparse_path_coefficient_gram_v1 import gram
from sparse_path_program_v1 import run


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);p=Path(__file__).parent;out=p/'SPARSE_PATH_PRIOR_SQUARE_ALIAS_V1.json';assert not out.exists()
    result=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_RESULT.json').read_text());ap=p/'COUPLED_SPARSE_PATH_CONTINUE_V1_PROGRAMS.pt';assert digest(ap)==result['artifact_sha256']
    programs=torch.load(ap,weights_only=True,map_location='cpu')['programs'];joint=[a for a in programs if a['mode']=='joint']
    oldpath=p/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt';old=torch.load(oldpath,weights_only=True,map_location='cpu')
    binding=json.loads((p/'COUPLED_SPARSE_PATH_CONTINUE_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('/pytorch_model.bin'));assert digest(ck)==binding[ck]
    state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].double();metric=u.T@u
    raw=old['model']['a'].double();norm=raw.norm(dim=1);readers=raw/norm[:,None];n=len(readers);assert n==256
    o=state['transformer.h.17.attn.c_proj.weight'].double();scale=joint[0]['source_scale'];pulled=readers@o/scale;pn=pulled.norm(dim=1)
    bank=torch.stack([readers.T,(pulled/pn[:,None]).T]);N=2*n
    def index(i,j):return i*N-i*(i-1)//2+j-i
    ids=torch.arange(n);support=torch.cat([index(ids,ids),index(ids,ids+n),index(ids+n,ids+n)])
    l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
    native=d@((l@readers.T)*(r@readers.T));fitted=old['writer'].double()*norm.square()
    references=[];errors=[];torch.manual_seed(11701);x,y=torch.randn(5,1152),torch.randn(5,1152)
    for label,w in [('fitted',fitted),('native_projection',native)]:
        writer=torch.cat([w,w*(2**.5*pn),w*pn.square()],1)
        ref=dict(mode='joint',bank=bank,support=support,physical_writer=writer,source_scale=scale);references.append((label,ref))
        expected=((x+y@o.T)@readers.T).square()@w.T
        actual=run(ref,x,y,torch.ones(len(x)))['numerator'];errors.append(float((actual-expected).norm()/expected.norm()))
    matches=[];quote=[]
    for a in joint:
        en=gram(a,a,metric)[0].diag()
        for label,ref in references:
            g,h,wg=gram(a,ref,metric);ren=gram(ref,ref,metric)[0].diag();cos=g/(en[:,None]*ren[None,:]).sqrt()
            best,indexes=cos.max(1);rows=[]
            for e in range(len(best)):
                j=int(indexes[e]);rows.append(dict(edge=e,old_square=j%n,old_piece=['rr','ra','aa'][j//n],function_cosine=float(best[e]),input_feature_cosine=float(h[e,j]),
                    output_cosine=float(wg[e,j]/(en[e]*ren[j]).sqrt()),component_norm_ratio=float((en[e]/ren[j]).sqrt())))
            matches.append(dict(seed=a['seed'],reference=label,matches_at_point95=int((best>=.95).sum()),rows=rows))
            q=cos[:,2*n+150];edge=int(q.argmax());quote.append(dict(seed=a['seed'],reference=label,edge=edge,old_square=150,piece='aa',cosine=float(q[edge])))
    result=dict(pred_a=max(errors)<=1e-10,pred_b=all(q['cosine']>=.95 for q in quote if q['reference']=='native_projection'),pullback_errors=errors,
        quote_aliases=quote,matches=matches,old_square_bank_sha256=digest(oldpath),source_program_sha256=digest(ap),
        scope='Coefficient-shape alias of source-pair pieces of prior squares, not whole-square identity or matching amplitudes. Old quote behavioral/gating failures remain; no new circuit count.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='matches'},indent=2));print(json.dumps([{k:v for k,v in m.items() if k!='rows'} for m in matches],indent=2));assert result['pred_a']

if __name__=='__main__':main()
