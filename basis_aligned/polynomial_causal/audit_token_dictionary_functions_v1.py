"""Inspect frozen learned functions; exact quadratic spectra, exploratory token lists."""
import hashlib,json,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from quadratic_token_dictionary_v1 import embed
from audit_token_dictionary_debias_v1 import P,CK


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json').read_text())
    cache=Path(receipt['cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu')
    a,b=saved['codes'],saved['dictionary']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    x,root,basis,scale,mean=embed(u,l,r,d)
    tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    def label(i):return tokenizer.decode([i]) if i<tokenizer.get_vocab_size() else f'<extra output row {i}>'
    energy=a.square().sum(0)*b.square().sum(1)
    selected=energy.topk(8).indices
    rows=[]
    for j in selected.tolist():
        coeff=b[j]@basis
        q=(l.T*coeff)@r;q=(q+q.T)/2
        ev=torch.linalg.eigvalsh(q);sq=ev.square().sort(descending=True).values
        total=float(sq.sum());positive=float(ev[-1].clamp_min(0));negative=float(ev[0].clamp_max(0))
        # A real symmetrized product has at most one positive and one negative eigenvalue.
        product_capture=(positive**2+negative**2)/total
        token_lists={}
        for name,sign in [('positive',1),('negative',-1)]:
            indices=(sign*a[:,j]).topk(12).indices.tolist()
            token_lists[name]=[dict(id=i,text=label(i),coefficient=float(a[i,j])) for i in indices]
        rows.append(dict(index=j,active_tokens=int((a[:,j]!=0).sum()),
                         separate_component_energy_fraction=float(energy[j]/x.square().sum()),
                         function_norm_error=abs(total-float(b[j].square().sum())),
                         best_single_real_product_capture=product_capture,
                         best_rank8_signed_square_capture=float(sq[:8].sum()/sq.sum()),
                         signed_square_rank90=int(torch.searchsorted(sq.cumsum(0),.9*sq.sum()))+1,
                         positive_eigenvalues=int((ev>ev.abs().max()*1e-10).sum()),
                         negative_eigenvalues=int((ev<-ev.abs().max()*1e-10).sum()),
                         tokens=token_lists))
    result=dict(instrument_passed=max(v['function_norm_error'] for v in rows)<1e-8,
                selection='Top8 by sum_v A_vj^2 ||B_j||^2; component energies are not additive capture.',
                source_cache_sha256=receipt['cache']['sha256'],functions=rows,
                wall_seconds=time.perf_counter()-start,
                scope='Frozen weight-only dictionary inspection. Token lists are exploratory, signs arbitrary. Spectral simplicity does not establish stable factors or a behavioral circuit.')
    (P/'TOKEN_DICTIONARY_FUNCTIONS_V1_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({**result,'functions':[{k:v for k,v in row.items() if k!='tokens'} for row in rows]},indent=2))
    assert result['instrument_passed']


if __name__=='__main__':main()
