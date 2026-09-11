"""Weight-only token readouts for eight output modes of a frozen stable boundary.

A U-mode orthonormality<=1e-8; B six modes top128loadingenergy>=.25;
C all extra-output energy<=.01. Token strings are descriptive, not semantics.
"""
import hashlib,json,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cancellation_group_v1_audit import load
from native_support_exchange_v1_audit import P,CK
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'CANCELLATION_TOKEN_READOUTS_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    parent=json.loads((P/'CANCELLATION_BOUNDARY_FUTURE_V1_AUDIT.json').read_text());ids=parent['products']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;current=load(parent['later_source'],wh)
    a,b,w=current[0][ids],current[1][ids],current[2][:,ids]
    ib=torch.linalg.qr(torch.cat((a,b)).T,mode='reduced').Q;ob=torch.linalg.qr(w,mode='reduced').Q
    core=dense(a@ib,b@ib,ob.T@w);left,singular,right=torch.linalg.svd(core.reshape(len(ids),-1),full_matrices=False)
    directions=ob@left[:,:8];physical=torch.linalg.solve_triangular(wh,directions,upper=True)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    token_modes=sd['lm_head.weight'].double()@physical
    tokenizer_path=Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));vocab=tokenizer.get_vocab_size()
    error=float((token_modes.T@token_modes-torch.eye(8)).abs().max());rows=[]
    for mode in range(8):
        coefficients=token_modes[:,mode];peak=int(coefficients.abs().argmax())
        sign=1. if coefficients[peak]>=0 else -1.;coefficients=coefficients*sign
        squared=coefficients.square();energy=squared.sum();lists={}
        for name,direction in [('positive',1),('negative',-1)]:
            selected=(direction*coefficients).topk(12).indices.tolist()
            lists[name]=[dict(id=i,text=tokenizer.decode([i]) if i<vocab else f'<extra output row {i}>',
                              coefficient=float(coefficients[i])) for i in selected]
        row=dict(mode=mode,group_coefficient_energy_share=float(singular[mode].square()/singular.square().sum()),
            top128_token_energy=float(squared.topk(128).values.sum()/energy),
            extra_output_energy=float(squared[vocab:].sum()/energy),
            common_token_direction_energy=float(coefficients.sum().square()/(len(coefficients)*energy)),
            effective_token_count=float(energy.square()/squared.square().sum()),tokens=lists)
        rows.append(row)
    cache=Path('/dev/shm/bilin18_cancellation_token_readouts_v1.pt');assert not cache.exists()
    torch.save(dict(physical_output_directions=physical,token_modes=token_modes,
                    input_basis=ib,quadratic_functions=(singular[:8,None]*right[:8]).reshape(8,106,106),
                    products=ids,source=parent['later_source'],sign_convention='Saved SVD signs; printed lists separately orient maximum-absolute token positive.'),cache)
    result=dict(predictions=dict(pred_a_instrument=error<=1e-8,
        pred_b_sparse_readouts=sum(r['top128_token_energy']>=.25 for r in rows)>=6,
        pred_c_padding=all(r['extra_output_energy']<=.01 for r in rows)),
        output_orthonormality_error=error,rows=rows,tokenizer=dict(path=str(tokenizer_path),vocabulary_size=vocab,
            sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()),source=parent['later_source'],
        cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),seconds=time.perf_counter()-start,
        scope='Exploratory token readout lists from a frozen fitted group. No input activation sign, '
              'frequency, task, OOD or selective behavior inferred; common logit direction not assumed removable through RMS/tanh.')
    with out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    print(json.dumps(rows[:3],indent=2,ensure_ascii=False))


if __name__=='__main__':main()
