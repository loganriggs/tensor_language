"""Exact compact component pullback through native attention OV, with routing explicit.

CPU only. Angles describe weight subspaces, not semantic task sharing. The full
QK1*QK2 routing and all normalization/background interfaces remain required.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,time
from pathlib import Path
import torch

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    result=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text())
    assert result['predictions']['pred_a_instrument']
    cache=result['cache'];assert digest(cache['path'])==cache['sha256']
    saved=torch.load(cache['path'],map_location='cpu',weights_only=True)
    selected=saved['programs'][saved['best']]
    assert result['fits'][saved['best']]['converged']
    a,w,b=[selected[k] for k in ['reader','writers','partner_readers']]
    frame=torch.cat((a[None,:],b),dim=0)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    o=sd['transformer.h.17.attn.c_proj.weight'].double()
    vc=sd['transformer.h.17.attn.c_v.weight'].double()
    vb=sd['transformer.h.0.attn.c_v.weight'].double()
    mix=float(sd['transformer.h.17.attn.lamb'])
    values=torch.cat(((1-mix)*vc,mix*vb),dim=1)
    ports=torch.stack([frame@o[:,128*h:128*(h+1)]@values[128*h:128*(h+1)] for h in range(9)])
    gates=ports[:,0,:];normalized=gates/gates.norm(dim=1,keepdim=True)
    gate_cos=normalized@normalized.T
    bases=[];ranks=[]
    for h in range(9):
        _,sing,right=torch.linalg.svd(ports[h,1:,:],full_matrices=False)
        rank=int((sing>1e-10*sing[0]).sum());ranks.append(rank);bases.append(right[:rank].T)
    pairs=[]
    for h in range(9):
        for k in range(h+1,9):
            principal=torch.linalg.svdvals(bases[h].T@bases[k])
            pairs.append(dict(heads=[h,k],gate_cosine=float(gate_cos[h,k]),
                              partner_principal_cosines=principal.tolist()))
    # Formal signed routing is intentionally independent here: this validates
    # the OV substitution for every routing, rather than claiming a QK test.
    torch.manual_seed(1193);n,sources=13,5
    xi=torch.randn(n,sources,2304);routing=torch.randn(n,9,sources)
    base=torch.randn(n,1152)
    head_values=(xi@values.T).reshape(n,sources,9,128)
    attention=torch.einsum('nhp,nphd->nhd',routing,head_values).reshape(n,1152)@o.T
    raw=base+attention
    # FP64 polynomial identity with an explicit common RMS divisor. This is
    # not a claim to reproduce the model's FP32 RMS rounding bit-for-bit.
    rho=(raw.square().mean(dim=1)+torch.finfo(torch.float32).eps).sqrt()
    x=raw/rho[:,None]
    direct=(x@a)[:,None]*((x@b.T)@w.T)
    projected=base@frame.T+torch.einsum('nhp,hfi,npi->nf',routing,ports,xi)
    folded=projected[:,0,None]*(projected[:,1:]@w.T)/rho[:,None].square()
    error=float((folded-direct).norm()/direct.norm())
    output=dict(instrument_passed=error<1e-10,signed_multisource_pullback_relative_error=error,
                source_cache_sha256=cache['sha256'],selected_fit=saved['best'],native_value_mix=mix,
                gate_source_norms=gates.norm(dim=1).tolist(),gate_source_cosines=gate_cos.tolist(),
                partner_source_ranks=ranks,head_pairs=pairs,
                price=dict(body_forwards=0,corpus_access=False,heads=9,source_tuple_width=2304,
                           projected_value_readers=9*17,downstream_products=16,
                           routing_and_residual_background_retained=True),
                scope='Exact OV/source pullback of a frozen weight-only candidate. Pairwise source geometry is descriptive; QK1*QK2 routing stays explicit. No task identification, physical simplification or input-distribution claim.',
                wall_seconds=time.perf_counter()-start)
    target=P/'SHARED_READER_UPSTREAM_PORTS_V1_AUDIT.json'
    with target.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['head_pairs','gate_source_cosines']},indent=2))
    assert output['instrument_passed']


if __name__=='__main__':main()
