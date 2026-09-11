"""Fixed midpoint leading common/private producer functions; no text or GPU.

Registered on AGENT_BOARD before execution: A errors<1e-9; B seven of nine
common functions capture>=.5 with16 products; C median common k90<=.8 private.
One real product has at most one positive and one negative eigenvalue.
The optimal k-product Frobenius approximation retains k largest eigenvalues
of each sign. This scalar result does not solve the joint shared dictionary.
"""
import hashlib
import json
import time
from pathlib import Path
import torch
from producer_function_pairs_v1 import pairs, shared_private
from sparse_product_dictionary_v1 import form, unit_product

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    out=P/'PRODUCER_SCALAR_PRODUCTS_V1_AUDIT.json';assert not out.exists()
    start=time.perf_counter();inputs={}
    def cached(receipt):
        c=json.loads((P/receipt).read_text())['cache']
        assert digest(c['path'])==c['sha256'];inputs[c['path']]=c['sha256']
        return torch.load(c['path'],weights_only=True,map_location='cpu')
    frozen=cached('COUPLED_PRODUCER_MIDPOINT_V1_AUDIT.json')
    producer=cached('MLP16_PRODUCER_OVERLAP_V1_RESULT.json')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.16.mlp.{s}.weight'].double() for s in ['Left','Right','Down']]
    g=producer['metric'];rows=[];errors=[];atoms=[]
    for h in range(9):
        left,right,c=pairs(frozen['frames'][h].T,producer['ov_readers'][h],g)
        split=shared_private(left,right,c)
        for kind in ['common','private']:
            reader=split[kind][0];q=form(l,r,reader@d)
            ev,v=torch.linalg.eigh(q);energy=q.square().sum()
            errors.append(float(abs(energy-reader@g@reader)/energy))
            errors.append(float((q@v-v*ev).norm()/q.norm()))
            pos=ev.clamp_min(0).flip(0);neg=(-ev).clamp_min(0)
            curve=(pos.square()+neg.square()).cumsum(0)/energy
            k90=int(torch.searchsorted(curve,.9))+1
            k50=int(torch.searchsorted(curve,.5))+1
            # Construct the actual optimal 16-product function, not just a score.
            pp=v[:,-16:].flip(1)*pos[:16].sqrt()
            nn=v[:,:16]*neg[:16].sqrt()
            pl,pr=(pp+nn).T,(pp-nn).T
            approx=form(pl,pr,torch.ones(16))
            errors.append(float(abs(1-(q-approx).square().sum()/energy-curve[15])))
            if h==0 and kind=='common':
                _,_,magnitude=unit_product(q)
                errors.append(float(abs(magnitude.square()/energy-curve[0])))
            rows.append(dict(head=h,kind=kind,pair_cosine=float(c[0]),
                k50=k50,k90=k90,capture={str(k):float(curve[k-1]) for k in [1,4,16,64,128,256,512]},
                positive_rank=int((ev>1e-10*ev.abs().max()).sum()),
                negative_rank=int((ev<-1e-10*ev.abs().max()).sum())))
            atoms.append(dict(head=h,kind=kind,reader=reader,product_left=pl,product_right=pr))
        print(json.dumps(rows[-2:]),flush=True)
    common=[x for x in rows if x['kind']=='common'];private=[x for x in rows if x['kind']=='private']
    med=lambda xs:sorted(xs)[len(xs)//2]
    mc=med([x['k90'] for x in common]);mp=med([x['k90'] for x in private])
    valid=max(errors)<1e-9
    cache=Path('/dev/shm/bilin18_producer_scalar_products_v1.pt');assert not cache.exists()
    torch.save(dict(atoms=atoms,inputs=inputs),cache)
    result=dict(predictions=dict(pred_a_instrument=valid,
        pred_b_small_common_function=valid and sum(x['capture']['16']>=.5 for x in common)>=7,
        pred_c_common_simpler=valid and mc<=.8*mp),heads=rows,
        summary=dict(common_median_k90=mc,private_median_k90=mp,
            common_mean_capture16=sum(x['capture']['16'] for x in common)/9,
            private_mean_capture16=sum(x['capture']['16'] for x in private)/9),
        maximum_instrument_error=max(errors),seconds=time.perf_counter()-start,
        source_sha256=digest(__file__),inputs=inputs,
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size),
        scope='One leading canonical pair per head, fixed midpoint QK/frozen OV, MLP16 quadratic term only. Exact scalar real-product coefficient optimum, not shared multioutput dictionary or causal result. No text.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2));assert valid

if __name__=='__main__':main()
