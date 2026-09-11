"""Frozen weight-discovered scalar function on already cached FineWeb states.

A factor/form<=1e-10 and cached native precision replay<=.01 norm-relative;
B one-product relative MSE<=.25; C four products<=.1. No fitting or forwards.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'SHARED_FUNCTION_FINEWEB_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    receipt=json.loads((P/'SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json').read_text());source=receipt['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    program=torch.load(source['path'],weights_only=True,map_location='cpu')
    cache=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt')
    sha=hashlib.sha256(cache.read_bytes()).hexdigest();assert sha=='b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de'
    data=torch.load(cache,weights_only=True,map_location='cpu');x=data['ports']['input'].reshape(-1,1152).double()
    assert x.shape==(8192,1152)
    form=program['native_form'];q=program['native_output_readout'];wh=program['unembedding_whitener']
    y=torch.einsum('ni,ij,nj->n',x,form,x)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    coefficient=q@wh@d
    factor=((x@l.T)*(x@r.T))@coefficient
    stored=(data['ports']['native_output'].reshape(-1,1152).double()-sd['transformer.h.17.mlp.Down_bias'].double())@(wh.T@q)
    exact=float((factor-y).norm()/y.norm());precision=float((stored-y).norm()/y.norm())
    rows=[];predictions=[]
    for k in (1,2,4,8,16):
        item=program['programs'][str(k)];a,b=item['left'],item['right']
        pred=((x@a.T)*(x@b.T)).sum(1);predictions.append(pred)
        py=pred-pred.mean();yy=y-y.mean()
        rows.append(dict(products=k,relative_mse=float((pred-y).square().sum()/y.square().sum()),
            centered_relative_mse=float((py-yy).square().sum()/yy.square().sum()),
            centered_correlation=float((py@yy)/(py.norm()*yy.norm())),
            sign_agreement=float(((pred>0)==(y>0)).double().mean()),
            mean=float(pred.mean()),rms=float(pred.square().mean().sqrt())))
    passed=dict(pred_a_instrument=exact<=1e-10 and precision<=.01,
        pred_b_one_product=rows[0]['relative_mse']<=.25,pred_c_four_products=rows[2]['relative_mse']<=.1)
    result=dict(predictions=passed,rows=rows,native_factor_form_replay=exact,native_precision_replay=precision,
        native_mean=float(y.mean()),native_rms=float(y.square().mean().sqrt()),
        native_centered_rms=float((y-y.mean()).square().mean().sqrt()),source=source,
        data_source=dict(path=str(cache),sha256=sha,rows=64,positions=8192,previously_opened=True),
        body_forwards=0,fit_steps=0,seconds=time.perf_counter()-start,
        scope='Frozen scalar-function validation on cached FineWeb, not fresh/OOD text, full-logit preservation or a selectively manipulable circuit.')
    artifact=Path('/dev/shm/bilin18_shared_function_fineweb_v1.pt');assert not artifact.exists()
    torch.save(dict(native=y,predictions=torch.stack(predictions),products=[1,2,4,8,16],source=source,data_sha256=sha),artifact)
    result['cache']=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
