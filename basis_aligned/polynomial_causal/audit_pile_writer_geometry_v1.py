"""Exact coefficient-function comparison for unchanged readers, changed writers."""
import json,time
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from joint_quadratic_fit_v1 import product_cross
from prepare_million_token_panel_v1 import digest
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def main():
    tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    path=P/'PILE_FIXED_READER_TRANSFER_V1_WRITERS.pt';refits=torch.load(path,weights_only=True,map_location='cpu')
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True);u=sd['lm_head.weight'].double();m=u.T@u;del u
    results={}
    for name,refit in refits.items():
        f=torch.load(P/refit['source'],weights_only=False,map_location='cpu');model=QuadraticModel(f['config']['kind'],1152);model.load_state_dict(f['best']['model'])
        a,b,c=model.components();g=c.T@product_cross(a,b,a,b)@c
        old=f['best']['writer'];new=refit['writer']
        def inner(w,v):return ((w.T@m@v)*g).sum()
        oo,nn,on=inner(old,old),inner(new,new),inner(old,new)
        assert float(oo)>0 and float(nn)>0
        cos=float(on/(oo*nn).sqrt());assert abs(cos)<=1+1e-8
        delta=inner(new-old,new-old)
        identity=float(abs(delta-(oo+nn-2*on))/torch.maximum(oo,nn));assert identity<1e-8
        results[name]=dict(coefficient_function_cosine=cos,new_to_old_coefficient_norm=float((nn/oo).sqrt()),coefficient_change_relative_l2=float((delta/oo).sqrt()),difference_identity_relative_error=identity)
    result=dict(schema='pile.writer.geometry.v1',results=results,source_sha256=digest(path),wall_seconds=time.perf_counter()-tic,scope='Exact all-input coefficient-function geometry weighted by all-U. Natural-state error measured separately by transfer receipt. Rotating a function in coefficient space is not causal identification or evidence of a new semantic component.')
    with (P/'PILE_WRITER_GEOMETRY_V1_AUDIT.json').open('x') as h:json.dump(result,h,indent=2);h.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
