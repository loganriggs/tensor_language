"""Freeze three candidate modules and verify their actual shared arithmetic."""
import json
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from compiled_quadratic_layer_v1 import CompiledQuadraticLayer,compile_description
from prepare_million_token_panel_v1 import digest
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def main():
    torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(91162131)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);bias=sd['transformer.h.17.mlp.Down_bias'].double()
    refits=torch.load(P/'PILE_FIXED_READER_TRANSFER_V1_WRITERS.pt',weights_only=True,map_location='cpu')
    x=torch.randn(128,1152,dtype=torch.float64);out={};checks={}
    for label,name,new in [('shared_original','data_shared_reader',False),('shared_refit','data_shared_reader',True),('product_refit','data_product',True)]:
        ref=refits[name];source=P/ref['source'];state=torch.load(source,map_location='cpu',weights_only=False)
        model=QuadraticModel(state['config']['kind'],1152);model.load_state_dict(state['best']['model']);a,b,c=model.components()
        writer=ref['writer'] if new else state['best']['writer']
        expected=(((x@a.T)*(x@b.T))@c)@writer.T+bias
        desc=compile_description(model,writer,bias);compiled=CompiledQuadraticLayer(desc)
        bridge64=float((compiled(x)-expected).norm()/expected.norm());desc32={k:v.float() if torch.is_tensor(v) else v for k,v in desc.items()};comp32=CompiledQuadraticLayer(desc32)
        bridge32=float((comp32(x.float()).double()-expected).norm()/expected.norm())
        assert bridge64<=1e-10 and bridge32<=1e-3
        out[label]=desc32;checks[label]=dict(fp64_shared_arithmetic_relative_l2=bridge64,fp32_shared_arithmetic_relative_l2=bridge32,stored_numbers=compiled.numbers(),source_sha256=digest(source),writer_origin='Pile training only' if new else 'Original training fit')
    artifact=P/'PHYSICAL_QUADRATIC_V1_MODULES.pt';assert not artifact.exists();torch.save(out,artifact)
    control=dict(schema='physical.quadratic.compile.control.v1',passed=True,checks=checks,artifact_sha256=digest(artifact),artifact_bytes=artifact.stat().st_size,scope='Shared64 input projections are computed once; frozen coefficients absorb row normalizers. Full native output bias included. Random-state arithmetic control, not native capability or behavioral preservation.')
    with (P/'PHYSICAL_QUADRATIC_V1_COMPILE_CONTROL.json').open('x') as h:json.dump(control,h,indent=2);h.write('\n')
    print(json.dumps(control))
if __name__=='__main__':main()
