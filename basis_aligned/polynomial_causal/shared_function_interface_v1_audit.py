"""Fixed scalar function: physical read/write interface and exploratory text.

A unit read/write and U norm<=1e-8; B top128token energy>=.25;
C common-token energy<=.5. No semantic label inferred from token/context lists.
"""
import hashlib,json,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from native_support_exchange_v1_audit import P,CK


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'SHARED_FUNCTION_INTERFACE_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    receipt=json.loads((P/'SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json').read_text());source=receipt['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu');q=saved['native_output_readout'];wh=saved['unembedding_whitener']
    write=torch.linalg.solve_triangular(wh,q[:,None],upper=True)[:,0];read=wh.T@q
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);token=sd['lm_head.weight'].double()@write
    tokenizer_path=Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));vocab=tokenizer.get_vocab_size()
    squared=token.square();energy=float(squared.sum());sparse=float(squared.topk(128).values.sum()/energy)
    common=float(token.sum().square()/(len(token)*energy));lists={}
    for name,sign in [('positive',1),('negative',-1)]:
        ids=(sign*token).topk(12).indices.tolist()
        lists[name]=[dict(id=i,text=tokenizer.decode([i]) if i<vocab else f'<extra output {i}>',coefficient=float(token[i])) for i in ids]
    validation=json.loads((P/'SHARED_FUNCTION_FINEWEB_V1_AUDIT.json').read_text())
    activations_source=validation['cache'];data_source=validation['data_source']
    for item in (activations_source,data_source):assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    scalar=torch.load(activations_source['path'],weights_only=True,map_location='cpu')['native'].reshape(64,128)
    rows=torch.load(data_source['path'],weights_only=True,map_location='cpu')['rows']
    contexts={}
    for name,sign in [('positive',1),('negative',-1)]:
        val,pos=(sign*scalar).max(dim=1);selected=val.topk(6).indices.tolist()
        contexts[name]=[dict(row=i,position=int(pos[i]),activation=float(scalar[i,pos[i]]),
            context=tokenizer.decode(rows[i,max(0,int(pos[i])-24):int(pos[i])+1].tolist()),
            next_token=tokenizer.decode([int(rows[i,int(pos[i])+1])])) for i in selected]
    replay=max(abs(float(read@write)-1),abs(energy-1))
    artifact=Path('/dev/shm/bilin18_shared_function_interface_v1.pt');assert not artifact.exists()
    torch.save(dict(physical_read=read,physical_write=write,token_loadings=token,source=source,
        scalar_definition='read dot (MLP17 output minus Down_bias)',
        edit_definition='MLP17 output + physical_write * scalar_delta; preserve all native downstream nonlinearities'),artifact)
    result=dict(predictions=dict(pred_a_interface=replay<=1e-8,pred_b_sparse_tokens=sparse>=.25,pred_c_not_common=common<=.5),
        interface_replay=replay,top128_token_energy=sparse,common_token_energy=common,
        extra_output_energy=float(squared[vocab:].sum()/energy),tokens=lists,contexts=contexts,
        tokenizer=dict(path=str(tokenizer_path),sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()),
        source=source,data_source=data_source,cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),
        seconds=time.perf_counter()-start,
        scope='Exploratory fixed-candidate output/context inspection on historical FineWeb. Algebraic port coordinate only; no behavioral selectivity or token-list semantic identity.')
    with out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
