"""Freeze prior square150 reader; extract its exact native coefficient write."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    pilot=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    ck=next(k for k in pilot['binding'] if k.endswith('pytorch_model.bin'))
    old=torch.load(root/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt',weights_only=True,map_location='cpu')
    index=json.loads((root/'LL1_SHARED_SQUARE_ALIAS_V1_AUDIT.json').read_text())['old_square_index']
    raw=old['model']['a'][index].double();u=raw/raw.norm()
    sd=torch.load(ck,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    physical=d@((l@u)*(r@u))
    wh=torch.load('/dev/shm/bilin18_matched_shared_groups_v1_ll1_native.pt',weights_only=True,map_location='cpu')['output_whitener'].double()
    writer=wh@physical;old_writer=wh@old['writer'][:,index].double()*raw.norm().square()
    native=(l,r,wh@d);atom=(u[None],u[None],writer[:,None]);energy=writer.square().sum()
    identity=float(abs(inner(native,atom)-energy)/energy)
    token=sd['lm_head.weight'].double()@physical
    metric_error=float(abs(token.square().sum()-energy)/energy)
    old_cos=float(old_writer@writer/(old_writer.norm()*writer.norm()))
    tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    vocab=tokenizer.get_vocab_size();lists={}
    normalized=token/token.norm()
    for name,sign in (('positive',1),('negative',-1)):
        ids=(sign*normalized).topk(12).indices.tolist()
        lists[name]=[dict(id=i,text=tokenizer.decode([i]) if i<vocab else f'<extra output {i}>',normalized_loading=float(normalized[i])) for i in ids]
    path=root/'STABLE_SQUARE150_NATIVE_INTERFACE_V1.pt'
    torch.save(dict(reader=u,physical_writer=physical,definition='MLP17 normalized input x -> physical_writer*(reader dot x)^2. Add native remainder and Down_bias for complete MLP17. Final RMS/tanh remain native.'),path)
    result=dict(pred_a=max(identity,metric_error)<1e-9,pred_b=old_cos>=.95,index=index,
                prior_writer_native_cosine=old_cos,prior_writer_norm_over_native=float(old_writer.norm()/writer.norm()),
                native_coefficient_energy_fraction=float(energy/99245061353.47293),projection_identity_error=identity,unembedding_metric_replay=metric_error,
                top128_token_energy=float(normalized.square().topk(128).values.sum()),common_token_energy=float(normalized.sum().square()/len(token)),tokens=lists,
                artifact=path.name,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parameter_numbers=u.numel()+physical.numel(),
                scope='Exact native coefficient projection onto a frozen previously discovered squared reader. Remainder is required; direct token writes precede final normalization/tanh and do not prove a semantic or selective behavioral effect. No corpus used.')
    (root/'STABLE_SQUARE150_NATIVE_INTERFACE_V1_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
