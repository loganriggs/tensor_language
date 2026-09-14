"""Fixed five-program exact-sharing screen; no quantization or model fitting.

A: serialize/reload/reconstruct every tensor byte and metadata entry exactly.
B: bank plus routing and decoder saves >=1% versus independent program files.
All original executor sources remain required. No compute-reuse inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import torch

P = Path(__file__).resolve().parent
NAMES = ['regional_shared_producers_8_2_9_8_v1', 'regional_shared_head2_v1',
         'three_corner_head17_interaction_v1', 'first_token_value_path_v1',
         'fixed_writer_self_mlp10_v1']
DECODER = '''"""Reconstruct original program structures from an exact shared tensor bank."""
def decode(tree, bank):
    kind, value = tree
    if kind == 'tensor':
        return bank[value].clone()
    if kind == 'dict':
        return {key: decode(child, bank) for key, child in value}
    if kind == 'list':
        return [decode(child, bank) for child in value]
    if kind == 'tuple':
        return tuple(decode(child, bank) for child in value)
    if kind == 'literal':
        return value
    raise ValueError(kind)
'''


def tensor_id(tensor):
    t = tensor.detach().cpu().contiguous()
    header = json.dumps([str(t.dtype), list(t.shape)]).encode()
    return hashlib.sha256(header + t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    out = P/'INTERACTION_PACKAGE_SHARED_BANK_V1_RESULT.json'
    artifact = P/'INTERACTION_PACKAGE_SHARED_BANK_V1_PROGRAM.pt'
    decoder = P/'interaction_package_shared_bank_v1_decode.py'
    if any(f.exists() for f in [out, artifact, decoder]):
        raise FileExistsError('Existing registered evidence preserved')
    bank, uses, inputs, trees = {}, {}, {}, {}

    def encode(value, path):
        if isinstance(value, torch.Tensor):
            key = tensor_id(value)
            bank.setdefault(key, value.detach().cpu().contiguous().clone())
            uses.setdefault(key, []).append(path)
            return ('tensor', key)
        if isinstance(value, dict):
            return ('dict', [(k, encode(v, path+'/'+str(k))) for k,v in value.items()])
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, [encode(v,path+'/'+str(i)) for i,v in enumerate(value)])
        if value is None or isinstance(value, (str,int,float,bool)):
            return ('literal',value)
        raise TypeError(type(value))

    independent_files = independent_payload = source_bytes = 0
    for name in NAMES:
        folder = P/'extracted_circuits'/name
        path = folder/'program.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        trees[name] = encode(value,name)
        inputs[name] = dict(program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            serialized_bytes=path.stat().st_size,
                            sources={f.name:dict(bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in folder.glob('*.py')})
        independent_files += path.stat().st_size
        source_bytes += sum(v['bytes'] for v in inputs[name]['sources'].values())
    independent_payload = sum(bank[k].numel()*bank[k].element_size()*len(v) for k,v in uses.items())
    torch.save(dict(bank=bank,trees=trees), artifact)
    decoder.write_text(DECODER)
    namespace = {}
    exec(DECODER, namespace)
    loaded = torch.load(artifact, weights_only=True)
    # Re-encoding reconstructs every dtype, shape, tensor byte and literal structure.
    before = {k: list(v) for k,v in uses.items()}
    checks = {name: encode(namespace['decode'](tree,loaded['bank']),name)==tree
              for name,tree in loaded['trees'].items()}
    shared_bytes = artifact.stat().st_size + decoder.stat().st_size
    unique_payload = sum(t.numel()*t.element_size() for t in bank.values())
    duplicates = [dict(sha256=k,shape=list(bank[k].shape),dtype=str(bank[k].dtype),
                       bytes=bank[k].numel()*bank[k].element_size(),uses=v)
                  for k,v in before.items() if len(v)>1]
    result = dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(checks.values()),
                  pred_b=shared_bytes<=.99*independent_files,programs=NAMES,checks=checks,
                  independent_program_bytes=independent_files,bank_and_decoder_bytes=shared_bytes,
                  serialized_saving_fraction=1-shared_bytes/independent_files,
                  independent_tensor_payload=independent_payload,unique_tensor_payload=unique_payload,
                  required_executor_source_bytes=source_bytes,
                  complete_declared_independent_bytes=independent_files+source_bytes,
                  complete_declared_shared_bytes=shared_bytes+source_bytes,
                  tensor_leaves=sum(map(len,before.values())),unique_tensors=len(bank),
                  duplicate_groups=duplicates,inputs=inputs,
                  artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  decoder_sha256=hashlib.sha256(decoder.read_bytes()).hexdigest(),
                  scope='Five distinct conditional computations; excludes alternate same-circuit sparse/even/mixed versions. Exact whole-tensor storage sharing only, not new circuits, shared execution, native validation or closed-model pricing. Decoder clones to preserve independent mutability, so no resident-memory saving claimed.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['inputs','duplicate_groups']},indent=2))


if __name__ == '__main__':
    main()
