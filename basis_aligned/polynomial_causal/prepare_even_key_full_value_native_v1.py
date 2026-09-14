"""CPU backend equivalence and dependency freeze; never starts GPU work."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import torch
import torch.nn.functional as F
from even_key_value_bank_v1 import execute
from even_key_value_native_backend_v1 import FullValueComponents
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
STEM='EVEN_KEY_FULL_VALUE_NATIVE_V1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    out=P/(STEM+'_CPU_CONTROL.json');bound=P/(STEM+'_BINDING.json')
    assert not out.exists() and not bound.exists()
    torch.set_num_threads(2)
    p=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    checkpoint=Path('/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    state=torch.load(checkpoint,weights_only=True,mmap=True)
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    runner=FullValueComponents(p,'cpu');records=[]
    for i in [0,18,36,54]:
        n=len(rows[i]['ids']);ids=torch.tensor([rows[i]['ids']])
        initial=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']),(1152,),eps=torch.finfo(torch.float32).eps).double()
        first=initial@p['first_value'].double().T
        for dtype in [torch.float32,torch.float64]:
            current=F.rms_norm(cache['raw9'][0,i,:n][None].to(dtype),(1152,),eps=torch.finfo(torch.float32).eps)
            gold,_=execute(current,initial,p);even,odd=runner(current,first)
            error=float((gold-even).norm()/gold.norm())
            records.append(dict(row=i,dtype=str(dtype),relative_error=error,odd_norm=float(odd.norm())))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(r['relative_error']<=1e-10 for r in records),records=records,scope='CPU backend versus frozen full-value bank, four old contexts, FP32/FP64 inputs; no GPU/native behavior.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    assert result['pred_a'],result
    paths=set(json.loads((P/'KEY_SPAN_NATIVE_V1_BINDING.json').read_text())['files'])
    additions=[P/'even_key_value_native_backend_v1.py',P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',
               P/(STEM+'_PREREGISTRATION.md'),Path(__file__),out,
               ROOT/'basis_aligned/bilinear_quotient/ops/run_even_key_full_value_native_v1.py']
    paths.update('/workspace/tensor_language/'+str(f.relative_to(ROOT)) for f in additions)
    def physical(name):
        if name.startswith('/workspace/tensor_language/'):
            return ROOT/name[len('/workspace/tensor_language/'):]
        if name.startswith('/workspace/.hf_home/'):
            return Path('/home/loganriggs/.local/share/bilin18/hf_home')/name[len('/workspace/.hf_home/'):]
        return Path(name)
    binding={name:digest(physical(name)) for name in sorted(paths)}
    pinned=[v for k,v in binding.items() if k.endswith('/pytorch_model.bin')]
    assert pinned==['680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3']
    bound.write_text(json.dumps(dict(files=binding),indent=2)+'\n')
    print(json.dumps(result,indent=2));print('Bound',len(binding),'dependencies')


if __name__=='__main__':main()
