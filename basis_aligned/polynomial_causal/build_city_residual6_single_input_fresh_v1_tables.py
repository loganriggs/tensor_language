"""Extend only the fixed weight-derived token tables for the fresh rows."""
import hashlib,json,os
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;STEM='CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1'
@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2)
    rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows']
    ids=torch.tensor(sorted({t for r in rows for t in r['ids']}))
    binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    initial=F.rms_norm(sd['transformer.wte.weight'][ids].float(),(1152,));lam=sd['transformer.h.0.lambdas']
    first=F.linear(F.rms_norm(lam[0]*initial+lam[1]*initial,(1152,)),sd['transformer.h.0.attn.c_v.weight'])
    original=torch.load(P/'CITY_ATTENTION7_GENERATOR_V1_PROGRAM.pt',weights_only=True)
    updated=dict(original,token_ids=ids,initial_table=initial,first_table=first)
    changed={'token_ids','initial_table','first_table'}
    unchanged=all(torch.equal(v,updated[k]) for k,v in original.items() if k not in changed)
    lookup={int(t):i for i,t in enumerate(original['token_ids'])};overlap=[(i,lookup[int(t)]) for i,t in enumerate(ids) if int(t) in lookup]
    max_error=max(float((updated[k][i]-original[k][j]).abs().max()) for i,j in overlap for k in ['initial_table','first_table'])
    out=P/(STEM+'_ATTENTION7.pt');assert not out.exists()
    torch.save(updated,out)
    result={'pred_a':unchanged,'pred_b':max_error<=1e-6,'tokens':len(ids),'overlap_tokens':len(overlap),'overlap_max_abs':max_error,'non_table_weights_unchanged':unchanged,'table_float_scalars':initial.numel()+first.numel(),'checkpoint_sha256':binding[checkpoint],'program_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'model_forwards':0,'scope':'Outcome-blind weight-table extension; no new fitted parameters or behavioral results.'}
    (P/(STEM+'_TABLES_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
