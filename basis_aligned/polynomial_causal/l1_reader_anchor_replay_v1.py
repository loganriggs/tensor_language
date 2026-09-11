"""Independent objective check; original missing diagnostic gate stays unverified."""
import hashlib,json
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import P,CK

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    previous=json.loads((P/'L1_READER_ANCHOR_V1_AUDIT.json').read_text())
    path=Path(previous['cache']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==previous['cache']['sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu');z=saved['codes'].double();b=saved['dictionary'].double()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    y=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])
    y/=y.norm(dim=1,keepdim=True)
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700));y=y[torch.cat((order[:3072],order[:3072]+4608))]
    assert z.shape==(6144,2304) and b.shape==(2304,1152)
    objective=float((.5*(y.square().sum()-2*((y@b.T)*z).sum()+((z.T@z)*(b@b.T)).sum())+.05*z.abs().sum())/len(y))
    error=abs(objective-previous['current_objective'])
    result=dict(predictions=dict(pred_a_source=True,pred_b_independent_objective=error<=1e-10),
        gram_objective=objective,row_residual_objective=previous['current_objective'],absolute_error=error,
        source=previous['cache'],original_pred_a_status='unverified: no same-iteration logged objective',
        correction='Original pred_a_instrument=true improperly accepted a missing replay. This independent algebra check validates the computed snapshot objective; it cannot supply the absent historical diagnostic.',
        scope='No fitting, live mutation or changed B/C/D thresholds.')
    with (P/'L1_READER_ANCHOR_REPLAY_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
