"""Post-screen sign accounting; preserve original failed support prediction.

A FP64 algebra <=1e-9 and FP32/64 target score agreement <=1e-5.
B own forward raw target contribution negative on >=75% of each target family.
C fixed-norm target gain >=75% observed mean CE gain; mean absolute RMS term
<=25% mean absolute direct term in each target family.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root=Path(__file__).parent;stem='SHARED_NODE_PARENT1_FINEWEB_V1'
    output=root/'SHARED_NODE_PARENT1_SIGN_AUDIT_V1.json'
    assert not output.exists()
    panel=torch.load(root/(stem+'_ROWS.pt'),weights_only=True,map_location='cpu')
    cache=torch.load(root/(stem+'_ENDPOINTS.pt'),weights_only=True,map_location='cpu')
    saved=torch.load(root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')
    node=saved['nodes'][1]
    binding=json.loads((root/(stem+'_BINDING.json')).read_text())['files']
    checkpoint=next(path for path in binding if path.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    target_rows=weights['lm_head.weight'][panel['rows'][:,-1]]
    ports=cache['ports'];x=ports['input'].double()
    h32=ports['pre']+ports['native_output'];h=h32.double()
    u=node['reader'].double();partners=node['partners'].double()
    writers=torch.linalg.solve_triangular(saved['output_whitener'].double(),node['writers'].double(),upper=True)
    eps=torch.finfo(torch.float32).eps
    scale=(h.square().mean(1)+eps).sqrt()
    raw=(target_rows.double()*h).sum(1)/scale
    score=30*torch.tanh(raw/30)
    raw32=(target_rows*F.rms_norm(h32,(1152,))).sum(1)
    agreement=float((30*torch.tanh(raw32/30)-score).abs().max())
    scores=cache['scores'].double()
    records=[];checks=[];terms=[]
    for branch in (0,1):
        scalar=(x@u)*(x@partners[:,branch])
        delta=(scalar[:,None]*writers[:,branch]).float().double()
        changed32=h32-(delta.float())
        changed=changed32.double()
        new_scale=(changed.square().mean(1)+eps).sqrt()
        direct_raw=(target_rows.double()*changed).sum(1)/scale
        final_raw=(target_rows.double()*changed).sum(1)/new_scale
        direct=30*torch.tanh(direct_raw/30)-score
        final=30*torch.tanh(final_raw/30)-score
        radial=final-direct
        ce_added=scores[:,:,branch+1].flatten()
        logz_change=ce_added+final
        reconstructed=-direct-radial+logz_change
        checks.append(float((reconstructed-ce_added).abs().max()))
        forward=(target_rows.double()*delta).sum(1)
        terms.append(torch.stack((forward,direct,radial,logz_change,ce_added),-1).reshape(3,24,5))
        for family in range(3):
            sl=slice(family*24,(family+1)*24)
            records.append(dict(branch=branch,family=family,
                fraction_negative_forward_target=float((forward[sl]<0).double().mean()),
                mean_forward_raw_target_contribution=float(forward[sl].mean()),
                mean_direct_target_score_gain=float(direct[sl].mean()),
                mean_rms_target_score_change=float(radial[sl].mean()),
                mean_implied_log_normalizer_change=float(logz_change[sl].mean()),
                mean_ce_added=float(ce_added[sl].mean()),
                mean_absolute_direct=float(direct[sl].abs().mean()),
                mean_absolute_rms=float(radial[sl].abs().mean())))
    own=[r for r in records if r['branch']==r['family']]
    idx=torch.randint(24,(2000,24),generator=torch.Generator().manual_seed(4701))
    contrasts=torch.stack((scores[0,:,1]-scores[0,:,2],scores[1,:,2]-scores[1,:,1]))
    intervals=torch.quantile(contrasts[:,idx].mean(2),torch.tensor([.025,.975],dtype=torch.float64),dim=1)
    valid=max(checks)<=1e-9 and agreement<=1e-5
    result=dict(pred_a=valid,pred_b=valid and all(r['fraction_negative_forward_target']>=.75 for r in own),
        pred_c=valid and all(r['mean_direct_target_score_gain']>=.75*(-r['mean_ce_added']) and
            r['mean_direct_target_score_gain']>0 and r['mean_absolute_rms']<=.25*r['mean_absolute_direct'] for r in own),
        fp64_accounting_error=max(checks),fp32_fp64_target_score_error=agreement,rows=records,
        own_minus_other_ce_added_95=intervals.tolist(),
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [root/(stem+'_ROWS.pt'),root/(stem+'_ENDPOINTS.pt'),root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt']},
        scope='Post-result accounting on frozen historical FineWeb endpoints. Log-normalizer '
              'change inferred from measured CE plus recomputed target scores, not independent full-vocab replay. '
              'No fitting; original support prediction remains failed, inhibitory interpretation unvalidated.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':main()
