"""Same no-fit input-gating test with the prospectively frozen corpus-shift panel.

A coefficient reconstruction, orientation, finite/pairing checks <=1e-9.
B each branch has paired standardized own-minus-other suppression >=.25.
C both paired bootstrap95% lower bounds >0.
The 24-prefix analysis is post-result; the128-prefix analysis is prospective.
"""
import hashlib
import json
import sys
from pathlib import Path
import torch


@torch.no_grad()
def main(label):
    assert label in ('initial','separate','corpus_shift')
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent
    stem='SHARED_NODE_PARENT1_'+dict(initial='FINEWEB_V1',separate='SUPPRESSION_V1',corpus_shift='CORPUS_SHIFT_V1')[label]
    output=root/f'BRANCH_INPUT_GATING_V2_{label.upper()}.json'
    assert not output.exists()
    panel_path=root/(stem+'_ROWS.pt');cache_path=root/(stem+'_ENDPOINTS.pt')
    panel=torch.load(panel_path,weights_only=True,map_location='cpu')
    cache=torch.load(cache_path,weights_only=True,map_location='cpu')
    source=root/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    saved=torch.load(source,weights_only=True,map_location='cpu');node=saved['nodes'][1]
    n=len(panel['documents']);x=cache['ports']['input'].double()
    assert x.shape==(3*n,1152)
    for i,record in enumerate(panel['metadata']):
        assert record['document']==panel['documents'][i%n] and record['family']==i//n
    binding=json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    c=node['writers'].double();norm=c.norm(dim=0);unit=c/norm
    physical=torch.linalg.solve_triangular(saved['output_whitener'].double(),unit,upper=True)
    orientation=torch.stack([weights['lm_head.weight'][family].double().mean(0)@physical[:,j]
                             for j,family in enumerate(panel['families'])])
    signs=orientation.sign();assert bool((signs!=0).all())
    scalar=(x@node['reader'])[:,None]*(x@node['partners'])
    amplitudes=scalar*norm*signs
    direct=scalar@c.T;rebuilt=amplitudes@(unit*signs).T
    replay=float((direct-rebuilt).norm()/direct.norm())
    strength=-amplitudes.reshape(3,n,2)
    paired=torch.stack((strength[0,:,0]-strength[1,:,0],strength[1,:,1]-strength[0,:,1]))
    mean=paired.mean(1);sd=paired.std(1,correction=1);standardized=mean/sd
    indices=torch.randint(n,(2000,n),generator=torch.Generator().manual_seed(4901))
    intervals=torch.quantile(paired[:,indices].mean(2),torch.tensor([.025,.975]),dim=1)
    valid=replay<=1e-9 and bool(torch.isfinite(amplitudes).all()) and bool((sd>0).all())
    result=dict(pred_a=valid,pred_b=valid and bool((standardized>=.25).all()),
        pred_c=valid and bool((intervals[0]>0).all()),prefix_units=n,label=label,
        prospective_validation=label in ('separate','corpus_shift'),coefficient_replay=replay,
        paired_mean_suppression_difference=mean.tolist(),paired_standardized_difference=standardized.tolist(),
        paired_mean_95=intervals.tolist(),family_mean_suppression=strength.mean(1).tolist(),
        oriented_writer_family_loadings=(orientation*signs).tolist(),
        all_context_amplitude_correlation=float(torch.corrcoef(amplitudes.T)[0,1]),
        within_family_amplitude_correlations=[float(torch.corrcoef(strength[i].T)[0,1]) for i in range(3)],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,panel_path,cache_path)},
        scope='Exact frozen polynomial amplitudes, paired historical-prefix diagnostic. '
              'Standardization is a reported statistic, not fitted model coefficients. '
              'No behavioral intervention, OOD, sufficiency, or semantic identification claim.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':main(sys.argv[1])
