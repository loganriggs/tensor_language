"""CPU training-weight scatter audit for possible oblique dictionary extension."""
import json
from pathlib import Path
import time
import torch
from audit_native_reader_metric_v1 import P,CK,digest
from tyler_reader_shape_v1 import tyler_shape,trace_normalize,normalize_columns,roots
from orthogonal_reader_msp_v1 import polar


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter()
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    l,r=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')]
    order=torch.randperm(4608,generator=torch.Generator().manual_seed(700))[:3072]
    y=normalize_columns(torch.cat((l[order],r[order])).T)
    ordinary=trace_normalize(y@y.T/y.shape[1])
    shape,optimization=tyler_shape(y,max_steps=500,tolerance=1e-11,max_seconds=120)
    square,inverse=roots(shape)
    root_error=float((square@inverse-torch.eye(len(shape))).norm()/len(shape)**.5)
    values=torch.linalg.eigvalsh(ordinary)
    ordinary_condition=float(values[-1]/values[0])
    difference=float((shape-ordinary).norm()/ordinary.norm())
    torch.manual_seed(731)
    gaussian=normalize_columns(torch.randn_like(y))
    gaussian_cov=trace_normalize(gaussian@gaussian.T/gaussian.shape[1])
    gaussian_values=torch.linalg.eigvalsh(gaussian_cov)
    gaussian_condition=float(gaussian_values[-1]/gaussian_values[0])
    # Regenerate the prior known planted dictionary; no fitting or new outcome choice.
    torch.manual_seed(728);d=24
    truth=polar(torch.randn(d,d))@torch.diag(torch.logspace(0,torch.log10(torch.tensor(50.)).item(),d))@polar(torch.randn(d,d))
    normalized_truth=normalize_columns(truth)
    orthogonal_ceiling=float(torch.linalg.svdvals(normalized_truth).sum()/d)
    best_orthogonal=polar(normalized_truth)
    attained=float((best_orthogonal*normalized_truth).sum()/d)
    cache=Path('/dev/shm/bilin18_native_reader_shape_v1.pt')
    assert not cache.exists()
    torch.save(dict(tyler_shape=shape,ordinary_shape=ordinary,training_products=order,
        square_root=square,inverse_root=inverse,normalization='native reader unit directions; training products only'),cache)
    result=dict(predictions={
        'pred_a_converged_positive_shape':optimization['converged'] and optimization['minimum_eigenvalue']>1e-8 and root_error<=1e-8,
        'pred_b_robust_shape_differs':difference>=.05,
        'pred_c_nontrivial_covariance_condition':ordinary_condition>=10},
        optimization=optimization,root_identity_relative_error=root_error,
        robust_vs_ordinary_relative_difference=difference,
        ordinary_shape_condition=ordinary_condition,matched_gaussian_condition=gaussian_condition,
        gaussian_scope='One dimension/sample-count matched isotropic draw, not a confidence interval or native null distribution.',
        planted_orthogonal_mean_atom_cosine_ceiling=orthogonal_ceiling,
        planted_polar_attainment=attained,planted_ceiling_replay_error=abs(attained-orthogonal_ceiling),
        training_vectors=y.shape[1],heldout_weights_accessed=False,
        cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size),
        source_sha256=digest(__file__),seconds=time.perf_counter()-started,
        gpu_access=False,corpus_access=False,
        scope='Native weight geometry only. Whitening is not dictionary identification; sparsity/extraction/OOD remain untested.')
    with (P/'NATIVE_READER_SHAPE_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='optimization'},indent=2))
    print(json.dumps({k:v for k,v in optimization.items() if k!='history'}))


if __name__=='__main__':main()
