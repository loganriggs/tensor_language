"""Saved-matrix numerical checks and scope audit of the reader-family ceiling."""
import hashlib, json, time
from pathlib import Path
import torch


def main():
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    started=time.perf_counter(); p=Path(__file__).resolve().parent
    result=json.loads((p/'SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json').read_text())
    cache=result['cache']; assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    saved=torch.load(cache['path'],weights_only=True,map_location='cpu'); rows={}
    for name,state in saved.items():
        s,e,v=[state[k] for k in ['second','eigenvalues','eigenvectors']]
        total=float(s.trace()); residual=float((s@v-v*e[None,:]).norm())
        orth=float((v.T@v-torch.eye(len(v))).norm())
        eleven=next(x for x in result['metrics'][name]['curve'] if x['rank']==11)
        torch.manual_seed(1349); change=torch.randn(11,11)
        u,_,vh=torch.linalg.svd(change); change=u@torch.diag(torch.linspace(1.,10.,11))@vh
        frame=v[:,-11:]; changed=torch.linalg.qr(frame@change).Q
        projector_error=float((frame@frame.T-changed@changed.T).norm())
        # Numerical margin check, not interval-arithmetic certification.
        padded=min(1.,2*(float(e[-11:].sum())+11*residual)/total)
        rows[name]=dict(eigen_residual_over_total=residual/total,
            full_basis_orthogonality_error=orth,minimum_eigenvalue=float(e.min()),
            nonorthogonal_basis_projector_error=projector_error,
            rank11_reported_bound=eleven['universal_rank_bound'],
            rank11_residual_padded_bound=padded,
            maximum_relative_improvement_over_spectral=eleven['universal_rank_bound']/eleven['capture']-1,
            rank11_half_capture_ruled_out=padded<.5,
            rank128_mixed_fraction=next(x for x in result['metrics'][name]['curve'] if x['rank']==128)['mixed'])
    price_per_reader=1152+1152**2; native=3*1152*4608
    output=dict(instrument_passed=all(r['eigen_residual_over_total']<1e-10 and
                r['full_basis_orthogonality_error']<1e-10 and r['nonorthogonal_basis_projector_error']<1e-10 and r['minimum_eigenvalue']>0 for r in rows.values()),
        metrics=rows,largest_dense_partner_reader_count_below_native=(native-1)//price_per_reader,
        native_weight_numbers=native,body_forwards=0,corpus_access=False,
        source_cache_sha256=cache['sha256'],wall_seconds=time.perf_counter()-started,
        redteam=dict(narrow_claim='Low-dimensional GLOBAL shared reader families cannot capture most Frobenius coefficient energy, even with unrestricted mixed partners.',
            optimizer_failure_explanation='Not applicable to the universal ceiling; spectral frame is not claimed stationary for the extraction objective.',
            nonorthogonal_reader_explanation='Any rank-r span admits an orthonormal basis; native rank11 projector checked after condition10 re-basing.',
            metric_limit='These ceilings do not apply to natural-state or causal-effect weighting. No data-based structural negative.',
            surviving_structure='Many readers with sparse/structured partners; local groups; cross-layer producer sharing; low-energy behavioral components.'),
        scope='Numerical eigen-residual and representation/price audit; not a formal floating-point interval certificate or absence-of-circuits theorem.')
    with (p/'SHARED_INPUT_SUBSPACE_NATIVE_V1_REDTEAM.json').open('x') as f:
        json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2));assert output['instrument_passed']


if __name__=='__main__':main()
