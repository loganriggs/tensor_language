"""Applicability of a sufficient fixed-dictionary sparse-code uniqueness bound."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    parent=json.loads((P/'OVERCOMPLETE_OLS_REENCODE_V1_RESULT.json').read_text());rows=[]
    for arm in parent['arms']:
        if arm['name']!='learned':continue
        source=arm['cache'];assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        b=saved['analysis_basis'].double();normerror=float((b.norm(dim=1)-1).abs().max());assert normerror<=1e-10
        gram=(b@b.T).abs();gram.fill_diagonal_(0)
        mu=float(gram.max());nearest=gram.max(1).values
        threshold=(1+1/mu)/2
        support=int((saved['code_values']!=0).sum(1).max())
        rows.append(dict(seed=arm['seed'],source=source,mutual_coherence=mu,
            median_nearest_absolute_cosine=float(nearest.median()),strict_sparsity_threshold=threshold,
            maximum_actual_support=support,sufficient_condition_holds=support<threshold,unit_norm_error=normerror))
    result=dict(arms=rows,
        theorem='For a fixed unit-atom dictionary, k < (1+1/mu)/2 suffices for unique sparsest exact codes and their basis-pursuit recovery.',
        scope='Failure of this sufficient bound does not imply nonunique codes. It does not certify the learned dictionary, noisy approximation, Lasso with nonzero penalty, folded product decomposition or circuits.')
    with (P/'READER_COHERENCE_CERTIFICATE_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
