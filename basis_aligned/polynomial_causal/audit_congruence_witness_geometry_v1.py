"""CPU geometry audit of converged V1 matrices, not a native block claim."""
import json,hashlib
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    receipt=json.loads((P/'NATIVE_CONGRUENCE_SPECTRUM_V1_RESULT.json').read_text())
    cache=Path(receipt['witness_cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['witness_cache']['sha256']
    vectors=torch.load(cache,weights_only=True,map_location='cpu')['vectors']
    rows=[]
    for index,z in enumerate(vectors):
        values=torch.linalg.svdvals(z)
        energy=z.square().sum()
        rows.append(dict(index=index,eigenvalue=receipt['solver']['eigenvalues'][index],
            eigenvalue_over_exact_mean=receipt['solver']['eigenvalues'][index]/receipt['solver']['mean_nontrivial_eigenvalue'],
            singular_energy_capture={str(k):float(values[:k].square().sum()/energy) for k in [1,2,4,16,64,256]},
            symmetric_energy_fraction=float(((z+z.T)/2).square().sum()/energy),
            square_norm_over_squared_norm=float((z@z).norm()/energy),
            nonnormality=float((z@z.T-z.T@z).norm()/energy)))
    result=dict(rows=rows,scope='Geometry of three individually converged FP64 witness matrices. Four-mode native search remains incomplete. Concentration or nonnormality does not establish a block partition or behavior.')
    (P/'NATIVE_CONGRUENCE_WITNESS_GEOMETRY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
