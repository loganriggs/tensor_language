"""Read one atomically published completed sweep; diagnostics are not convergence."""
import io,hashlib,json
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross


def main():
    torch.set_num_threads(2)
    path=Path('/dev/shm/bilin18_sparse_product_dictionary_v1.pt')
    snapshot=path.read_bytes();saved=torch.load(io.BytesIO(snapshot),weights_only=True,map_location='cpu')
    l,r,a=saved['left'],saved['right'],saved['codes']
    gram=product_cross(l,r,l,r);ev=torch.linalg.eigvalsh(gram)
    correlation=gram/gram.diag().sqrt()[:,None]/gram.diag().sqrt()[None,:]
    correlation.fill_diagonal_(0)
    counts=(a!=0).sum(0)
    result=dict(snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),iteration=saved['history'][-1]['iteration'],
                penalty=saved['penalty'],products=len(l),dead_atoms=int((counts==0).sum()),
                participation_rank=float(ev.sum().square()/ev.square().sum()),
                eigen_rank90=int(torch.searchsorted(ev.flip(0).cumsum(0),.9*ev.sum()))+1,
                pairs_abs_cosine_above_09=int((correlation.abs()>.9).sum())//2,
                pairs_abs_cosine_above_099=int((correlation.abs()>.99).sum())//2,
                median_maximum_abs_cosine=float(correlation.abs().max(1).values.median()),
                minimum_gram_eigenvalue=float(ev[0]),
                latest_objective=saved['history'][-1]['objective'],
                latest_capture=saved['history'][-1]['captured_energy'],
                scope='One completed live sweep snapshot, not terminal fit or initial-state measurement. Correlated product initialization is a plausible optimization limitation, not proved by this snapshot alone.')
    Path(__file__).with_name('LIVE_PRODUCT_COHERENCE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
