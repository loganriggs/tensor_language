"""Separate common vocabulary shifts from relative-token output contrasts."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);D=torch.load(P/'CANONICAL_VOCABULARY_V1.pt',weights_only=True)['directions'].double();view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);energy=view['mode_variances'];common=D.sum(0)/len(D)**.5;fraction=float((common.square()*energy).sum()/energy.sum());centered=D-D.mean(0);U,sv,_=torch.linalg.svd(centered*energy.sqrt(),full_matrices=False);mix=U.T@centered;torch.manual_seed(2120);g=torch.randn(17,4,dtype=torch.float64);direct=g@centered.T;replay=(g@mix.T)@U.T;error=float((direct-replay).norm()/direct.norm());assert error<1e-6
 out=dict(common_shift_energy_fraction=fraction,contrast_energy_fraction=1-fraction,original_mode_common_shift_fractions=common.square().tolist(),contrast_mode_energy_fractions=(sv.square()/sv.square().sum()).tolist(),centered_interpretation_replay=error,scope='Diagnostic vocabulary centering only, not a model change. Uses Gaussian canonical feature variances. Common shifts before final softcap need not leave probabilities invariant. Token rankings and spectral stability alone do not establish semantic circuits.');torch.save(dict(output_directions=U,feature_mixing=mix@view['feature_mixing'],root_mean=view['root_mean'],mode_variances=sv.square()),P/'CANONICAL_CONTRAST_VIEW_V1.pt');(P/'CANONICAL_CONTRAST_VIEW_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
