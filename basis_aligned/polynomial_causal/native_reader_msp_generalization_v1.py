"""Weight-only audit of the frozen first full-reader MSP fit."""
import hashlib
import json
from pathlib import Path
import time
import torch

P=Path(__file__).parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter()
    report=json.loads((P/'FULL_READER_DICTIONARY_MSP_V1_SEED_0.json').read_text())
    path=Path(report['cache']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==report['cache']['sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    y=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])
    y/=y.norm(dim=1,keepdim=True)
    order=saved['order']
    train=torch.cat((order[:3072],order[:3072]+4608))
    test=torch.cat((order[3072:],order[3072:]+4608))
    _,vectors=torch.linalg.eigh(y[train].T@y[train])
    result={}
    for name,basis in [('identity',torch.eye(1152)),('pca',vectors.T),('msp',saved['analysis_basis'])]:
        coordinates=y@basis.T
        result[name]={}
        for split,ids in [('train',train),('test',test)]:
            z=coordinates[ids];energy=z.pow(4)
            atom_sum=energy.sum(0)
            effective=atom_sum.square()/energy.square().sum(0).clamp_min(1e-30)
            result[name][split]=dict(fourth_moment=float(energy.sum()/len(ids)),
                top128_energy=float(z.square().topk(128,dim=1).values.sum()/len(ids)),
                near_alignment_share=float(energy[z.abs()>=.9].sum()/energy.sum()),
                fraction_readers_with_alignment_over_point9=float((z.abs().max(1).values>=.9).double().mean()),
                median_atom_effective_readers=float(effective.median()),
                weighted_atom_effective_readers=float((effective*atom_sum).sum()/atom_sum.sum()))
    learned=result['msp'];errors=[abs(learned['train']['top128_energy']-report['scores']['train_top128_energy']),
        abs(learned['test']['top128_energy']-report['scores']['test_top128_energy']),
        abs(learned['train']['fourth_moment']-report['optimization']['final']['objective'])]
    predictions=dict(pred_a_replay=max(errors)<=1e-10,
        pred_b_objective_gap=learned['train']['fourth_moment']-learned['test']['fourth_moment']>=.01,
        pred_c_alignment=learned['train']['near_alignment_share']>=.25,
        pred_d_alignment_gap=learned['train']['near_alignment_share']-learned['test']['near_alignment_share']>=.10)
    output=dict(predictions=predictions,bases=result,replay_errors=errors,source=report['cache'],
        seconds=time.perf_counter()-started,scope='Frozen first-start weight generalization, no basis or text fitting')
    (P/'NATIVE_READER_MSP_GENERALIZATION_V1_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
