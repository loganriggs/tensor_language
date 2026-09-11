"""Frozen readout centering/padding diagnostic; neither intervention nor refit."""
import hashlib,json
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    out=P/'CANCELLATION_TOKEN_CENTERING_V1_AUDIT.json';assert not out.exists()
    prior=json.loads((P/'CANCELLATION_TOKEN_READOUTS_V1_AUDIT.json').read_text());cache=prior['cache']
    assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    saved=torch.load(cache['path'],weights_only=True,map_location='cpu');v=saved['token_modes'];n=prior['tokenizer']['vocabulary_size']
    rows=[];worst=0.
    for j in range(8):
        original=v[:,j];mean=original.mean();center=original-mean
        energy=original.square().sum();partition=mean.square()*len(original)+center.square().sum()
        replay=float(abs(energy-partition)/energy);worst=max(worst,replay)
        valid=original[:n];validcenter=valid-valid.mean()
        rows.append(dict(mode=j,partition_replay=replay,
            centered_top128=float(center.square().topk(128).values.sum()/center.square().sum()),
            valid_centered_top128=float(validcenter.square().topk(128).values.sum()/validcenter.square().sum()),
            centered_effective_tokens=float(center.square().sum().square()/center.pow(4).sum()),
            centered_energy_fraction=float(center.square().sum()/energy)))
    weights=torch.tensor([r['group_coefficient_energy_share'] for r in prior['rows']])
    padding=torch.tensor([r['extra_output_energy'] for r in prior['rows']])
    aggregate=float((weights*padding).sum()/weights.sum())
    result=dict(predictions=dict(pred_a_partition=worst<=1e-10,
        pred_b_centered_sparse=sum(r['centered_top128']>=.25 for r in rows)>=6,pred_c_padding=aggregate<=.01),
        rows=rows,eight_mode_padding_fraction=aggregate,eight_mode_group_coverage=float(weights.sum()),source=cache,
        scope='Fixed top8 modes only; centering and excluding47extra rows are diagnostics. '
              'No output refit or evidence common direction is behaviorally removable through RMS/tanh.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
