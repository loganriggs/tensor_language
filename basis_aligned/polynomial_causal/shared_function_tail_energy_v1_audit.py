"""Cached natural-state tail importance; no fitting, forwards or new selection."""
import hashlib,json
from pathlib import Path
import torch


def main():
    p=Path(__file__).resolve().parent;out=p/'SHARED_FUNCTION_TAIL_ENERGY_V1_AUDIT.json';assert not out.exists()
    parent=json.loads((p/'SHARED_FUNCTION_FINEWEB_V1_AUDIT.json').read_text());source=parent['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    data=torch.load(source['path'],weights_only=True,map_location='cpu')
    y=data['native'];p4=data['predictions'][2];p16=data['predictions'][4]
    e4=y-p4;e16=y-p16;tail=p16-p4;total=y.square().sum()
    values=dict(error4=float(e4.square().sum()/total),error16=float(e16.square().sum()/total),
        tail_energy=float(tail.square().sum()/total),cross=float(2*(e16*tail).sum()/total))
    replay=abs(values['error4']-values['error16']-values['tail_energy']-values['cross'])
    centered_reduction=1-float((e16-e16.mean()).square().sum()/(e4-e4.mean()).square().sum())
    weight=json.loads((p/'SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json').read_text())
    coefficient_fraction=weight['rows'][4]['capture']-weight['rows'][2]['capture']
    amplification=values['tail_energy']/coefficient_fraction
    result=dict(predictions=dict(pred_a_identity=replay<=1e-10,pred_b_not_mean_only=centered_reduction>=.75,
        pred_c_tail_amplification=amplification>=10),energies=values,energy_replay=replay,
        centered_error_reduction=centered_reduction,tail_coefficient_fraction=coefficient_fraction,
        natural_to_coefficient_energy_ratio=amplification,source=source,
        scope='Post-result diagnosis on cached FineWeb scalar outputs. Not data-guided factor discovery or a repair of original one/four-product misses.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
