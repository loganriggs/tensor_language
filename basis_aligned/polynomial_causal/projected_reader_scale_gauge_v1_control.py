"""Exact reciprocal product scaling versus coordinate-dependent stationarity."""
import json
from pathlib import Path
import torch
from projected_sparse_dictionary_v1 import value_gradient
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(925)
    native=(torch.randn(7,5),torch.randn(7,5),torch.randn(4,7));white=torch.randn(7,4)
    total=dense(native[0],native[1],white@native[2]).square().sum()
    raw=torch.randn(8,5);values=torch.randn(12,3)
    ids=torch.stack([torch.randperm(8)[:3] for _ in range(12)]);scale=torch.rand(12)+.5
    rows=[];references=[]
    for multiplier in (1.,1000.):
        changed=values.clone();changed[:6]*=multiplier;changed[6:]/=multiplier
        loss,(gb,gc),writer,details=value_gradient(native,white,raw,ids,changed,scale,total)
        row_norm=changed.norm(dim=1);assert bool((row_norm>0).all())
        quotient=float((gc.norm(dim=1)*row_norm).norm()/loss.abs().clamp_min(1e-12))
        rows.append(dict(multiplier=multiplier,loss=float(loss),
            global_code_stationarity=details['codes_relative_stationarity'],rowwise_code_stationarity=quotient,
            dictionary_stationarity=details['dictionary_relative_stationarity'],writer_solve=details['writer_solve']))
        references.append((gb,writer))
    loss_error=abs(rows[0]['loss']-rows[1]['loss'])
    writer_error=float((references[0][1]-references[1][1]).norm()/references[0][1].norm())
    feature_error=float((references[0][0]-references[1][0]).norm()/references[0][0].norm())
    quotient_error=abs(rows[0]['rowwise_code_stationarity']-rows[1]['rowwise_code_stationarity'])/rows[0]['rowwise_code_stationarity']
    inflation=rows[1]['global_code_stationarity']/rows[0]['global_code_stationarity']
    result=dict(predictions=dict(pred_a_same_function_chart=max(loss_error,writer_error,feature_error)<=1e-8,
        pred_b_rowwise_invariance=quotient_error<=1e-8,pred_c_global_inflation=inflation>=1e4),arms=rows,
        loss_replay=loss_error,writer_replay=writer_error,feature_gradient_replay=feature_error,
        rowwise_relative_error=quotient_error,global_metric_inflation=inflation,
        scope='Small exact reciprocal Left/Right gauge. Does not change or rescue native convergence; feature gradient can independently remain nonzero. Rowwise metric needs nonzero rows and is not a global optimum certificate.')
    with Path(__file__).with_name('PROJECTED_READER_SCALE_GAUGE_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(result['predictions'].values())


if __name__=='__main__':main()
