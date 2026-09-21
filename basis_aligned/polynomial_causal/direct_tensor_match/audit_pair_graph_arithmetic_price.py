"""Literal source-stage arithmetic; activation products distinguished from coefficient multiplies.
No timing/speedup claim. Fixed affine/later-component interfaces excluded equally.
"""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2)
programs=torch.load(P/'COST_MATCHED_PAIR_BASELINES_V1.pt',weights_only=True);rows=[]
for key,bundle in programs.items():
 projection_mult=sum(p['shared_reader'].numel() for p in bundle.values());projection_add=sum(p['shared_reader'].shape[1]*(p['shared_reader'].shape[0]-1) for p in bundle.values());readout_mult=sum(p['product_weights'].numel() for p in bundle.values());readout_add=sum(p['product_weights'].shape[1]*(p['product_weights'].shape[0]-1) for p in bundle.values());products=sum(len(p['product_weights']) for p in bundle.values());extra_add=sum(2*int((p['product_indices'][2]==1).sum()) for p in bundle.values())
 rows.append(dict(program=key,activation_products=products,projection_coefficient_multiplications=projection_mult,readout_coefficient_multiplications=readout_mult,total_source_multiplications=projection_mult+readout_mult+products,source_additions=projection_add+readout_add+extra_add,pair_block_extra_additions=extra_add,index_int64_values=sum(p['product_indices'].numel() for p in bundle.values())))
for width in [367,560]:
 leaves=2*width+32;projection=1152*leaves;readout=6*width+32;products=width+32
 rows.append(dict(program=f'shared_mixed_{width}_plus32',activation_products=products,projection_coefficient_multiplications=projection,readout_coefficient_multiplications=readout,total_source_multiplications=projection+readout+products,source_additions=leaves*1151+6*(width-1)+32,index_int64_values=1))
out=dict(records=rows,scope='Literal scalar arithmetic for dense source projections, distinct activation products and output mixtures, before common affine/normalization/later-state operations. Both additions/subtractions count one. Constant multiplications retained; no fused hardware timing inferred. Baseline index arrays counted separately from float coefficients.')
(P/'PAIR_GRAPH_ARITHMETIC_PRICE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
