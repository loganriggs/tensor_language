from pathlib import Path
import json
p=Path(__file__).resolve().parent
d=1152;k=4608
native=dict(products=2*k,stored_weight_coefficients=3*d*k,linear_coefficient_multiplications=5*d*k,linear_additions=4*k*(d-1)+k+d*(k-1),description='Four input projections reusing L/R across n and m, channel cross-term sum, one dense residual output projection; homogeneous source-dependent midpoint function.')
records=[]
for outputs in [4,16,64,256]:
 for rank in [1,4,16]:
  products=outputs*rank;inputs=2*d*products;writers=d*outputs;adds=2*products*(d-1)+outputs*(rank-1)+d*(outputs-1)
  records.append(dict(outputs=outputs,rank=rank,products=products,stored_weight_coefficients=inputs+writers,linear_coefficient_multiplications=inputs+writers,linear_additions=adds,mean_and_offset_coefficients=d+outputs,product_reduction=native['products']/products,weight_storage_reduction=native['stored_weight_coefficients']/(inputs+writers),linear_multiplication_reduction=native['linear_coefficient_multiplications']/(inputs+writers),dense_evaluator_readout_entries=products*outputs,nonzero_grouping_readout_entries=products))
out=p/'MIDPOINT_COVERAGE_COSTS_V1.json';assert not out.exists();out.write_text(json.dumps(dict(native=native,candidates=records,scope='Arithmetic of a standalone two-input midpoint function with residual-width output. Upstream n,m generation, normalization, nonlinearities and full-model execution excluded. Candidate residual writers precontracted offline; group sums use implicit unit coefficients. Dense tensor storage/eager grouped-readout matmul are not compiled-away in current diagnostic evaluator, so these operation counts are a proposed implementation, not measured runtime.'),indent=2)+'\n')
print(json.dumps([v for v in records if v['outputs']==256 and v['rank']==4],indent=2))
