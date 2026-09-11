"""Does loading concentration survive the nonorthogonal function Gram metric?"""
import json,hashlib,time
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross
from audit_token_dictionary_debias_v1 import P,CK


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'PRODUCT_WRITER_REPAIR_V2_RESULT.json').read_text())
    cache=Path(receipt['cache']['path']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu')
    root=saved['root'];pl,pr=saved['left'],saved['right'];scale=saved['scale']
    g=product_cross(pl,pr,pl,pr);sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    u=sd['lm_head.weight'].double();p=torch.linalg.solve_triangular(root,u.T,upper=False).T
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ['Left','Right','Down']]
    native_cross=product_cross(l,r,pl,pr)
    c=root.T@d@native_cross/scale
    centered_target_cross=(u-u.mean(0))@(d@native_cross/scale)
    baseline=torch.linalg.solve(g,c.T).T
    rows={}
    for name,z in [('unpenalized',baseline),('converged_low_penalty',saved['whitened_writers']['0.1'])]:
        a=p@z;a-=a.mean(0)
        indices=a.abs().topk(16,dim=1).indices
        selected=torch.zeros_like(a);selected.scatter_(1,indices,a.gather(1,indices))
        remainder=a-selected
        norms=((a@g)*a).sum(1);error=((remainder@g)*remainder).sum(1)
        loading=(selected.square().sum(1)/a.square().sum(1).clamp_min(1e-30))
        fraction=1-error/norms.clamp_min(1e-30)
        rows[name]=dict(median_top16_loading_share=float(loading.median()),
                       aggregate_top16_function_capture=float(1-error.sum()/norms.sum()),
                       median_top16_function_capture=float(fraction.median()),
                       negative_function_capture_tokens=int((fraction<0).sum()),
                       centered_approximation_function_energy=float(norms.sum()),
                       minimum_squared_function_norm=float(norms.min()),
                       full_writer_native_centered_capture=float((2*(a*centered_target_cross).sum()-norms.sum())/len(u)),
                       top16_writer_native_centered_capture=float((2*(selected*centered_target_cross).sum()-((selected@g)*selected).sum())/len(u)))
    result=dict(instrument_passed=all(v['minimum_squared_function_norm']>=0 for v in rows.values()),
                arms=rows,aggregate_function_capture_gain=rows['converged_low_penalty']['aggregate_top16_function_capture']-rows['unpenalized']['aggregate_top16_function_capture'],
                median_function_capture_gain=rows['converged_low_penalty']['median_top16_function_capture']-rows['unpenalized']['median_top16_function_capture'],
                wall_seconds=time.perf_counter()-start,
                scope='Top16 token-wise truncation is diagnostic and generally leaves native U writer space. Function-capture fields concern the fitted centered family; native-centered-capture fields use the true target. Neither measures behavior.')
    (P/'LEGAL_WRITER_FUNCTION_CONCENTRATION_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
