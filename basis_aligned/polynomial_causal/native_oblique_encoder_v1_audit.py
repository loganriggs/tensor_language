"""Fixed native oblique basis: sparse support and penalty-scaling diagnostic."""
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P,CK
from lasso_reader_encoding_v1 import encode

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    parent=json.loads((P/'OBLIQUE_READER_DICTIONARY_V1_ordinary_covariance_SEED_0.json').read_text())
    path=Path(parent['cache']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==parent['cache']['sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu');basis=saved['analysis_basis']
    heldout=saved['order'][3072:]
    selected=heldout[torch.randperm(len(heldout),generator=torch.Generator().manual_seed(832))[:128]]
    ids=torch.cat((selected,selected+4608))
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    readers=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])[ids]
    norms=readers.norm(dim=1)
    parent_ids=saved['code_indices'][ids].long();parent_values=saved['code_values'][ids]
    before=torch.einsum('nk,nkd->nd',parent_values,basis[parent_ids])
    before_error=((before-readers)/norms[:,None]).square().sum(1)
    baseline=1-float(before_error.mean());rows=[]
    for name,penalty in [('fixed',.05),('dimension_scaled',.05*math.sqrt(12/1152))]:
        started=time.perf_counter();indices,values,report=encode(basis,readers,128,penalty)
        after=torch.einsum('nk,nkd->nd',values,basis[indices])
        errors=((after-readers)/norms[:,None]).square().sum(1)
        gain=before_error-errors
        row=dict(name=name,penalty=penalty,seconds=time.perf_counter()-started,encoding=report,
            capture=1-float(errors.mean()),gain=float(gain.mean()),fraction_improved=float((gain>0).double().mean()),
            median_gain=float(gain.median()),minimum_gain=float(gain.min()),maximum_gain=float(gain.max()))
        rows.append(row);print(json.dumps(row),flush=True)
    predictions=dict(pred_a_encoding=all(r['encoding']['converged'] and r['encoding']['support_ls_normal_residual']<=1e-8 for r in rows),
        pred_b_reader_gain=any(r['gain']>=.02 for r in rows),
        pred_c_scaled_gain=rows[1]['capture']>=rows[0]['capture']+.01)
    result=dict(predictions=predictions,baseline_capture=baseline,arms=rows,selected_products=selected.tolist(),
        source=parent['cache'],scope='Post-result subset diagnostic of frozen native weights; no new basis, corpus fit, full-U score or circuit validation')
    (P/'NATIVE_OBLIQUE_ENCODER_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(predictions=predictions,baseline_capture=baseline)),flush=True)

if __name__=='__main__':main()
