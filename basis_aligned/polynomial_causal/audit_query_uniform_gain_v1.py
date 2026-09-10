"""Conditional native-bank homogeneity audit; no new model forward."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import source_gain_attention as M

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'BILIN18_L9_QUERY_PARTITION_NATIVE_V1_RESULT.json'
OUT=ROOT/'QUERY_UNIFORM_GAIN_V1_AUDIT.json'


if __name__=='__main__':
    torch.set_num_threads(2);r=json.loads(SOURCE.read_text());assert r['predictions']['pred_a_instrument']
    path=Path(r['bank_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['bank_sha256']
    panels={}
    with np.load(path,allow_pickle=False) as arrays:
        weight=torch.from_numpy(arrays['output_weight'])
        for panel in r['reports']:
            for side in ('base','donor'):
                prefix=panel+'__'+side+'__'
                bank={key:torch.from_numpy(arrays[prefix+key]) for key in ('factors','norm','gram','values')}
                bank.update(eps2=float(arrays[prefix+'eps2']),width=int(arrays[prefix+'width']))
                z=torch.ones(bank['gram'].shape[:2],dtype=torch.float64)
                write=lambda gain:torch.einsum('bjh,djh->bd',M.evaluate(bank,gain),weight)
                y=write(z);yn=float(y.norm());errors={str(scale):float((write(scale*z)-y).norm())/yn for scale in (.5,2.)}
                ds=torch.einsum('bs,bijst,bt->bij',z,bank['norm'],z)+bank['eps2']
                derivative=M.evaluate(bank,z)*bank['eps2']*(1/ds[:,0]+1/ds[:,1])[:,:,None]
                euler=torch.einsum('bjh,djh->bd',derivative,weight)
                zero=float(write(torch.zeros_like(z)).norm())
                panels[panel+'__'+side]={'native_write_norm':yn,'uniform_gain_relative_errors':errors,'analytic_uniform_gain_derivative_relative_norm':float(euler.norm())/yn,'zero_gain_write_norm':zero}
                assert max(errors.values())<1e-10 and zero==0.
    result={'passed':True,'panels':panels,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'bank_sha256':r['bank_sha256'],'model_forwards':0,'scope':'Conditional native query-read banks with native keys/values fixed. Positive uniform source gains nearly cancel in numerator/denominator, while zero kills the read. Does not attribute final-logit interactions or imply semantic redundancy.'}
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'max_uniform_gain_relative_error':max(max(v['uniform_gain_relative_errors'].values()) for v in panels.values()),'max_analytic_uniform_derivative':max(v['analytic_uniform_gain_derivative_relative_norm'] for v in panels.values()),'model_forwards':0}))
