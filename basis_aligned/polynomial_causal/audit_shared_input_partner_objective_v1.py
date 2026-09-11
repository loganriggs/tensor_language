"""Cross-evaluate fixed native readers under the conditional low-rank partner objective."""
import json,hashlib,time
from pathlib import Path
import torch
from audit_token_dictionary_debias_v1 import P,CK


def main():
    torch.set_num_threads(2);start=time.perf_counter()
    receipt=json.loads((P/'SHARED_INPUT_FACTOR_NATIVE_V1_RESULT.json').read_text())
    cache=Path(receipt['cache']['path']);assert hashlib.sha256(cache.read_bytes()).hexdigest()==receipt['cache']['sha256']
    saved=torch.load(cache,weights_only=True,map_location='cpu')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double();rows={}
    eye=torch.eye(1152,dtype=torch.float64)
    for metric,output in [('full',u),('centered',u-u.mean(0))]:
        root=torch.linalg.cholesky(output.T@output);total=receipt['objectives'][metric]['total_energy'];rows[metric]={}
        for name,state in saved.items():
            a=state['readers'][state['best']];m=state['partner'];j=eye+(2**.5-1)*a[:,None]*a[None,:]
            sv=torch.linalg.svdvals(root.T@m@j);squares=sv.square()/2
            rows[metric][name]=dict(unrestricted_partner_native_capture=float(squares.sum()/total),
                                   rank16_partner_native_capture=float(squares[:16].sum()/total),
                                   rank16_share_of_component=float(squares[:16].sum()/squares.sum()))
    delta=rows['centered']['full']['rank16_partner_native_capture']-rows['centered']['centered']['rank16_partner_native_capture']
    result=dict(comparisons=rows,centered_rank16_gain_using_full_reader=delta,
                alternative_reader_improves_centered_rank16=delta>1e-8,
                wall_seconds=time.perf_counter()-start,
                scope='Fixed readers only; exact conditional partner rank16 optimum. The source D miss does not test joint reader-plus-rank16 optimization. No behavioral result.')
    (P/'SHARED_INPUT_PARTNER_OBJECTIVE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
