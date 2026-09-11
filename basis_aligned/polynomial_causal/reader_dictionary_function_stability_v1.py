"""Compare frozen MSP functions rather than raw basis coordinates."""
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P,CK

def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter()
    report=json.loads((P/'FULL_READER_DICTIONARY_MSP_V1_RESULT.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    metric=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    m=torch.load(metric['cache']['path'],weights_only=True,map_location='cpu')['unembedding_gram']
    writer=d.T@m@d
    def inner(left,right):
        la,ra=left;lb,rb=right
        return float((writer*((la@lb.T)*(ra@rb.T)+(la@rb.T)*(ra@lb.T))).sum()/2)
    functions=[]
    for row in report['starts']:
        saved=torch.load(row['cache']['path'],weights_only=True,map_location='cpu')
        code=torch.zeros(9216,1152).scatter_(1,saved['code_indices'].long(),saved['code_values'])
        read=code@saved['analysis_basis'];functions.append(read.split(4608))
    total=metric['native_total'];energies=[inner(f,f) for f in functions]
    replay=[1-(energy-2*inner((l,r),f)+total)/total for f,energy in zip(functions,energies)]
    errors=[abs(value-row['scores']['coefficient_capture']) for value,row in zip(replay,report['starts'])]
    cross=inner(*functions);cosine=cross/(energies[0]*energies[1])**.5
    result=dict(predictions=dict(pred_a_replay=max(errors)<=1e-8 and min(energies)>0,pred_b_function_stability=cosine>=.9),
        capture_replay=replay,replay_errors=errors,mean_atom_alignment=report['atom_alignment']['mean'],
        function_cosine=cosine,squared_function_difference_over_native=(energies[0]+energies[1]-2*cross)/total,
        seconds=time.perf_counter()-started,scope='Two frozen coefficient functions, not semantic-unit identity or OOD behavior')
    (P/'READER_DICTIONARY_FUNCTION_STABILITY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
