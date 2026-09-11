"""A midpoint decomposition<=1e-10 B readerterm>=.75total,writerterm<=.5total.
Only weight-metric sign alignment, no fitting or native replication repair.
"""
import json
from pathlib import Path
import torch
from sparse_path_program_v1 import run
from sparse_path_stability_atlas_v1 import digest


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);p=Path(__file__).parent;out=p/'STABLE_PATH_REPLICA_ACCOUNTING_V1.json';assert not out.exists()
    old=json.loads((p/'STABLE_PATH_NATIVE_CACHE_V1.json').read_text());ap=p/'STABLE_PATH_BANK_V1.pt';assert digest(ap)==old['source_bank_sha256'];assert old['pred_a']
    programs=torch.load(ap,weights_only=True,map_location='cpu')['programs']
    binding=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double();w0,w1=[a['physical_writer'] for a in programs];uw0,uw1=u@w0,u@w1
    sign=(uw0*uw1).sum(0).sign();assert (sign!=0).all();w1=w1*sign
    up=torch.load(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt',weights_only=True,map_location='cpu')['ports'];cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports']
    den=cache['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    a0,a1=[run(q,up['residual'].double(),up['head_values'].double(),den)['amplitudes']/den[:,None] for q in programs];a1=a1*sign
    delta=a0@w0.T-a1@w1.T
    reading=(a0-a1)@((w0+w1)/2).T;writing=((a0+a1)/2)@(w0-w1).T
    error=float((reading+writing-delta).norm()/delta.norm());r=float(reading.norm()/delta.norm());w=float(writing.norm()/delta.norm())
    centered0=a0-a0.mean(0);centered1=a1-a1.mean(0)
    cor=(centered0*centered1).sum(0)/(centered0.norm(dim=0)*centered1.norm(dim=0)).clamp_min(1e-30)
    relative=(a0-a1).norm(dim=0)/((a0.square().sum(0)+a1.square().sum(0))/2).sqrt().clamp_min(1e-30)
    result=dict(pred_a=error<=1e-10,pred_b=r>=.75 and w<=.5,identity_error=error,reading_term_relative_norm=r,writing_term_relative_norm=w,
        reading_writing_cosine=float((reading*writing).sum()/reading.norm()/writing.norm()),amplitude_centered_correlations=cor.tolist(),amplitude_symmetric_relative_rms=relative.tolist(),
        source_native_result_sha256=digest(p/'STABLE_PATH_NATIVE_CACHE_V1.json'),scope='Exact physical-write difference attribution under fixed coefficient-normalized feature gauges and output sign alignment. Norm ratios not variance fractions. No data-fit correction, behavioral repair or absent-structure claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']

if __name__=='__main__':main()
