"""Audit prediction against no change; no fitting or new model execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,hashlib
from pathlib import Path
import torch

P=Path(__file__).resolve().parent


def main():
    start=time.perf_counter();torch.set_num_threads(2)
    receipt=json.loads((P/'FROZEN_QK_FINEWEB_V1_RESULT.json').read_text());cache=receipt['cache']
    assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    saved=torch.load(cache['path'],weights_only=True,map_location='cpu');base=saved['logits']['baseline'].double()
    output={}
    for variant in ['learned','random']:
        physical=saved['logits']['physical_'+variant].double();predicted=saved['logits']['predicted_'+variant].double()
        effect=physical-base;residual=predicted-physical
        rows=[x for x in receipt['rows'] if x['variant']==variant]
        changes=torch.tensor([r['ce_change'] for r in rows],dtype=torch.float64)
        zero_mae=float(changes.abs().mean());pred_mae=receipt['summaries'][variant]['ce_prediction_absolute_error']
        attention=torch.cat([v['baseline'] for v in saved['attention']]).double()
        edited_attention=torch.cat([v['physical_'+variant] for v in saved['attention']]).double()
        output[variant]=dict(no_change_ce_mae=zero_mae,predictor_ce_mae=pred_mae,no_change_to_predictor_ce_error_ratio=zero_mae/pred_mae,
            no_change_passes_absolute_ce_bar=zero_mae<=1e-3,no_change_logit_change_relative_error=1.,
            predictor_logit_change_relative_error=float(residual.norm()/effect.norm()),
            median_absolute_ce_change=float(changes.abs().median()),maximum_absolute_ce_change=float(changes.abs().max()),
            rows_ce_increased=int((changes>0).sum()),rows_ce_decreased=int((changes<0).sum()),
            logit_change_rms=float(effect.square().mean().sqrt()),logit_prediction_residual_rms=float(residual.square().mean().sqrt()),
            attention_change_to_original_norm=float((edited_attention-attention).norm()/attention.norm()))
    result=dict(variants=output,source_cache_sha256=cache['sha256'],body_forwards=0,corpus_access=False,
                wall_seconds=time.perf_counter()-start,
                scope='Post-result nontriviality audit of a fixed single-source FineWeb validation. Absolute CE tolerance alone is weak for tiny effects; relative logit prediction distinguishes the compiled predictor from no change. No semantic or selective circuit identification.')
    with (P/'FROZEN_QK_FINEWEB_EFFECT_SIZE_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
