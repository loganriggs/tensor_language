"""Frozen FineWeb square versus full reader deletion; local native tail screen."""
import hashlib
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


@torch.no_grad()
def main():
    start=time.perf_counter();torch.set_num_threads(2)
    root=Path(__file__).parent
    artifact=root/'STABLE_SQUARE150_NATIVE_INTERFACE_V1.pt'
    prior=json.loads((root/'STABLE_SQUARE150_NATIVE_INTERFACE_V1_AUDIT.json').read_text())
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==prior['artifact_sha256']
    program=torch.load(artifact,weights_only=True,map_location='cpu');reader=program['reader'].double();writer=program['physical_writer'].double()
    path=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt')
    data_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    assert data_hash=='b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de'
    data=torch.load(path,weights_only=True,map_location='cpu');positions=[63,127]
    x=data['ports']['input'][:,positions].reshape(-1,1152).double()
    pre=data['ports']['pre'][:,positions].reshape(-1,1152).float()
    cached=data['ports']['native_output'][:,positions].reshape(-1,1152).float()
    targets=data['rows'][:,[p+1 for p in positions]].reshape(-1).long()
    pilot=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    ck=next(k for k in pilot['binding'] if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();unembedding=sd['lm_head.weight'].float()
    def mlp(inputs):return ((inputs@l.T)*(inputs@r.T))@d.T+bias
    def logits(outputs):return 30*torch.tanh((F.rms_norm(pre+outputs.float(),(1152,))@unembedding.T)/30)
    t=x@reader;square=t.square()[:,None]*writer
    native=mlp(x);projected=mlp(x-t[:,None]*reader)
    full=t[:,None]*(((l@reader)[None]*(x@r.T)+(r@reader)[None]*(x@l.T))@d.T)-square
    formula_error=float((native-projected-full).norm()/full.norm())
    reference=logits(native);cached_reference=logits(cached)
    precision=float((reference-cached_reference).norm()/cached_reference.norm())
    native_log=reference.double().log_softmax(-1);native_prob=native_log.exp()
    native_ce=-native_log.gather(1,targets[:,None]).squeeze(1)
    quote_ids=[366,705,1,6];control_ids=[11,13,26,25]
    rows=[];effects={};per_position={}
    for name,output in [('square_delete',native-square),('reader_delete',projected)]:
        scores=logits(output);log=scores.double().log_softmax(-1)
        ce=-log.gather(1,targets[:,None]).squeeze(1);kl=(native_prob*(native_log-log)).sum(-1)
        delta=scores.double()-reference.double()
        effects[name]=dict(quote=delta[:,quote_ids],control=delta[:,control_ids])
        per_position[name]=dict(ce_added=ce-native_ce,kl=kl,**effects[name])
        rows.append(dict(name=name,mean_ce_added=float((ce-native_ce).mean()),mean_abs_position_ce_change=float((ce-native_ce).abs().mean()),
                         mean_kl=float(kl.mean()),mean_quote_score_change=float(delta[:,quote_ids].mean()),
                         mean_abs_quote_score_change=float(delta[:,quote_ids].abs().mean()),
                         mean_abs_control_score_change=float(delta[:,control_ids].abs().mean())))
    sq=effects['square_delete']['quote'];actual=effects['reader_delete']['quote']
    error=float((sq-actual).norm()/actual.norm())
    positive=float((sq.mean(1)>0).double().mean())
    ratio=float(sq.abs().mean()/effects['square_delete']['control'].abs().mean())
    median=float(actual.square().mean(1).sqrt().median())
    tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    result=dict(pred_a=formula_error<=1e-8 and precision<=1e-5,pred_b=error<=.1,pred_c=positive>=.9 and ratio>=2,pred_d=median>=.01,
                native_formula_replay=formula_error,native_cached_tail_precision=precision,
                quote_effect_prediction_relative_error=error,positive_quote_position_fraction=positive,
                quote_to_punctuation_absolute_effect_ratio=ratio,median_full_reader_quote_effect_rms=median,
                square_vs_full_reader_output_relative_error=float((square-full).norm()/full.norm()),
                square_to_full_reader_output_energy=float(square.square().sum()/full.square().sum()),
                scalar_abs_quantiles=torch.quantile(t.abs(),torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist(),
                quote_tokens=[dict(id=i,text=tokenizer.decode([i])) for i in quote_ids],
                control_tokens=[dict(id=i,text=tokenizer.decode([i])) for i in control_ids],
                quote_target_positions=int(torch.isin(targets,torch.tensor(quote_ids)).sum()),
                prediction_positions=len(targets),positions=positions,rows=rows,
                data_source_sha256=data_hash,program_source_sha256=prior['artifact_sha256'],seconds=time.perf_counter()-start,
                scope='Frozen local MLP17-input projection and square deletion on historical FineWeb states. No reader renormalization, no data fitting, no body forwards, no fresh/OOD or selective semantic circuit claim.')
    cache=Path('/dev/shm/bilin18_stable_square150_reader_screen_v1.pt')
    torch.save(dict(per_position=per_position,native_ce=native_ce,targets=targets,scalar=t),cache)
    result['cache']=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    (root/'STABLE_SQUARE150_READER_SCREEN_V1_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
