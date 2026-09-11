"""Frozen three-program tail validation, 128 cached FineWeb positions.

A native precision tail replay; B native baseline beats both learned programs
in KL and absolute mean CE change; C all programs preserve both within .05.
No parameter selection, fitting, new corpus rows or model-body forwards.
"""
import hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from native_support_exchange_v1_audit import P,CK
from rectangular_sparse_reader_v1 import RectangularSparseReaderProgram


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'MATCHED_PRICE_FINEWEB_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    data_path=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt')
    sha=hashlib.sha256(data_path.read_bytes()).hexdigest();assert sha=='b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de'
    data=torch.load(data_path,weights_only=True,map_location='cpu');positions=[63,127]
    x=data['ports']['input'][:,positions].reshape(-1,1152).double()
    pre=data['ports']['pre'][:,positions].reshape(-1,1152).float()
    cached=data['ports']['native_output'][:,positions].reshape(-1,1152).float()
    targets=data['rows'][:,[p+1 for p in positions]].reshape(-1).long();assert len(targets)==128
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    bias=sd['transformer.h.17.mlp.Down_bias'].double();u=sd['lm_head.weight'].float()
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    def logits(output):return 30*torch.tanh((F.rms_norm(pre+output.float(),(1152,))@u.T)/30)
    reference=logits(cached);native_log=reference.double().log_softmax(-1);native_prob=native_log.exp()
    native_ce=-native_log.gather(1,targets[:,None]).squeeze(1)
    exact=((x@l.T)*(x@r.T))@d.T+bias;control=logits(exact)
    precision=float((control-reference).norm()/reference.norm())
    control_ce=F.cross_entropy(control.double(),targets);ce_replay=abs(float(control_ce-native_ce.mean()))
    native_receipt=json.loads((P/'MATCHED_PRICE_NATIVE_V1_AUDIT.json').read_text());source=native_receipt['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    native_program=torch.load(source['path'],weights_only=True,map_location='cpu')['energy']
    outputs={'native_selection':((x@native_program['left'].T)*(x@native_program['right'].T))@native_program['down'].T+native_program['bias']}
    sources={'native_selection':source}
    parent=json.loads((P/'FULL_SUPPORT_EXCHANGE_V1_RESULT.json').read_text())
    for row in parent['arms']:
        if row['mode']!='exchange':continue
        source=row['cache'];assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
        saved=torch.load(source['path'],weights_only=True,map_location='cpu')
        program=RectangularSparseReaderProgram.from_artifact(saved,saved['down'].double(),bias)
        name=f"dictionary_{row['seed']}";outputs[name]=program(x);sources[name]=source
    rows=[];per_position={}
    for name,output in outputs.items():
        log=logits(output).double().log_softmax(-1);ce=-log.gather(1,targets[:,None]).squeeze(1)
        added=ce-native_ce;kl=(native_prob*(native_log-log)).sum(-1);delta=output-cached.double()
        error=float(((delta@metric)*delta).sum()/((cached.double()@metric)*cached.double()).sum())
        rows.append(dict(name=name,native_ce=float(native_ce.mean()),candidate_ce=float(ce.mean()),
            ce_added=float(added.mean()),absolute_mean_ce_change=abs(float(added.mean())),
            mean_absolute_position_ce_change=float(added.abs().mean()),kl=float(kl.mean()),relative_mlp_error=error))
        per_position[name]=torch.stack((ce,added,kl),1)
    base=rows[0]
    pred=dict(pred_a_instrument=precision<=1e-5 and ce_replay<=1e-4,
        pred_b_native_behavior_advantage=all(base['kl']<=r['kl'] and base['absolute_mean_ce_change']<=r['absolute_mean_ce_change'] for r in rows[1:]),
        pred_c_preservation=all(r['absolute_mean_ce_change']<=.05 and r['kl']<=.05 for r in rows))
    artifact=Path('/dev/shm/bilin18_matched_price_fineweb_v1.pt');assert not artifact.exists()
    torch.save(dict(per_position=per_position,native_ce=native_ce,positions=positions,targets=targets,sources=sources),artifact)
    result=dict(predictions=pred,rows=rows,native_logit_precision_replay=precision,native_ce_precision_replay=ce_replay,
        positions=positions,sequence_rows=64,prediction_positions=128,body_forwards=0,fit_steps=0,
        source=dict(path=str(data_path),sha256=sha,previously_opened=True),program_sources=sources,
        cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),seconds=time.perf_counter()-start,
        scope='Frozen tail replacement on a small historical FineWeb panel; not fresh/OOD, whole-corpus ranking, extraction or selective circuit evidence.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
