"""Saved native reader/norm statistics: exact fixed-write edits, no model calls."""
import hashlib,json
from pathlib import Path
import numpy as np
import mixed_state_projector_v1 as Q
import rms_softcap_edit_statistics_v1 as S


def main():
    p=Path(__file__).resolve().parent;source=p/'THIRD_NOUN_ATTENTION_READER_V1_RESULT.json'
    data=json.loads(source.read_text());assert data['predictions']['pred_a_instrument']
    stats=[]
    for record in data['statistics']:
        stats.append({k:np.asarray(v) if isinstance(v,list) else v for k,v in record.items()})
    q=Q.projector(data['corners'],2,4)
    def mixed(x):return np.einsum('ij,wj->wi',q,np.asarray(x).reshape(8,32))
    def margin(pairs):
        a=np.array(pairs);return a[:,0]-a[:,1]
    native=margin([S.decode(s,np.zeros(19)) for s in stats]);target=mixed(native)
    dose=np.array([-1.]*18+[0.]);num_only=[];norm_only=[];actual=[];ratio_norm=[]
    for s in stats:
        n0=s['base_reader'];n1=n0+dose@s['source_readers']
        r0=np.sqrt(s['base_norm2']/s['dimension']+s['eps'])
        square=s['base_norm2']+2*dose@s['base_source_dots']+dose@s['source_gram']@dose
        r1=np.sqrt(square/s['dimension']+s['eps'])
        num_only.append(s['cap']*np.tanh(n1/(s['cap']*r0)))
        norm_only.append(s['cap']*np.tanh(n0/(s['cap']*r1)))
        actual.append(S.decode(s,dose));ratio_norm.append(r1/r0)
    full=margin(actual);numerator=margin(num_only);norm=margin(norm_only)
    layer_effects=[]
    for i in range(18):
        a=np.zeros(19);a[i]=-1
        layer_effects.append(mixed(native-margin([S.decode(s,a) for s in stats])))
    reports=[]
    for w,ref in enumerate(data['reports']):
        t=target[w];den=float(t@t)
        def geometry(effect):
            return {'signed_projection':float(effect@t/den),'effect_norm_ratio':float(np.linalg.norm(effect)/np.sqrt(den))}
        effect=mixed(native-full)[w];num=mixed(native-numerator)[w];nr=mixed(native-norm)[w]
        combined=mixed(full-numerator-norm+native)[w]
        per_layer={f'attn:{i:02d}':geometry(v[w]) for i,v in enumerate(layer_effects)}
        replay=abs(np.linalg.norm(mixed(full)[w])/np.sqrt(den)-ref['remaining_ratios']['A'])
        assert replay<1e-4
        reports.append({'world_id':ref['world_id'],'attention_removal_replay_ratio_error':float(replay),
            'attention_full':geometry(effect),'numerator_only':geometry(num),'norm_only':geometry(nr),
            'numerator_norm_interaction':geometry(combined),'layers':per_layer,
            'layer_additivity_error_ratio':float(np.linalg.norm(sum(v[w] for v in layer_effects)-effect)/np.sqrt(den)),
            'largest_signed_layer':max(per_layer,key=lambda k:per_layer[k]['signed_projection'])})
    result={'passed':True,'model_forwards':0,'reports':reports,
        'edited_over_native_norm_range':[float(min(ratio_norm)),float(max(ratio_norm))],
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'decoder_helper_sha256':hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
        'scope':'Post-outcome fixed-native-write localization. Separate numerator/norm edits are decoder diagnostics, not physical residual interventions. Layer edits do not recompute their native downstream producers. No sufficiency threshold changed or circuit promoted.'}
    with (p/'THIRD_NOUN_READER_ROUTE_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='layers'} for r in reports],indent=2))


if __name__=='__main__':main()
