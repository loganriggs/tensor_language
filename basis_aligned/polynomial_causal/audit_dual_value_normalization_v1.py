"""Opened-case source-normalization mediation; no frozen-gain adoption claim."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import dual_value_field_reference as D
import forward_endpoint_program_reference as F
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent;OUT=BASE/'DUAL_VALUE_NORMALIZATION_AUDIT_V1.json'
ROWS=BASE/'DUAL_VALUE_FIELD_INTERCHANGE_V1_ROWS.pt'


def read(program,x,original=None):
    positions=torch.arange(x.shape[1]);features=program.features(x,positions)
    if original is not None:
        layer=program.background.layers[-1];eps=torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
        ratio=torch.sqrt((x.square().mean(-1,keepdim=True)+eps)/(original.square().mean(-1,keepdim=True)+eps))
        features={k:v*ratio.unsqueeze(-1) for k,v in features.items()}
    query={k:v[:,-1:] for k,v in features.items()}
    z=program.aggregate(query,features,positions[-1:],positions)
    return (.5*program.background.head(x[:,-1:])+.5*z@program.folded.T)[:,0]


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True);package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model=program.background
    maximum=0.;results={}
    with torch.inference_mode():
        table=F.dictionary(model)
        for pop,block in data.items():
            indices=[i for i,m in enumerate(block['metadata']) if m['world']<4];metadata=[block['metadata'][i] for i in indices];saved={}
            for start in range(0,len(indices),8):
                take=indices[start:start+8];tok=block['tokens'][take];context=program.prepare(tok);base=context['x']
                world=metadata[start]['world'];assert all(m['world']==world for m in metadata[start:start+len(take)])
                rm=block['raw_maps'][world];jm=block['join_maps'][world];mask,_=F.joins(tok)
                raw=D.raw_field(model,tok,rm)-D.raw_field(model,tok)
                joined=F.messages(model,tok,mask,jm,table=table)-F.messages(model,tok,mask,table=table)
                saved.setdefault('native',[]).append(context['logits'][:,-1].clone())
                for name,delta in (('raw',raw),('join',joined),('both',raw+joined)):
                    assert not bool(delta[:,-1].ne(0).any())
                    live=read(program,base+delta);fixed=read(program,base+delta,base)
                    maximum=max(maximum,float((live-block['query_logits']['map_'+name][take]).abs().max()))
                    saved.setdefault('live_'+name,[]).append(live);saved.setdefault('fixed_'+name,[]).append(fixed)
                maximum=max(maximum,float((context['logits'][:,-1]-block['query_logits']['native'][take]).abs().max()),float((read(program,base,base)-context['logits'][:,-1]).abs().max()))
            logits={k:torch.cat(v) for k,v in saved.items()};groups={}
            for field,hops in (('raw',(1,2,3)),('join',(3,))):
                for hop in hops:
                    sel=[m[field+'_consumer'] and m['hop']==hop for m in metadata]
                    groups[f'{field}_h{hop}']={mode+'_'+arm:M.panel(logits,mode+'_'+arm,metadata,sel,(field+'_desired' if arm==field else 'both_desired' if arm=='both' else 'answer')) for mode in ('live','fixed') for arm in ('raw','join','both')}
            interactions={}
            for mode in ('live','fixed'):
                v=logits[mode+'_both']-logits[mode+'_raw']-logits[mode+'_join']+logits['native'];v-=v.mean(-1,keepdim=True)
                interactions[mode]={'centered_rms':float(v.square().mean().sqrt()),'centered_max_abs':float(v.abs().max())}
            results[pop]={'groups':groups,'interactions':interactions}
            print(json.dumps({'population':pop,'interactions':interactions,'accuracy':{g:{a:round(v['target_accuracy'],5) for a,v in arms.items()} for g,arms in groups.items()}}),flush=True)
    receipt={'scope':'first4 openedworlds/pop,768queries; frozen gains are diagnostic interventions, not native semantic repair',
             'saved_live_replay_max_abs':maximum,'passed':maximum<=1e-9,'populations':results,
             'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'independent_constants':program.independent_constant_count(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');assert receipt['passed'];print(json.dumps({'passed':receipt['passed'],'replay':maximum,'wall_seconds':receipt['wall_seconds']}))


if __name__=='__main__':main()
