"""Opened-case final-head response diagnostic; no selected head is adopted."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import forward_endpoint_program_reference as F

BASE=Path(__file__).resolve().parent;OUT=BASE/'FINAL_ENDPOINT_READER_AUDIT_V1.json'
ROWS=BASE/'FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_ROWS.pt'


def query_heads(program,x):
    positions=torch.arange(x.shape[1]);features=program.features(x,positions)
    query={k:v[:,-1:] for k,v in features.items()}
    z=program.aggregate(query,features,positions[-1:],positions).reshape(len(x),4,32)
    return torch.stack([.5*(z[:,h]@program.folded[:,32*h:32*(h+1)].T) for h in range(4)],1)


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True)
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model=program.background
    maximum=0.;result={}
    with torch.inference_mode():
        table=F.dictionary(model)
        for pop,block in data.items():
            indices=[i for i,m in enumerate(block['metadata']) if m['world']<4]
            head_changes=[];base_logits=[];mapped_logits=[];meta=[block['metadata'][i] for i in indices]
            for start in range(0,len(indices),8):
                take=indices[start:start+8];tok=block['tokens'][take];context=program.prepare(tok)
                mask,_=F.joins(tok);world=meta[start]['world'];assert all(meta[k]['world']==world for k in range(start,start+len(take)))
                mapping=block['endpoint_maps'][world]
                delta=F.messages(model,tok,mask,mapping,table=table)-F.messages(model,tok,mask,table=table)
                assert not bool(delta[:,-1].ne(0).any())
                native=query_heads(program,context['x']);mapped=query_heads(program,context['x']+delta)
                skip=.5*model.head(context['x'][:,-1]);base=skip+native.sum(1);changed=skip+mapped.sum(1)
                maximum=max(maximum,float((base-block['query_logits']['native'][take]).abs().max()),
                            float((changed-block['query_logits']['map_all'][take]).abs().max()))
                head_changes.append(mapped-native);base_logits.append(base);mapped_logits.append(changed)
            changes=torch.cat(head_changes);native=torch.cat(base_logits);mapped=torch.cat(mapped_logits)
            total=mapped-native;center=lambda x:x-x.mean(-1,keepdim=True)
            answers=torch.tensor([m['answer'] for m in meta]);desired=torch.tensor([m['desired_answer'] for m in meta]);groups={}
            for hop in range(4):
                for eligible in ((False,) if hop<2 else (False,True)):
                    sel=torch.tensor([m['hop']==hop and m['eligible']==eligible for m in meta]);effect=center(total[sel]);denom=max(float(effect.square().sum()),1e-12)
                    heads={}
                    for h in range(4):
                        part=center(changes[sel,h]);single=native+changes[:,h];complement=mapped-changes[:,h]
                        heads[str(h)]={'signed_effect_projection':float((part*effect).sum())/denom,
                                       'relative_effect_norm':float(part.square().sum().sqrt())/max(float(effect.square().sum().sqrt()),1e-6),
                                       'single_target_accuracy':float((single.argmax(-1)==desired)[sel].double().mean()),
                                       'complement_target_accuracy':float((complement.argmax(-1)==desired)[sel].double().mean()),
                                       'single_gold_probability_change':float((single.softmax(-1).gather(-1,answers[:,None])-native.softmax(-1).gather(-1,answers[:,None]))[sel].mean())}
                    groups[f'hop{hop}_eligible{eligible}']={'n':int(sel.sum()),'centered_total_effect_rms':float(effect.square().mean().sqrt()),
                                                        'native_accuracy':float((native.argmax(-1)==answers)[sel].double().mean()),
                                                        'all_target_accuracy':float((mapped.argmax(-1)==desired)[sel].double().mean()),'heads':heads}
            result[pop]=groups;print(json.dumps({'population':pop,'eligible_hop2':groups['hop2_eligibleTrue'],'eligible_hop3':groups['hop3_eligibleTrue']}),flush=True)
    receipt={'scope':'opened first4worlds/pop diagnostic,768queryvariants; no head adoption or fresh semantic evidence',
             'saved_output_replay_max_abs':maximum,'passed':maximum<=1e-9,'populations':result,
             'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'independent_constants':program.independent_constant_count(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');assert receipt['passed'];print(json.dumps({'passed':receipt['passed'],'replay':maximum,'wall_seconds':receipt['wall_seconds']}))


if __name__=='__main__':main()
