"""Opened first4worlds/population screen of a physically local initializer."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import field_intervention_metrics as M
import local_query_initializer_reference as L
from hop_ablate import load

ROOT=Path('/workspace/tensor_language');BASE=ROOT/'basis_aligned/polynomial_causal';ROWS=BASE/'LATE_HOP_INSTRUCTION_V1_ROWS.pt';OUT=BASE/'LOCAL_QUERY_INITIALIZER_V1_RESULT.json'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    controls=L.controls();assert controls['passed'];data=torch.load(ROWS,map_location='cpu',weights_only=True)
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt';assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model,_=load('attn4-rms-seed0');model=model.double().eval()
    audits=[];results={};earlier=0.
    with torch.inference_mode():
        for pop,block in data.items():
            tokens=block['native_tokens'][:4*96];candidate=[];native=[];kls=[]
            for i in range(0,len(tokens),8):
                tok=tokens[i:i+8];before=model(tok);after=L.execute(program,tok);oracle=L.native(model,tok)
                audits.append(M.correspondence(after,oracle));earlier=max(earlier,float((after[:,:-1]-before[:,:-1]).abs().max()))
                candidate.append(after[:,-1].clone());native.append(before[:,-1].clone())
                lp=before.log_softmax(-1);lq=after.log_softmax(-1);kls.append((lp.exp()*(lp-lq)).sum(-1).clamp_min(0))
            a=torch.cat(candidate);b=torch.cat(native);kl=torch.cat(kls);groups={}
            for hop in range(4):
                sel=torch.arange(len(tokens))%4==hop;target=b[sel]-b[sel].mean(-1,keepdim=True);predicted=a[sel]-a[sel].mean(-1,keepdim=True)
                err=float((predicted-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)
                groups[str(hop)]={'n':int(sel.sum()),'all_mean_kl':float(kl[sel].mean()),'query_mean_kl':float(kl[sel,-1].mean()),
                    'token_p99_kl':float(torch.quantile(kl[sel].flatten(),.99)),'maximum_kl':float(kl[sel].max()),'query_vector_relative_rms':err}
                groups[str(hop)]['passed']=groups[str(hop)]['all_mean_kl']<=.001 and groups[str(hop)]['query_mean_kl']<=.001 and groups[str(hop)]['token_p99_kl']<=.01 and err<=.01
            results[pop]=groups
    result={'scope':'opened8worlds768states, context-free local initializer; no freshidentification or wholemodelweight saving',
        'controls':controls,'mechanical_passed':all(a['passed'] for a in audits) and earlier<=1e-9,
        'candidate_passed':all(g['passed'] for p in results.values() for g in p.values()),'populations':results,
        'oracle_max_abs':max(a['max_abs'] for a in audits),'oracle_max_relative_rms':max(a['relative_rms'] for a in audits),'earlier_output_max_abs':earlier,
        'query_initialization_source_edges_before':204,'query_initialization_source_edges_after':12,'source_edges_removed':192,
        'independent_constants':program.independent_constant_count(),'native_coefficients_removed':0,
        'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['mechanical_passed']


if __name__=='__main__':main()
