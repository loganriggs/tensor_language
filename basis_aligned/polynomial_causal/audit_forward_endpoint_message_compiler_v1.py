"""Opened-case, CPU-only exact translation and literal dependency accounting."""
import hashlib
import json
from pathlib import Path
import time
import torch
import exact_source_edit_reference as E
import join_endpoint_field_swap_reference as R
import forward_endpoint_program_reference as F

BASE=Path(__file__).resolve().parent
OUT=BASE/'FORWARD_ENDPOINT_MESSAGE_COMPILER_V1_AUDIT.json'


def main():
    torch.set_num_threads(2);started=time.perf_counter();checks=F.controls();assert checks['passed']
    assert not OUT.exists()
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model=program.background
    maximum=0.;native_maximum=0.;rows=0;counts=[];subset=True
    with torch.inference_mode():
        table=F.dictionary(model)
        for pop,worlds in R.populations().items():
            for w in worlds:
                tok=w['recipient'][:1];parsed,records=F.joins(tok)
                selected=(w['masks'][0]|w['masks'][1]).clone();selected[:,::2]=False;selected=selected[None]
                subset &= not bool((selected & ~parsed).any())
                mapping=torch.arange(24);a,b=int(w['answers'][3]),int(w['answers'][7]);mapping[a]=b;mapping[b]=a
                for swap in (False,True):
                    mapped=mapping if swap else None
                    exact=sum(R.paths(model,tok,w['masks'],swap).values())
                    compiled=F.messages(model,tok,selected,mapped,table=table)
                    maximum=max(maximum,float((compiled-exact).abs().max()))
                    all_compiled=F.messages(model,tok,parsed,mapped,table=table)
                    all_native=F.native_messages(model,tok,parsed,mapped)
                    native_maximum=max(native_maximum,float((all_compiled-all_native).abs().max()))
                counts.append({'population':pop,'world':w['world'],'joins':len(records[0]),'message_cells':int(parsed.sum())});rows+=1
    receipt={'scope':'opened-case CPU algebra audit; randomized-layout semantic validation remains untested',
             'controls':checks,'worlds':rows,'selected_paths_are_parsed_joins':subset,
             'selected_endpoint_path_max_abs':maximum,'all_parsed_native_hook_max_abs':native_maximum,
             'passed':bool(checks['passed'] and subset and max(maximum,native_maximum)<=1e-9),
             'formula':'phi(e)=O_L2H1 V_L2H1 Emb(e)/8; message(t,s)=P_L2H1(t,s)*RMSgain_L2(s)*phi(endpoint(s))',
             'dictionary_shape':list(table.shape),'derived_cache_coefficients':table.numel(),
             'derived_cache_bytes':table.numel()*table.element_size(),'independent_program_constants':program.independent_constant_count(),
             'new_independent_coefficients':0,'native_coefficients_removed':0,
             'opaque_dependencies':['native prefix states','L2H1 contextual QK scores','L2 source RMS gains','native complement and final readers','derived arbitrary endpoint vectors'],
             'structural_saving_claim':False,'counts':counts,'wall_seconds':time.perf_counter()-started,
             'reference_sha256':hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest(),
             'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='counts'},indent=2))
    assert receipt['passed']


if __name__=='__main__':main()
