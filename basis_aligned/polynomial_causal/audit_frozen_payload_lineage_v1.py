"""Opened binding-state lineage accounting; no field repair or semantic fit."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import frozen_payload_lineage_reference as L

BASE=Path(__file__).resolve().parent;OUT=BASE/'FROZEN_PAYLOAD_LINEAGE_V1_AUDIT.json'
ROWS=BASE/'SOURCE_PORT_FIELD_INTERCHANGE_V1_ROWS.pt'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    checks=L.controls();assert checks['passed'];data=torch.load(ROWS,map_location='cpu',weights_only=True)
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True));model=program.background
    maximum=diagonal_max=0.;result={};cachebytes=[]
    with torch.inference_mode():
        for pop,block in data.items():
            local=[];residual=[];parts_norms={}
            for world in range(4):
                index=next(i for i,m in enumerate(block['metadata']) if m['world']==world)
                tok=block['tokens'][index:index+1];cache=L.prepare(model,tok);parts=L.decompose(model,tok,cache)
                maximum=max(maximum,float((sum(parts.values())-cache['native_prefix']).abs().max()))
                local.append(parts['local_value_roots'][:,1:48:2]);residual.append(cache['embedding'][:,1:48:2]*.125)
                for name,value in parts.items():parts_norms.setdefault(name,[]).append(value[:,:48].square().sum())
                for source in range(1,48,2):
                    root=torch.zeros_like(cache['embedding']);root[:,source]=cache['embedding'][:,source]
                    direct=L.propagate(model,root,cache)[:,source]
                    diagonal_max=max(diagonal_max,float((direct-parts['local_value_roots'][:,source]).abs().max()))
                cachebytes.append(sum(v.numel()*v.element_size() for g in cache['gates'] for v in g.values()))
            local=torch.cat(local).flatten();residual=torch.cat(residual).flatten();extra=local-residual
            result[pop]={'worlds':4,'value_positions':96,'local_lineage_rms':float(local.square().mean().sqrt()),
                         'residual_E_over8_rms':float(residual.square().mean().sqrt()),
                         'additional_local_transport_relative_norm':float(extra.norm()/local.norm()),
                         'local_vs_residual_cosine':float(torch.dot(local,residual)/(local.norm()*residual.norm())),
                         'local_to_residual_norm_ratio':float(local.norm()/residual.norm()),
                         'binding_part_squared_norms':{k:float(torch.stack(v).sum()) for k,v in parts_norms.items()}}
    receipt={'scope':'opened first4worlds/pop,onequery/world; exact frozen-payload lineage, not token-input derivative or semantic field adoption',
             'controls':checks,'full_root_partition_max_abs':maximum,'diagonal_vs_individual_root_max_abs':diagonal_max,
             'passed':max(maximum,diagonal_max)<=1e-9,'populations':result,'per_input_native_gate_cache_bytes':cachebytes,
             'independent_program_constants':program.independent_constant_count(),'native_coefficients_removed':0,
             'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),
             'wall_seconds':time.perf_counter()-started}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert receipt['passed']


if __name__=='__main__':main()
