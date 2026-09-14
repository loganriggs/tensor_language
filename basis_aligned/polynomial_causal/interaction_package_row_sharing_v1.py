"""Discriminate packed native rows from absence of exact storage reuse.

A: invert every 1152-axis move/flatten/index reconstruction byte-identically.
B: exact duplicate row savings minus int64 routes and layout JSON >=1% of
   complete tensor payload. Preserve dtypes; no approximate row matching.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import torch
from interaction_package_shared_bank_v1 import NAMES, tensor_id

P = Path(__file__).resolve().parent


def main():
    out = P/'INTERACTION_PACKAGE_ROW_SHARING_V1_RESULT.json'
    if out.exists():
        raise FileExistsError(out)
    pools, ids, uses, records, layouts = {}, {}, {}, [], {}
    total = unhandled = original_rows = 0

    def visit(value, path):
        nonlocal total, unhandled, original_rows
        if isinstance(value, dict):
            for k,v in value.items(): visit(v,path+'/'+str(k))
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value): visit(v,path+'/'+str(i))
        elif isinstance(value,torch.Tensor):
            total += value.numel()*value.element_size()
            if 1152 not in value.shape:
                unhandled += value.numel()*value.element_size()
                records.append(dict(path=path,shape=list(value.shape),partitioned=False,replay=True))
                return
            axis = list(value.shape).index(1152)
            moved = value.movedim(axis,-1)
            rows = moved.reshape(-1,1152)
            dtype = str(value.dtype)
            pool, index = pools.setdefault(dtype,[]), ids.setdefault(dtype,{})
            routes = []
            for i,row in enumerate(rows):
                key = tensor_id(row)
                if key not in index:
                    index[key]=len(pool);pool.append(row.clone())
                routes.append(index[key]);uses.setdefault(key,[]).append(path+':'+str(i))
            restored = torch.stack([pool[i] for i in routes]).reshape(moved.shape).movedim(-1,axis)
            replay = tensor_id(restored)==tensor_id(value)
            original_rows += len(rows)
            # Shape/dtype/axis are layout overhead. Routes are separately charged int64.
            layouts[path]=dict(shape=list(value.shape),dtype=dtype,axis=axis,rows=len(routes))
            records.append(dict(path=path,shape=list(value.shape),partitioned=True,rows=len(routes),replay=replay))

    for name in NAMES:
        visit(torch.load(P/'extracted_circuits'/name/'program.pt',weights_only=True),name)
    unique_rows=sum(len(v) for v in pools.values())
    unique_bytes=sum(r.numel()*r.element_size() for rows in pools.values() for r in rows)
    index_bytes=8*original_rows
    metadata_bytes=len(json.dumps(layouts,sort_keys=True).encode())
    shared=unhandled+unique_bytes+index_bytes+metadata_bytes
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(r['replay'] for r in records),
                pred_b=shared<=.99*total,independent_payload_bytes=total,
                row_bank_payload_bytes=unique_bytes,unpartitioned_payload_bytes=unhandled,
                route_index_bytes=index_bytes,layout_metadata_bytes=metadata_bytes,
                priced_shared_payload_bytes=shared,saving_fraction=1-shared/total,
                original_rows=original_rows,unique_rows=unique_rows,records=records,
                duplicate_row_groups=[dict(sha256=k,uses=v) for k,v in uses.items() if len(v)>1],
                scope='Exact dtype-preserving row-storage screen on fixed five-package portfolio. Pricing includes routing/layout payload but no new serialization/executor implementation; no adoption, semantic identification, compute-speed or whole-model claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','duplicate_row_groups']},indent=2))


if __name__ == '__main__': main()
