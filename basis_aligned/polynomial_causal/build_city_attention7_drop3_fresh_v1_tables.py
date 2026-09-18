"""Weight-only fresh token tables, then physical retained-head packing; no fitting."""
from pathlib import Path
import json,torch,hashlib
import build_city_residual6_single_input_fresh_v1_tables as tables
P=Path(__file__).resolve().parent;STEM='CITY_ATTENTION7_DROP3_FRESH_V1'
if __name__=='__main__':
 tables.STEM=STEM;tables.main()
 a=torch.load(P/(STEM+'_ATTENTION7.pt'),weights_only=True)
 heads=[h for h in range(9) if h!=3];idx=torch.tensor([128*h+j for h in heads for j in range(128)])
 packed={k:v.clone() for k,v in a.items()}
 for k in ['q1','k1','q2','k2','value']:packed[k]=a[k][idx].clone()
 packed['output']=a['output'][:,idx].clone();packed['head_ids']=torch.tensor(heads)
 out=P/(STEM+'_PACKED_ATTENTION7.pt');assert not out.exists();torch.save(packed,out)
 original=torch.load(P/'CITY_ATTENTION7_DROP3_PACK_V1_ATTENTION.pt',weights_only=True)
 unchanged=all(torch.equal(v,packed[k]) for k,v in original.items() if k not in ['token_ids','initial_table','first_table'])
 r={'non_table_weights_unchanged':unchanged,'physical_storage_owned':all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in packed.values()),'program_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'floating_scalars':sum(v.numel() for v in packed.values() if v.is_floating_point())}
 assert unchanged and r['physical_storage_owned'];(P/(STEM+'_PACKING_RESULT.json')).write_text(json.dumps(r,indent=2)+'\n');print(r)
