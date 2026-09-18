"""Export the two-prefix interchange executor and its declared weights."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    target=P/'extracted_circuits/city_interchange_prefix_v1';target.mkdir(exist_ok=True)
    copies={'attention7.pt':'CITY_FULL_INTERCHANGE_V1_ATTENTION7.pt','readers.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_READERS.pt','head8.pt':'CITY_RESIDUAL6_SINGLE_INPUT_V1_HEAD8.pt','attention7.py':'city_attention7_full_v1.py','readers.py':'city_mlp7_readers_v1.py'}
    for name,source in copies.items():shutil.copyfile(P/source,target/name)
    (target/'fields.py').write_text((P/'city_residual6_fields_v1.py').read_text().replace('from city_attention7_full_v1 import','from attention7 import').replace('from city_mlp7_readers_v1 import','from readers import'))
    (target/'execute.py').write_text((P/'city_interchange_prefix_v1.py').read_text().replace('from city_residual6_fields_v1 import','from fields import'))
    fresh=json.loads((P/'CITY_FULL_INTERCHANGE_V1_RESULT.json').read_text());installed=json.loads((P/'CITY_INTERCHANGE_PREFIX_INSTALLED_V1_RESULT.json').read_text())
    manifest={'schema':'regional.city_interchange_prefix.v1','name':target.name,'certified_complete':False,
      'counterfactual':'Swap both normalized city key fields and complete current/inherited value from paired donor, retain recipient queries/background; emit attention8 delta.',
      'ports':['recipient_residual6[1,last_destination+1,1152]','recipient_tokens[1,last_destination+1]','donor_residual6[1,city+1,1152]','donor_tokens[1,city+1]','city:int','destination[T]:bool'],
      'external_native_activation_inputs':2,'native_vector_state_arrays':2,'native_scalar_state_arrays':0,'logical_native_contexts':2,
      'native_state_scalars_T32':50688,'static_float_scalars':fresh['floating_scalars'],'weight_bytes':4*fresh['floating_scalars'],'supported_sequence_tokens':385,
      'external_computation':'Native blocks0–6 for both contexts; native MLP8 and full later model for recipient. Both queries and all normalization factors generated internally, mixed8 RMS approximate.',
      'four_traits':{'ood_prediction':all(fresh['pred_'+k] for k in 'ab'),'extraction':all(installed['pred_'+k] for k in 'abc'),'selective_removal':None,'composition_reuse':False},
      'selective_interchange':all(fresh['pred_'+k] for k in 'acde'),
      'property_scope':'Fresh same-corpus conditional interchange, not whole-head swap, token replacement or independent composition. Prefix replay is opened implementation evidence. Removal evidence belongs to separate operator.',
      'evidence':{'fresh_interchange':'../../CITY_FULL_INTERCHANGE_V1_RESULT.json','prefix_local':'../../CITY_INTERCHANGE_PREFIX_V1_CPU_RESULT.json','prefix_installed':'../../CITY_INTERCHANGE_PREFIX_INSTALLED_V1_RESULT.json','scope_correction':'../../CITY_INTERCHANGE_PREFIX_INSTALLED_V1_SCOPE_CORRECTION.json','isolated':'../../CITY_INTERCHANGE_PREFIX_V1_ISOLATED_RESULT.json'},
      'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(target.iterdir()) if f.suffix in ['.py','.pt']}}
    manifest['selective_manipulation']=manifest['selective_interchange']
    isolated=P/'CITY_INTERCHANGE_PREFIX_V1_ISOLATED_RESULT.json'
    manifest['isolated_execution_passes']=json.loads(isolated.read_text())['passes'] if isolated.exists() else None
    manifest['serialized_program_bytes']=sum(f.stat().st_size for f in target.glob('*.pt'))
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (target/'README.md').write_text('''# Conditional city interchange from two native prefixes

Copy all files in this directory. Requires PyTorch, CPU float32 native inputs and
batch1. Load attention7.pt, readers.pt, head8.pt with torch.load(weights_only=True)
into program keyed by filename stem. Import execute, then call:

```python
execute.execute(program, recipient_residual6, recipient_tokens,
                donor_residual6, donor_tokens, city, destination)
```

Recipient prefix ends at last destination; donor prefix ends at city. Token IDs
match each prefix. Destination is a Boolean mask of the full output length. Output
is float64 attention8 delta, exactly zero outside the edited positions. Unknown
tokens and unsupported prefix shapes fail. Tables cover385tokens from the frozen
fresh panel. No native query or RMS inputs; mixed8 RMS uses the stated approximation.

For T32, city12 and destinations13..30: recipient31tokens + donor13tokens,
50,688native floats versus73,728for two full sequences. Two logical native contexts
remain. Shared weights:23,300,998FP32values /93,203,992bytes. Weight price unchanged;
no matched-effect random-component simplicity advantage or runtime speedup claimed.
Native blocks0–6 and recipient MLP8/later model remain external.

Fresh prediction and selective interchange pass; independent composition remains
failed/unestablished. This is not full token replacement or a whole-head donation.
The installed receipt contains a stale copied scope sentence: its linked scope
correction states the true inputs without changing any measurements or gates.
''')
if __name__=='__main__':main()
