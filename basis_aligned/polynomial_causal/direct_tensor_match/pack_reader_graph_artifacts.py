"""Pack tensor views without changing program values or intentional within-program reuse.
Retains a one-time receipt; reruns verify existing packed artifacts.
"""
from pathlib import Path
import hashlib,json,torch
P=Path(__file__).parent;torch.set_num_threads(2)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def counts(program):
 logical={};storage={}
 def visit(v):
  if isinstance(v,dict):
   for x in v.values():visit(x)
  elif torch.is_tensor(v) and v.is_floating_point():
   logical[(v.data_ptr(),tuple(v.shape),tuple(v.stride()))]=v.numel();storage[v.untyped_storage().data_ptr()]=v.untyped_storage().nbytes()//v.element_size()
 visit(program);return dict(logical_float_coefficients=sum(logical.values()),backing_storage_floats=sum(storage.values()))
def packed(program):
 cache={}
 def visit(v):
  if isinstance(v,dict):return {k:visit(x) for k,x in v.items()}
  if not torch.is_tensor(v):return v
  key=(v.data_ptr(),tuple(v.shape),tuple(v.stride()),v.dtype)
  if key not in cache:cache[key]=v.clone(memory_format=torch.contiguous_format)
  assert torch.equal(v,cache[key]);return cache[key]
 return visit(program)
def main():
 receipt=P/'READER_GRAPH_PACKING_V1.json'
 if receipt.exists():
  data=json.loads(receipt.read_text())
  for rec in data['artifacts']:assert sha(P/rec['file'])==rec['after_sha256']
  print('Previously packed artifact hashes verified');return
 records=[]
 for file in ['SHARED_LINEAR_GRAPH_PROGRAMS_V1.pt','LOCAL_SHARED_READER_PROGRAMS_V1.pt']:
  path=P/file;before_hash=sha(path);models=torch.load(path,weights_only=True);before={k:counts(v) for k,v in models.items()};after={k:packed(v) for k,v in models.items()};after_counts={k:counts(v) for k,v in after.items()}
  for key in models:
   assert before[key]['logical_float_coefficients']==after_counts[key]['logical_float_coefficients']==after_counts[key]['backing_storage_floats']
  torch.save(after,path);records.append(dict(file=file,before_sha256=before_hash,after_sha256=sha(path),before=before,after=after_counts,all_tensor_values_bitwise_unchanged=True))
 receipt.write_text(json.dumps(dict(artifacts=records,correction='Basis slices kept unused SVD/eigenvector storage. Architecture coefficient counts and arithmetic were correct; old serialized models retained extra backing values. Packed every tensor contiguously, preserving intentional repeated within-program references and every tensor value. Earlier shared-linear LFS artifact replaced by equivalent packed export; no scientific metrics/gates changed.'),indent=2)+'\n');print(json.dumps(records,indent=2))
if __name__=='__main__':main()
