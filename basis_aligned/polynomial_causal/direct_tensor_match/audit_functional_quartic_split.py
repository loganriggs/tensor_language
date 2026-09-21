"""Audit duplicate input prefixes and rescore without overlapping rows.
Does not refit or reselect programs. Original registered eight-row scores remain.
"""
import collections,hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from source_interface import source_read
torch.set_num_threads(2);torch.set_grad_enabled(False)
cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)
c=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)['common']
hashes=[hashlib.sha256(row[:64].numpy().tobytes()).hexdigest() for row in cal['tokens']]
groups=collections.defaultdict(list)
for i,h in enumerate(hashes):groups[h].append(i)
keep=[i for i in range(24,32) if hashes[i] not in set(hashes[:24])]
overlap=[i for i in range(24,32) if i not in keep]
indices=torch.tensor([j for i in keep for j in range(i*64,(i+1)*64)])
z=cal['z'].flatten(0,1).double();truth=c['true_phi'];scale=c['scale'];records=[]
for family,stem in [('functional','FUNCTIONAL_QUARTIC_MODE3'),('fixed','FIXED_QUARTIC_PRODUCT')]:
 fit=json.loads((P/(stem+'_FIT_V1.json')).read_text())
 programs=torch.load(P/(stem+'_PROGRAMS_V1.pt'),weights_only=True)
 for name in ['initial',*fit['selected_by_training_objective'].values()]:
  program=programs[name]
  a=source_read(z,program,'a');b=source_read(z,program,'b')
  pred=((c['t']-.5*a)/scale-c['alpha'])*(b/scale-c['beta'])
  true=truth[indices];error=float((pred[indices]-true).norm()/(true-true.mean()).norm())
  perdoc=[]
  for i in keep:
   sl=slice(i*64,(i+1)*64);t=truth[sl]
   perdoc.append(dict(document_index=i,squared_error=float((pred[sl]-t).square().sum()),target_sum=float(t.sum()),target_square_sum=float(t.square().sum()),positions=64))
  records.append(dict(family=family,program=name,distinct_evaluation_variation_error=error,per_document=perdoc))
out=dict(exact_prefix_groups=list(groups.values()),excluded_evaluation_indices=overlap,retained_evaluation_indices=keep,records=records,scope='Post-evaluation duplicate audit: one training/evaluation prefix duplicate removed from an additional seven-row score. No refitting/reselection. Original registered eight-row result retained.')
(P/'FUNCTIONAL_QUARTIC_SPLIT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print('Excluded',overlap,'retained',keep)
for r in records:print(r['family'],r['program'],r['distinct_evaluation_variation_error'])
