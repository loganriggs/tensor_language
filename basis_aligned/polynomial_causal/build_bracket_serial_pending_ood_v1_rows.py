#!/usr/bin/env python3
"""Freeze an eighth serial-pending bracket construction without model access."""
from __future__ import annotations
from collections import Counter
import hashlib,json
from pathlib import Path
import tiktoken
HERE=Path(__file__).resolve().parent;OUT=HERE/'BRACKET_SERIAL_PENDING_OOD_V1_ROWS.json'
MARKS={'parenthesis':{'open':'(','close':')','open_id':357},'square':{'open':'[','close':']','open_id':685},'quote':{'open':'"','close':'"','open_id':366}};UNORDERED=(('parenthesis','square'),('parenthesis','quote'),('square','quote'))
GROUPS=(('botany ledger','orchid','canopy','specimen'),('aviation digest','runway','altitude','approach'),('culinary folio','saffron','broth','garnish'),('theater archive','rehearsal','costume','dialogue'),('finance memo','dividend','ledger','forecast'),('marine atlas','estuary','current','vessel'),('medical handbook','symptom','therapy','prognosis'),('architecture brief','facade','column','foundation'),('history catalogue','dynasty','treaty','chronicle'),('ecology report','habitat','species','migration'),('computing manual','kernel','buffer','protocol'),('astronomy circular','nebula','telescope','ephemeris'))
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def build_rows(token):
 answers={n:token(v['close'])[0] for n,v in MARKS.items()};rows=[]
 for gi,(document,a,b,c) in enumerate(GROUPS):
  for left,right in UNORDERED:
   middle=next(n for n in MARKS if n not in (left,right))
   def text(inner):return f"The {document} retained {{ a framing note; later {MARKS[middle]['open']} a secondary note remained open; after another transition {MARKS[inner]['open']} the {a}, the {b}, and the {c} awaited completion"
   base,donor=text(left),text(right);bi,di=token(base),token(donor);assert len(bi)==len(di);bo=max(i for i,v in enumerate(bi) if v==MARKS[left]['open_id']);do=max(i for i,v in enumerate(di) if v==MARKS[right]['open_id']);assert bo==do and [i for i,z in enumerate(zip(bi,di)) if z[0]!=z[1]]==[bo]
   row={'family_id':'serial_pending_stack_top','program_role':'target','group_index':gi,'base_text':base,'donor_text':donor,'base_ids':bi,'donor_ids':di,'base_answer_id':answers[left],'donor_answer_id':answers[right],'base_open_position':bo,'donor_open_position':do,'base_type':left,'donor_type':right,'outer_type':'brace','middle_type':middle};row['row_id']=canonical(row);rows.append(row)
  for inner in MARKS:
   mids=[n for n in MARKS if n!=inner]
   def text(mid):return f"The {document} retained {{ a framing note; later {MARKS[mid]['open']} a secondary note remained open; after another transition {MARKS[inner]['open']} the {a}, the {b}, and the {c} awaited completion"
   base,donor=text(mids[0]),text(mids[1]);bi,di=token(base),token(donor);assert len(bi)==len(di);bo=max(i for i,v in enumerate(bi) if v==MARKS[inner]['open_id']);do=max(i for i,v in enumerate(di) if v==MARKS[inner]['open_id']);assert bo==do and len([i for i,z in enumerate(zip(bi,di)) if z[0]!=z[1]])==1
   row={'family_id':'serial_middle_change_inner_fixed','program_role':'control','group_index':gi,'base_text':base,'donor_text':donor,'base_ids':bi,'donor_ids':di,'base_answer_id':answers[inner],'donor_answer_id':answers[inner],'base_open_position':bo,'donor_open_position':do,'inner_type':inner,'outer_type':'brace','base_middle_type':mids[0],'donor_middle_type':mids[1]};row['row_id']=canonical(row);rows.append(row)
 return rows
def main():
 if OUT.exists():raise FileExistsError(OUT)
 rows=build_rows(tiktoken.get_encoding('gpt2').encode);counts=Counter(r['program_role'] for r in rows);pairs=Counter()
 for r in rows:
  if r['program_role']=='target':pairs[(r['base_answer_id'],r['donor_answer_id'])]+=1;pairs[(r['donor_answer_id'],r['base_answer_id'])]+=1
 assert len(rows)==72 and counts=={'target':36,'control':36} and len(pairs)==6 and set(pairs.values())=={12};payload={'schema':'bracket_serial_pending_ood_v1_rows','model_loaded':False,'outcomes_opened':[],'row_count':72,'endpoint_count':144,'counts':dict(counts),'ordered_pair_counts':{f'{a}->{b}':n for (a,b),n in sorted(pairs.items())},'rows':rows};payload['row_manifest_sha256']=canonical(rows);OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({k:payload[k] for k in ('row_count','endpoint_count','counts','ordered_pair_counts','row_manifest_sha256')},indent=2))
if __name__=='__main__':main()
