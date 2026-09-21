"""Byte-correct discovery annotation; no semantic claim from decoded replacement glyphs."""
from pathlib import Path
import json,codecs,torch,tiktoken,collections
p=Path(__file__).resolve().parent;torch.set_num_threads(2)
r=json.loads((p/'MIDPOINT_STABLE_GROUP_PROFILE_V1.json').read_text());tokens=torch.load(p/'MIDPOINT_STABLE_GROUP_REMOVAL_TOKENS_V1.pt',weights_only=True);enc=tiktoken.get_encoding('gpt2');annotations={};boundary_crops=[]
for doc,row in enumerate(tokens.tolist()):
 raw=b''.join(enc.decode_single_token_bytes(t) for t in row);drop=0
 while drop<len(raw) and 128<=raw[drop]<192:drop+=1
 # Cache rows may begin mid-codepoint. Verify all subsequent complete bytes.
 strict=codecs.getincrementaldecoder('utf-8')('strict');strict.decode(raw[drop:],final=False)
 if drop:boundary_crops.append(dict(document=doc+96,leading_continuation_bytes=drop))
 dec=codecs.getincrementaldecoder('utf-8')('replace');decoded=''
 for pos,token in enumerate(row):
  piece=enc.decode_single_token_bytes(token);decoded+=dec.decode(piece,final=False);pending=bool(dec.getstate()[0]);last=decoded[-1:] if decoded else ''
  if pos<256:
   nextbytes=enc.decode_single_token_bytes(row[pos+1]);nexttext=nextbytes.decode('utf-8',errors='replace');starts_space=bool(nextbytes and nextbytes[:1].isspace())
   annotations[(doc+96,pos)]=dict(utf8_pending=pending,last_decoded_is_letter=last.isalpha(),next_starts_space=starts_space,next_ascii_letter_continuation=not pending and last.isalpha() and bool(nextbytes) and chr(nextbytes[0]).isascii() and chr(nextbytes[0]).isalpha(),next_initial_byte=nextbytes[0] if nextbytes else -1)
records=[]
for group in ['group1','group2','group3']:
 rr=[{**x,**annotations[(x['document'],x['position'])]} for x in r['records'] if x['group']==group]
 for field in ['utf8_pending','next_ascii_letter_continuation','next_starts_space']:
  for value in [False,True]:
   v=[x for x in rr if x[field]==value]
   records.append(dict(group=group,condition=field,value=value,sites=len(v),documents=len(set(x['document'] for x in v)),ce_added=sum(x['ce_added'] for x in v)/len(v) if v else None,mean_effect_energy=sum(x['effect_energy'] for x in v)/len(v) if v else None,mean_product=sum(x['products'][0] for x in v)/len(v) if v and group!='group1' else None))
# Leave-one-document-out lookup for scalar group2, paired versus global-mean baseline.
rr=[x for x in r['records'] if x['group']=='group2'];sums=collections.defaultdict(float);counts=collections.Counter();docsum=collections.defaultdict(float);doccount=collections.Counter();total=0;dt=collections.defaultdict(float);dn=collections.Counter()
for x in rr:
 key=x['input_token'];doc=x['document'];y=x['products'][0];sums[key]+=y;counts[key]+=1;docsum[doc,key]+=y;doccount[doc,key]+=1;total+=y;dt[doc]+=y;dn[doc]+=1
err=baseerr=0.;covered=0;covered_err=covered_base=0.
for x in rr:
 key=x['input_token'];doc=x['document'];y=x['products'][0];base=(total-dt[doc])/(len(rr)-dn[doc]);n=counts[key]-doccount[doc,key]
 pred=(sums[key]-docsum[doc,key])/n if n else base
 err+=(pred-y)**2;baseerr+=(base-y)**2
 if n:covered+=1;covered_err+=(pred-y)**2;covered_base+=(base-y)**2
out=p/'MIDPOINT_STABLE_GROUP_BOUNDARY_AUDIT_V1.json';assert not out.exists()
result=dict(initial_row_byte_crops=boundary_crops,records=records,current_token_lookup=dict(coverage=covered/len(rr),squared_error_ratio_vs_leave_document_out_mean=err/baseerr,covered_squared_error_ratio=covered_err/covered_base),scope='Exploratory reused-panel annotation via incremental UTF8 decoding of whole prefixes. Initial row-boundary continuation bytes are replaced for annotation; strict validation after stripping them passes. First strict attempt failed before output because a row began mid-codepoint; documents and effects unchanged. Pending bytes differ from an individual token decoding to replacement characters. Next-token continuation label uses the realized next token, so is descriptive, not a predictive input feature. Token lookup leaves recipient document out and falls back to other-document mean for unseen tokens.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
