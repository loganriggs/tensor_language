"""Recover exact token-prefix provenance before expanding input-state coverage."""
from pathlib import Path
import hashlib,json,torch
P=Path(__file__).resolve().parent;BQ=P.parents[1]/'bilinear_quotient'
def digest(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def main():
 torch.set_num_threads(2);data=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];records=[];sets=[]
 for p in data:
  tokens=torch.load(BQ/'.rowcache'/p['panel'],weights_only=True);old=tokens[:32,:65];assert digest(old)==p['token_sha256'];hashes=[digest(r) for r in old];sets.append(set(hashes))
  records.append(dict(panel=p['panel'],full_cache_shape=list(tokens.shape),captured_prefixes=32,positions=[0,63],normalized_state_rows=len(p['rows']),token_hash_replay=True,distinct_65token_prefixes=len(set(hashes)),mean_state_squared_radius=float(p['rows'].double().square().sum(1).mean()),document_identity_certified=False))
 first=torch.load(BQ/'.rowcache'/data[0]['panel'],weights_only=True);additional=first[32:96,:65];extra={digest(r) for r in additional}
 result=dict(panels=records,old_panel_exact_prefix_overlap=len(sets[0]&sets[1]),candidate_additional_prefixes=len(additional),candidate_distinct_prefixes=len(extra),candidate_overlap_with_oldcal=len(extra&sets[0]),candidate_overlap_with_eval=len(extra&sets[1]),candidate_token_sha256=digest(additional),candidate_state_rows=64*len(additional),scope='Exacttoken-prefixprovenance only. Different65tokenstrings do not establish documentdisjointness or untouchedhistoricalusage. Candidatecapture expandsoldcalprefix32to96withoutfittingevalpanel or changing positions. No new activations collected.')
 (P/'QUARTIC_CAPTURE_COVERAGE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
