"""Report-only decomposition of signed CE change into harm and improvement."""
import hashlib,json
from pathlib import Path


def main():
    p=Path(__file__).resolve().parent;src=p/'CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT.json'
    r=json.loads(src.read_text());reports={}
    for name,row in r['reports'].items():
        reports[name]={}
        for arm,removal in row['removal'].items():
            values=removal['ce_damage_per_row'];n=len(values)
            harm=sum(max(v,0) for v in values)/n
            improvement=sum(max(-v,0) for v in values)/n
            signed=sum(values)/n;absolute=sum(abs(v) for v in values)/n
            assert abs(harm-improvement-signed)<1e-12 and abs(harm+improvement-absolute)<1e-12
            reports[name][arm]={'mean_harm':harm,'mean_improvement_magnitude':improvement,
                'mean_signed_change':signed,'mean_absolute_change':absolute,
                'harm_count':sum(v>0 for v in values),'improvement_count':sum(v<0 for v in values),
                'max_harm':max(values),'max_improvement_magnitude':max(-min(values),0)}
    result={'schema':'report.loss_cancellation.v1','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'reports':reports,
        'scope':'CPU arithmetic on immutable native observations for requested explanation. Descriptive only; registered absolute-preservation criterion and failed verdict unchanged.'}
    with (p/'REPORT_LOSS_CANCELLATION_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps({'C_projector':reports['C']['projector'],'A1_complement':reports['A1']['complement']},indent=2))


if __name__=='__main__':main()
