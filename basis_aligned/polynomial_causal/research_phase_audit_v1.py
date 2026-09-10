"""Audit only closed recorded intervals; missing time is explicit."""
from datetime import datetime
from pathlib import Path
import json,sys


def audit(paths,start,end):
    start=datetime.fromisoformat(start);end=datetime.fromisoformat(end);intervals=[]
    for path in paths:
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for a,b in zip(rows,rows[1:]):
            if a.get('phase')=='turn_boundary':continue
            left=max(start,datetime.fromisoformat(a['utc']));right=min(end,datetime.fromisoformat(b['utc']))
            if right>left:intervals.append({'source':str(path),'category':a['category'],'start':left.isoformat(),'end':right.isoformat(),'seconds':(right-left).total_seconds()})
    intervals.sort(key=lambda r:r['start']);overlaps=[]
    for a,b in zip(intervals,intervals[1:]):
        if datetime.fromisoformat(b['start'])<datetime.fromisoformat(a['end']):overlaps.append([a['source'],b['source']])
    assert not overlaps,'Overlapping clocks cannot be summed as disjoint time'
    buckets={}
    for r in intervals:buckets[r['category']]=buckets.get(r['category'],0)+r['seconds']
    window=(end-start).total_seconds();covered=sum(buckets.values())
    return {'bucket_seconds':buckets,'recorded_seconds':covered,'unrecorded_seconds':window-covered,
            'coverage_fraction':covered/window,'intervals':intervals,
            'scope':'Recorded labels only. Merged or late-start phases cannot establish a complete ceremony budget.'}

if __name__=='__main__':
    # output startISO endISO files...
    result=audit(sys.argv[4:],sys.argv[2],sys.argv[3]);out=Path(sys.argv[1])
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='intervals'}))
