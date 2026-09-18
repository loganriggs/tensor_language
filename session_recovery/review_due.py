"""Avoid starting a scheduled review before its previous receipt's interval ends."""
import datetime as dt
import math
from pathlib import Path

def remaining_seconds(repo,kind,now=None):
    prefix,hours=('HOURLY_STRATEGIC_REVIEW_',1) if kind=='hourly' else ('THREE_HOURLY_MATHEMATICAL_REVIEW_',3)
    records=sorted((Path(repo)/'basis_aligned/polynomial_causal').glob(prefix+'*.md'))
    if not records:return 0
    stamp=dt.datetime.strptime(records[-1].stem[len(prefix):],'%Y-%m-%d_%H%M').replace(tzinfo=dt.timezone.utc)
    # Filenames round down to a minute. One minute covers the actual receipt second.
    due=stamp+dt.timedelta(hours=hours,minutes=1)
    return max(0,math.ceil((due-(now or dt.datetime.now(dt.timezone.utc))).total_seconds()))

if __name__=='__main__':
    import sys,time
    repo,kind=sys.argv[1:];assert kind in ['hourly','mathematical']
    start=time.monotonic()
    while (remaining:=remaining_seconds(repo,kind)):
        if time.monotonic()-start>3300:raise SystemExit('Review due-time wait exceeded55minutes; inspect recent concurrent review')
        time.sleep(min(30,remaining))
