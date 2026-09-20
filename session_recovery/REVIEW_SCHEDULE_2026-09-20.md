# Three-hour mathematical and literature reviews

The old cron entry was absent on this instance. Deployable replacement:
`bilin18-mathematical.cron`, installed as `/etc/cron.d/bilin18-mathematical`.
Supervisor cron is running; Codex login status reports authenticated.

Every five minutes the wrapper checks whether three hours have elapsed since the
latest mathematical receipt (plus one minute for filename rounding). If due,
it runs one bounded review, with a nonblocking lock to avoid overlaps. Actual
start can lag the deadline by up to six minutes, or longer if another review is
running. Failed invocations remain due and can retry. No backfilled reviews.

Required sections: live literature search with opened sources; mathematical
assumptions; same-boundary baselines; red-team of both positive and negative
results; executed CPU consequence; organization and efficiency. A completion
receipt must be newly dated, not just an old receipt within the interval.

Logs: `/var/log/bilin18-reviews.log` and `/root/.local/state/bilin18/logs/`.
This operates while the instance is running and CLI authentication works; it
writes repository reviews, not chat notifications. Historical hourly scheduling
is not changed or restored by this three-hour request.

The noninteractive runner follows [official OpenAI documentation](https://developers.openai.com/es-419/docs/non-interactive-mode).

Verification: shell syntax passed; due-time checks cover early and due boundaries;
a live wrapper invocation after the new review returned immediately without
launching another reviewer; installed cron file is root-owned mode0644.
The next review is expected around07:25UTC on20September, subject to runtime
availability. End-to-end execution at that deadline is not yet observed.
