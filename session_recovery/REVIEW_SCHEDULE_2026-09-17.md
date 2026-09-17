# Active reviews on the September 17 instance

Supervisor `cron` is running. `/etc/cron.d/bilin18-reviews` schedules strategy
reviews at minute 52 every hour and mathematical reviews at 02:58, 05:58,
08:58, 11:58, 14:58, 17:58, 20:58, 23:58 UTC. The hourly entry next fires
21:52 UTC. The mathematical entry next fires 23:58 UTC. An immediate hourly
review completed and its receipt verified; an immediate mathematical review also completed and its receipt verified. Scheduled reviews skip duplicate recent
receipts, so these entries may confirm a current review rather than write another.

Prompts apply better_circuits.md and communicating_results.md. Hourly reviews
must report measured timing and a systematic process-improvement assessment.
Each invocation is bounded to review plus a small CPU consequence, shares a lock,
and cannot enqueue GPU jobs or edit the primary agent's active experiment.

Runner: `scheduled_review.sh`; deployable cron file: `bilin18-reviews.cron`.
Logs: `/root/.local/state/bilin18/logs`; cron wrapper output:
`/var/log/bilin18-reviews.log`. These schedules require this instance to remain
running. They produce review files/board entries, not ChatGPT notifications.

Official implementation reference: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).
