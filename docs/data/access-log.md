# Data Access Log

The critical path. OASIS-3 access gates every experiment in this project, so its status
is tracked here rather than remembered.

**Fill this in. The dates below are placeholders and nothing here has been verified.**

## OASIS-3

| Field | Value |
|---|---|
| Application submitted | _unrecorded_ |
| Submitted by | _unrecorded_ |
| Approval received | _pending_ |
| DUA accepted on | _pending_ |
| Download completed | _pending_ |
| Local storage path | recorded in `.env`, not in git |
| Approximate size on disk | _unrecorded_ |

## Fallback trigger

[ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md) makes the fallback to OASIS-2
a declared event. If it fires, record here: the date, the cause, who decided, and what
scope changed. The report then names the fallback rather than presenting OASIS-2 results
as the original plan.

| Field | Value |
|---|---|
| Fallback triggered | no |
| Date | — |
| Cause | — |

## Why this file exists

The feasibility audit names the project's dominant failure mode as an over-scoped plan
that cannot secure data in time. A date in a file is harder to lose than a date in a
conversation, and the decision to cut scope is only cheap while there is still time to
make it.
