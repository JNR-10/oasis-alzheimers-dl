# Implementation Plan

Work is tracked as [GitHub issues](https://github.com/JNR-10/oasis-alzheimers-dl/issues),
one per slice, each declaring the issues that block it. This document holds what the
issues cannot: the calendar, who runs which track, and the gates where scope gets cut.

The ordering principle behind the ticket breakdown: **build the whole pipeline against a
synthetic cohort first.** OASIS-3 approval is pending and its arrival date is not ours to
control, but almost nothing is genuinely blocked on it. A team that waits loses two to
three weeks of five. When data lands, the work is swapping in a cohort source, not
starting a pipeline.

## Three tracks

Tickets are grouped so three people can work without queueing behind each other. Claim a
track, not a ticket.

| Track | Owns | Tickets |
|---|---|---|
| **A — Data** | Access, cohort construction, preprocessing | [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1), [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2), [#13](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13), [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14), [#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15), [#16](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16), [#17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17) |
| **B — Model** | Encoder, temporal module, training loop, arms | [#3](https://github.com/JNR-10/oasis-alzheimers-dl/issues/3), [#5](https://github.com/JNR-10/oasis-alzheimers-dl/issues/5), [#7](https://github.com/JNR-10/oasis-alzheimers-dl/issues/7), [#8](https://github.com/JNR-10/oasis-alzheimers-dl/issues/8), [#9](https://github.com/JNR-10/oasis-alzheimers-dl/issues/9), [#10](https://github.com/JNR-10/oasis-alzheimers-dl/issues/10), [#11](https://github.com/JNR-10/oasis-alzheimers-dl/issues/11) |
| **C — Protocol** | Splits, metrics, provenance, reporting | [#4](https://github.com/JNR-10/oasis-alzheimers-dl/issues/4), [#6](https://github.com/JNR-10/oasis-alzheimers-dl/issues/6), [#12](https://github.com/JNR-10/oasis-alzheimers-dl/issues/12), [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21), [#24](https://github.com/JNR-10/oasis-alzheimers-dl/issues/24) |

The experiment runs ([#18](https://github.com/JNR-10/oasis-alzheimers-dl/issues/18)–[#20](https://github.com/JNR-10/oasis-alzheimers-dl/issues/20)) are shared
— by then the pipeline exists and the work is running it, watching it, and reading what
comes out.

Fill in names once tracks are claimed:

| Track | Owner |
|---|---|
| A — Data | |
| B — Model | |
| C — Protocol | |

## Calendar

Five weeks. Dates are the Thursday of each week.

| Week | Dates | Target |
|---|---|---|
| 1 | Nov 5–11 | Startable tickets done: synthetic cohort, metrics, encoder, derivative question answered |
| 2 | Nov 12–18 | **B1 green on synthetic** ([#8](https://github.com/JNR-10/oasis-alzheimers-dl/issues/8)) — the pipeline proven end to end |
| 3 | Nov 19–25 | PA and PB green on synthetic; data arrives; cohort frozen |
| 4 | Nov 26–Dec 2 | Preprocessing done, B1 and PA on real data, **ablation 1 answered** |
| 5 | Dec 3–9 | Ablation 2, multi-seed runs, report |

**Week 4 contains Thanksgiving (Thursday Nov 26, 2026)** and is both the tightest week
and a short one. It carries ablation 1 — the result the project most needs. Plan week 3
to finish early rather than assuming week 4 has five working days in it.

## Gates

Three points where the plan gets cut rather than extended. Each has a check and a
consequence, decided on the date rather than negotiated when it arrives.

**Gate 1 — end of week 2. Is B1 green on synthetic?**
If not, the pipeline is the bottleneck, not the science. Drop
[#22](https://github.com/JNR-10/oasis-alzheimers-dl/issues/22) and [#23](https://github.com/JNR-10/oasis-alzheimers-dl/issues/23) immediately and move Track
A onto Track B's work until it is.

**Gate 2 — end of week 3. Has OASIS-3 access arrived?**
If not, trigger the [ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md) fallback to
OASIS-2 and record it in [`../data/access-log.md`](../data/access-log.md) with date and
cause. OASIS-2 is far smaller and resets the baselines; it does not inherit numbers. A
fallback declared in week 3 is recoverable, in week 4 it is not.

**Gate 3 — end of week 4. Is ablation 1 answered?**
If not, cut everything optional and run [#20](https://github.com/JNR-10/oasis-alzheimers-dl/issues/20) and
[#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21) only. One core ablation reported properly beats two reported
badly.

The full cut order is in [`README.md`](README.md) under *Scope under pressure*. The last
thing to go is the controlled comparison: a well-controlled negative result is a finding,
an uncontrolled positive one is not.

## Decisions that must land before training

Two, both gating and both easy to let slide:

- [**#13**](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13) — the label rule, moving
  [ADR-0004](../adr/0004-label-definition-and-target-visit.md) from proposed to accepted.
  A label rule chosen after seeing results is not a label rule, so this is checkable
  against commit dates.
- [**#16**](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16) — the registration target. Registering every scan
  independently to a template may absorb the within-subject atrophy the temporal module
  exists to detect. Settle it on a handful of subjects before the cohort is processed;
  reprocessing afterwards silently invalidates anything already trained.

## The number that decides viability

[**#14**](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14) reports the visit-count distribution: how many subjects
have two or more usable, labelled T1 visits. Not the headline dataset size — this number.

A cohort that is overwhelmingly single-visit cannot support a longitudinal claim
regardless of architecture. If that is what the data says, the team needs to know at Gate
2 and not in week 5, and the honest response is to report it as a finding about cohort
feasibility rather than to train the arms anyway.
