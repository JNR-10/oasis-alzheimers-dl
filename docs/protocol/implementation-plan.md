# Implementation Plan

Work is tracked as [GitHub issues](https://github.com/JNR-10/oasis-alzheimers-dl/issues),
one per slice, each declaring the issues that block it. This document holds what the
issues cannot: the calendar, who runs which track, and the gates where scope gets cut.

The ordering principle behind the ticket breakdown: **build the whole pipeline against a
synthetic cohort first.** Almost nothing is genuinely blocked on real data, and a team
that waits for it loses weeks it does not have. Swapping in a real cohort source at the
end is a smaller job than starting a pipeline.

**Access is no longer the constraint — compute is.** OASIS-3 and OASIS-2 are both in
hand as of 2026-11-22. What binds now is GPU time, which
[ADR-0010](../adr/0010-subsampled-cohort-and-reduced-resolution.md) answers by training
on 300 sampled subjects at 2 mm / 96³ with 3 visits per sequence — roughly an order of
magnitude cheaper per run than the original configuration.

The decision that saves the most is an ordering one: **sample subjects before
preprocessing, not after.** Preprocessing is the dominant one-time CPU cost, and
processing scans that will never be trained on spends all of it for nothing.

## Three tracks

Tickets are grouped so three people can work without queueing behind each other. Claim a
track, not a ticket.

| Track | Owner | Owns | Tickets |
|---|---|---|---|
| **A — Data** | `mohiitt` | Access, cohort construction, preprocessing | [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1), [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2), [#13](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13), [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14), [#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15), [#16](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16), [#17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17) |
| **B — Model** | `JNR-10` | Encoder, temporal module, training loop, arms | [#3](https://github.com/JNR-10/oasis-alzheimers-dl/issues/3), [#5](https://github.com/JNR-10/oasis-alzheimers-dl/issues/5), [#7](https://github.com/JNR-10/oasis-alzheimers-dl/issues/7), [#8](https://github.com/JNR-10/oasis-alzheimers-dl/issues/8), [#9](https://github.com/JNR-10/oasis-alzheimers-dl/issues/9), [#10](https://github.com/JNR-10/oasis-alzheimers-dl/issues/10), [#11](https://github.com/JNR-10/oasis-alzheimers-dl/issues/11) |
| **C — Protocol** | `NiruSJSU` | Splits, metrics, provenance, reporting | [#4](https://github.com/JNR-10/oasis-alzheimers-dl/issues/4), [#6](https://github.com/JNR-10/oasis-alzheimers-dl/issues/6), [#12](https://github.com/JNR-10/oasis-alzheimers-dl/issues/12), [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21), [#24](https://github.com/JNR-10/oasis-alzheimers-dl/issues/24) |

The experiment runs ([#18](https://github.com/JNR-10/oasis-alzheimers-dl/issues/18)–[#20](https://github.com/JNR-10/oasis-alzheimers-dl/issues/20)) go to `JNR-10`, who builds the arms — running
them is the natural extension of building them. [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21) then hands the aggregation and
reporting to `NiruSJSU`, so the person who produced the numbers is not the only person
who reads them.

[#22](https://github.com/JNR-10/oasis-alzheimers-dl/issues/22) and [#23](https://github.com/JNR-10/oasis-alzheimers-dl/issues/23) are **deliberately unassigned**. They are the first to cut, and
assigning work that is planned to be dropped manufactures a commitment nobody should
feel. They get claimed only if Gate 3 passes with time to spare.

### Load, by week

The tracks are not evenly loaded over time, and the mismatch is worth planning around
rather than discovering:

| Track | Weeks 1–2 | Weeks 3–5 |
|---|---|---|
| A — `mohiitt` | Light — only [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1) and [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2), both small | Heavy — the cohort and preprocessing gate everything downstream |
| B — `JNR-10` | Heavy — the longest internal chain, ending in [#8](https://github.com/JNR-10/oasis-alzheimers-dl/issues/8) | Moderate — running arms that already work |
| C — `NiruSJSU` | Moderate — [#4](https://github.com/JNR-10/oasis-alzheimers-dl/issues/4) and [#6](https://github.com/JNR-10/oasis-alzheimers-dl/issues/6) need no data | Heavy — [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21) and [#24](https://github.com/JNR-10/oasis-alzheimers-dl/issues/24) land in the final week |

`mohiitt` should float onto Track B or C for the first fortnight. Track A has the least
to do early and the most to do under Gate 2 pressure, so the spare capacity is real and
is best spent where the week-2 bottleneck is.

**Amended 2026-11-22.** Access arrived in week 3, which pulls Track A's heavy work
forward rather than leaving it stacked against Gate 2. [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1)
is closed, [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2) is unblocked and
small, and the download plus the subject sample can start immediately. The ordering that
matters: **draw the sample before preprocessing anything** — processing scans that the
sample will discard spends the compute saving the whole plan now rests on.

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

**Gate 2 — end of week 3. Does the cohort support a longitudinal claim?**
Access is settled, so this gate asks the question that still decides the project: how
many sampled subjects have two or more usable, labelled T1 visits
([#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14))?

If that number is too small, no architecture rescues it, and the honest response is to
report it as a finding about cohort feasibility rather than to train the arms anyway.
Two levers before that: raise `max_subjects` if GPU time allows, or trigger the
[ADR-0003](../adr/0003-oasis3-primary-oasis2-fallback.md) fallback to OASIS-2 and record
it in [`../data/access-log.md`](../data/access-log.md) with date and cause. Raising the
sample produces a new cohort rather than a bigger one, so it is a restart of anything
already trained — cheap in week 3, not in week 4.

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

  **Amendment, 2026-11-24.** ADR-0004 accepted: max gap **180 days**, unassessed
  sessions never labeled and kept only as history, every labeled visit is a target. Chosen from the real T1w↔CDR gap
  distribution (median 92 days); 90 days left fewer than 300 multi-visit subjects.
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

---

## Amendment, 2026-11-22 — the encoder is chosen, and was nearly chosen badly

**The question.** While implementing the visit encoder (#5), the configured backbone
was a randomly initialized `resnet18` at 33.2M parameters — heavy for a cohort of 300
subjects, which is roughly 450 training sequences once built (#14 reports the real
count; `docs/data/cohort-definition.md` holds `_tbd_` until then). `resnet10` is under
half the size. Whichever is chosen must be identical across every arm, or ablation 1
confounds architecture with capacity.

**The answer** is [ADR-0011](../adr/0011-pretrained-vit-encoder.md), and it took
neither option. Both were randomly initialized, so both traded capacity against
generalization. The configured encoder is now a **3D ViT-B/16 initialized from
BrainIAC's self-supervised weights** (88.3M parameters), which sidesteps the trade: the
representation comes from ~32k brain MRIs rather than from 450 training sequences.
`resnet18` is kept as the from-scratch reference and still runs.

Three consequences for the plan as written:

- **The encoder is selected once at the config root** (`configs/encoder/`), not inside
  each model config. The requirement that it be identical across every arm is enforced
  by `tests/unit/test_configs.py` rather than by discipline.
- **BrainIAC's pretraining corpus includes OASIS-3 and OASIS-2.** The reasoning for
  proceeding anyway, and the alternatives rejected, are in ADR-0011. The short version:
  it is self-supervised, so no label leaks, and it is constant across arms, so it
  cancels in the contrasts. Absolute numbers are not comparable with from-scratch
  literature and the report must say so.
- **Preprocessing now has a fixed target.** BrainIAC's template and a resize to 96³,
  rather than a free choice of 2 mm resampling — see the dated amendment in
  [`preprocessing.md`](preprocessing.md). This lands on [#15] and constrains, but does
  not remove, the registration question in [#16].

### Where the numbers live

Parameter counts follow from the architecture, so they can be stated — but they are
*checked*, not trusted: `tests/unit/test_configs.py` reads ADR-0011's table and measures
each encoder as configured. That test exists because the first version of that table
was typed, and two of its four figures were measured at the wrong embedding width and
input shape. Correcting them then exposed a third problem: MONAI 1.4's `ViT` allocates a
cross-attention layer in every block and never runs it — 28.3M of ViT-B's 116.7M
parameters. The encoder now removes those after construction (the forward pass is
bit-identical, and a test pins that), which brings ViT-B to 88.3M: the figure BrainIAC
publishes for its own encoder, and the architecture its checkpoint actually fits.

The memory and throughput footprint is machine-specific, so it is **not** reproduced
here. It is measured by `tests/unit/test_encoder.py::TestMemoryFootprint` and
regenerated with `make test-all`; it is excluded from CI because 3D volumes at the
production shape take minutes per backbone.

### One new cost, and where it lands

`vit_base_16` at 96³ is over a second per forward pass on a laptop CPU, so the real arms
([#18], [#19], [#20]) need a GPU to be worth starting. `configs/encoder/vit_tiny_smoke.yaml`
is the same code path at 2.2M parameters, for proving a training loop runs end to end on
any machine before it is queued on real hardware. No reported result comes from it, and
the encoder name is carried into every run record so the two cannot be confused.

[#15]: https://github.com/JNR-10/oasis-alzheimers-dl/issues/15
[#16]: https://github.com/JNR-10/oasis-alzheimers-dl/issues/16
[#18]: https://github.com/JNR-10/oasis-alzheimers-dl/issues/18
[#19]: https://github.com/JNR-10/oasis-alzheimers-dl/issues/19
[#20]: https://github.com/JNR-10/oasis-alzheimers-dl/issues/20
