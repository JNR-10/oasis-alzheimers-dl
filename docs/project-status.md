# Project status — from the start

**Written:** 2026-11-24  
**Course:** CMPE-258 Deep Learning, Fall 2026, San Jose State University  
**Team:** Jainil Narendrakumar Rana (`JNR-10`) · Nirupam Narayana Maladi (`NiruSJSU`) · Mohit Manoj Barade (`mohiitt`)  
**Repo:** [JNR-10/oasis-alzheimers-dl](https://github.com/JNR-10/oasis-alzheimers-dl)

This note is a plain-language snapshot of where the project is, what the dataset
situation is, and what each person has shipped. It is not a results report — no
real-cohort model has been trained yet.

---

## 1. What this project is trying to do

Alzheimer's detection from brain MRI is a crowded research area. The team's earlier
course project (CMPE-257, tabular ML on OASIS-1) found something important:

- A model using **MMSE alone** (one cognitive test score) matched a larger feature set.
- That meant many "MRI models" were partly just reading the cognitive score again.

This project asks a cleaner question on a **longitudinal** dataset (OASIS-3):

1. Does modeling **change across visits** beat a careful **single-scan** baseline?
2. Does an **anatomy auxiliary head** (hippocampus / ventricle volumes during training
   only) improve the representation?

Cognitive scores must **not** be used as model inputs at inference. Subjects must not
leak across train/val/test. Those rules are written as ADRs and are non-negotiable.

### The arms (experiments)

| Arm | Idea | Role |
|---|---|---|
| **B1** | Single-scan baseline (shared encoder, last visit only) | Control for ablation 1 |
| **PA** | Shared encoder + temporal module (uses visit history) | Treatment for ablation 1 |
| **PB** | PA + anatomy auxiliary head (training only) | Treatment for ablation 2 |
| B2 / B3 | 3DSC-TF / LongFormer | Optional external baselines (first to cut) |

Core claims rest on **B1 vs PA** and **PA vs PB**, with the same cohort, split,
preprocessing, and metrics.

---

## 2. How the work was organized

Work is split into three tracks so people do not block each other. Tickets live on
GitHub ([issues](https://github.com/JNR-10/oasis-alzheimers-dl/issues)). Schedule and
ownership live in [`protocol/implementation-plan.md`](protocol/implementation-plan.md).

| Track | Owner | Focus |
|---|---|---|
| **A — Data** | Mohit (`mohiitt`) | Access, derivatives, labels, cohort, preprocessing, anatomy targets |
| **B — Model** | Jainil (`JNR-10`) | Encoder, temporal module, training loop, arms on synthetic data |
| **C — Protocol** | Nirupam (`NiruSJSU`) | Splits, metrics, provenance/reporting; later multi-seed report |

**Strategy:** build the full pipeline on a **synthetic cohort** first, then swap in
real OASIS-3. Waiting for data would waste weeks.

**Calendar (five weeks, Thu week-ends):** roughly Nov 5 → Dec 9, 2026.  
Thanksgiving week (week 4) is short and carries ablation 1 — the most important result.

---

## 3. Dataset status (the critical path)

### What dataset?

- **Primary:** [OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/) — longitudinal
  T1 MRI + clinical assessments (CDR for labels). Large; needs a Data Use Agreement.
- **Fallback:** OASIS-2 — smaller longitudinal set (~18 GB on disk already). Declared in
  [ADR-0003](adr/0003-oasis3-primary-oasis2-fallback.md). Not mixed with OASIS-3.

### Access today

From [`data/access-log.md`](data/access-log.md):

| Item | Status |
|---|---|
| OASIS-3 application | Submitted 2026-11-05 by Mohit |
| OASIS-3 approval | **Granted** (exact date still blank in the log) |
| OASIS-2 access | **Granted** |
| Download of OASIS-3 T1w + FreeSurfer | **In progress** (issue [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1), assigned to Nirupam) |
| PET / CT download | **Not done on purpose** — unused by any arm; huge on disk |
| `OASIS_ROOT` in `.env` | Must be set locally; never committed |

**Binding constraint shifted:** access is no longer the blocker. **Compute** is.
That is why [ADR-0010](adr/0010-subsampled-cohort-and-reduced-resolution.md) exists:

- Train on **300 subjects** (sampled once, with a recorded seed)
- Volumes at **96³** (resize after BrainIAC template registration)
- At most **3 visits** per sequence
- **Sample before preprocessing** so unused scans are never processed

### Derivatives (answered)

OASIS-3 ships **FreeSurfer** outputs (`aseg.stats` / `aparc.stats`), not OASIS-1-style
FSL boxes. Recorded in [ADR-0009](adr/0009-anatomy-targets-from-oasis3-freesurfer.md)
(issue [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2), closed).

### Still open on data

| Issue | Owner | Why it matters |
|---|---|---|
| [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1) Download + record sizes/dates | Nirupam | Unblocks real cohort |
| [#13](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13) Accept ADR-0004 label rule | Mohit | **Accepted 2026-11-24** — 180-day gap; committed in this PR |
| [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14) Freeze cohort + 300-subject sample | Mohit | Hands subject list to preprocessing |
| [#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15) Preprocess sampled subjects | Mohit | N4, HD-BET, BrainIAC template, 96³ |
| [#16](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16) Within-subject vs template-only registration | Mohit | Affects whether atrophy survives registration |
| [#17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17) Anatomy targets on real FreeSurfer | Mohit | Supervises arm PB |

Until [#1] finishes and [#13]/[#14] freeze a real cohort, **no honest OASIS-3 accuracy
number exists**. Synthetic runs prove the pipeline; they are not the study result.

---

## 4. Overall project status (one paragraph)

The repo is past "empty scaffold." The **synthetic end-to-end pipeline works**: generate
fake OASIS-shaped visits → split by subject → train B1 / PA / PB → write provenance →
emit result tables. Real OASIS-3 access is granted and download is underway. Cohort
construction helpers for sampling, sequences, manifests, summaries, and FreeSurfer
parsing exist (Mohit). Label rule ADR-0004 is **accepted** (180-day MRI↔CDR window;
every eligible visit is a target). Preprocessing of real scans has not run. Optional
baselines B2/B3 and the final report are still ahead. T1w imaging download is still
in progress (clinical/tabular ~67 MB complete).

Issue count snapshot (2026-11-27): **11 closed**, **13 open**.

---

## 5. What each member has implemented

### 5.1 Mohit Manoj Barade (`mohiitt`) — Track A (Data)

**Role:** get data decisions right and build the path from raw OASIS → frozen cohort →
preprocessed volumes → anatomy targets.

| Deliverable | Issue / PR | State |
|---|---|---|
| OASIS-3 application recorded in access log | [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1) | Submission logged; download reassigned to Nirupam |
| Confirmed FreeSurfer derivatives; wrote ADR-0009 | [#2](https://github.com/JNR-10/oasis-alzheimers-dl/issues/2) | **Closed** |
| `sample_subjects()` — stratified, seed-stable, order-independent; refuse empty class | [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14) / [PR #27](https://github.com/JNR-10/oasis-alzheimers-dl/pull/27) | **Merged** (plus Jainil hardening commit) |
| `build_sequences()` + immutable manifest read/write | [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14) / [PR #28](https://github.com/JNR-10/oasis-alzheimers-dl/pull/28) | **Merged** |
| `summarize_cohort()` + FreeSurfer anatomy extract/normalize + ADR-0011 preprocess config | [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14)/[#17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17)/[#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15) / [PR #29](https://github.com/JNR-10/oasis-alzheimers-dl/pull/29) | **Merged** |
| ADR-0004 label rule accepted (180-day MRI↔CDR gap) | [#13](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13) | **Accepted** in this PR |

**Still Mohit's open tickets:** [#13](https://github.com/JNR-10/oasis-alzheimers-dl/issues/13)–[#17](https://github.com/JNR-10/oasis-alzheimers-dl/issues/17) (blocked on download / real distributions / sample).

**In simple terms:** Mohit figured out *what* anatomy files OASIS-3 gives, built the
code that *samples subjects*, *builds visit histories*, *writes manifests*, *summarizes
the cohort*, and *parses FreeSurfer volumes* — without inventing fake cohort metrics.
The remaining Track A work needs the actual download.

---

### 5.2 Jainil Narendrakumar Rana (`JNR-10`) — Track B (Model) + shared foundations

**Role:** make the learning pipeline real on synthetic data so the team is not idle
while OASIS downloads.

| Deliverable | Issue / PR | State |
|---|---|---|
| Project scaffold, ADRs, protocol, issue breakdown, contributor `make setup` | early commits | **Merged** |
| ADR-0010 subsample / 96³ / 3-visit compute budget | docs | **Accepted** |
| Pretrained BrainIAC 3D ViT-B/16 shared encoder (ADR-0011) | [PR #26](https://github.com/JNR-10/oasis-alzheimers-dl/pull/26) | **Merged** |
| Synthetic cohort, splits, metrics, bootstrap, temporal module, dataset/collate, three arms training, provenance, generated tables | [#3](https://github.com/JNR-10/oasis-alzheimers-dl/issues/3)–[#12](https://github.com/JNR-10/oasis-alzheimers-dl/issues/12) / [PR #25](https://github.com/JNR-10/oasis-alzheimers-dl/pull/25) | **Merged**; issues **closed** |
| Integration tests joining Mohit's `build_sequences` to domain types | follow-up on #25 | **Merged** |

**In simple terms:** Jainil built the machine that can train B1/PA/PB end-to-end on fake
but OASIS-shaped data, with the scientific controls (subject splits, no MMSE at
inference, provenance, auto-generated tables) enforced in code and tests. That closed
the entire synthetic ticket chain (#3–#12).

---

### 5.3 Nirupam Narayana Maladi (`NiruSJSU`) — Track C (Protocol) + download

**Role (plan):** splits, metrics, provenance, multi-seed aggregation, final report.  
**Role (current assignees):** also owns the **OASIS-3 download** ([#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1)) and the **real-cohort experiment runs** ([#18](https://github.com/JNR-10/oasis-alzheimers-dl/issues/18)–[#20](https://github.com/JNR-10/oasis-alzheimers-dl/issues/20)), plus [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21) and [#24](https://github.com/JNR-10/oasis-alzheimers-dl/issues/24).

| Deliverable | State |
|---|---|
| Merged PRs under `NiruSJSU` | **None yet** (as of 2026-11-24) |
| Track C synthetic pieces (splits #6, metrics #4, provenance #12) | Implemented inside Jainil's PR #25 and closed — Nirupam still owns the *later* reporting tickets |
| [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1) T1w + FreeSurfer download + fill access-log dates/size | **Open / ready** — critical path |
| [#18](https://github.com/JNR-10/oasis-alzheimers-dl/issues/18)–[#20](https://github.com/JNR-10/oasis-alzheimers-dl/issues/20) Train/eval B1, PA, PB on real cohort | **Blocked** on real cohort + GPU |
| [#21](https://github.com/JNR-10/oasis-alzheimers-dl/issues/21) Multi-seed + bootstrap tables | **Blocked** |
| [#24](https://github.com/JNR-10/oasis-alzheimers-dl/issues/24) Final report and figures | **Blocked** |

**In simple terms:** Nirupam's biggest near-term job is finishing the **download and
access-log record**. The protocol/metrics code for the synthetic phase was absorbed into
the pipeline PR; his heavy lift is later — real runs, aggregation, and the report.

---

## 6. What exists in the codebase now

### Done (usable)

- Docs: proposal, survey, audit, protocol, glossary, ADRs 0001–0011
- Synthetic cohort generator (`src/oasis_dl/data/synthetic.py`)
- Subject splits + leakage checks
- Metrics + subject-level bootstrap
- Visit encoder (BrainIAC ViT; ResNet fallback)
- Temporal module with elapsed-time encoding
- Arms B1 / PA / PB training on synthetic data
- Provenance + `make report` from recorded runs
- Cohort helpers: `sample_subjects`, `build_sequences`, manifest I/O, `summarize_cohort` (latter in PR #29)
- Anatomy parser for FreeSurfer stats (PR #29)
- Large automated test suite (~400+ tests on CI)

### Not done (real data path)

- Finish OASIS-3 T1w + FreeSurfer download; fill approval/DUA/size in access log
- ~~Accept ADR-0004~~ (180-day gap; accepted 2026-11-24 — committed in this PR)
- Build and freeze the real 300-subject cohort + manifest
- Run preprocessing (N4 → HD-BET → BrainIAC template → 96³) with QC
- Settle within-subject registration empirically (ADR for #16)
- Extract anatomy targets for every training subject; sanity-check vs predecessor trends
- Train/evaluate on real cohort; multi-seed; final report
- Optional B2/B3 baselines (cut first under pressure)

### Rules the team must not break

1. No cognitive scores at inference ([ADR-0002](adr/0002-no-cognitive-scores-at-inference.md))
2. Subject-level partitions, drawn once ([ADR-0001](adr/0001-subject-level-partitioning.md))
3. Never fabricate metrics — `_tbd_` stays until a real run produces a number ([AGENTS.md](../AGENTS.md))
4. Imaging / manifests / weights never enter git (DUA)

---

## 7. Picture of the pipeline

```text
OASIS-3 download (T1w + FreeSurfer only)     <-- #1 in progress (Nirupam)
        |
        v
Eligible sessions + label rule (ADR-0004)    <-- #13 done (180d); build_visits next
        |
        v
Sample 300 subjects (ADR-0010)               <-- code ready; needs real labels (#14)
        |
        v
Preprocess sampled subjects only             <-- #15/#16 Mohit
        |
        v
Anatomy targets from FreeSurfer              <-- parser ready (#17); needs cohort
        |
        v
Train B1 / PA / PB  ---- synthetic: DONE (Jainil)
                     \-- real: blocked (#18-#20)
        |
        v
Multi-seed + report                          <-- #21/#24 Nirupam
```

---

## 8. Immediate next steps (priority order)

1. **Nirupam:** finish [#1](https://github.com/JNR-10/oasis-alzheimers-dl/issues/1) — T1w + FreeSurfer download; record approval date, DUA date, disk size, `OASIS_ROOT`.
2. **Mohit:** ADR-0004 PR for #13 is up; next is the OASIS `CohortSource` + freeze [#14](https://github.com/JNR-10/oasis-alzheimers-dl/issues/14) sample (metadata can start; imaging sample waits on T1w resume).
3. **Mohit:** [#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15)/[#16](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16) preprocessing + registration ADR on the sample only.
4. **Then:** real-cohort training (#18–#20), multi-seed (#21), report (#24). Cut #22/#23 if time is tight.

---

## 9. Where to read more

| Question | Document |
|---|---|
| What do our words mean? | [`GLOSSARY.md`](../GLOSSARY.md) |
| Why is it built this way? | [`adr/`](adr/) |
| Who owns what / gates? | [`protocol/implementation-plan.md`](protocol/implementation-plan.md) |
| Access / DUA / download | [`data/access-log.md`](data/access-log.md) |
| Cohort tables (still `_tbd_`) | [`data/cohort-definition.md`](data/cohort-definition.md) |
| Agent / contributor rules | [`AGENTS.md`](../AGENTS.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

---

*Update this file when a gate flips (download complete, ADR-0004 accepted, first real
cohort frozen, or first real-arm result recorded). Do not put invented metric numbers
here.*
