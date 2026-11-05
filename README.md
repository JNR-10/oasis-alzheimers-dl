# Longitudinal Deep Learning for Alzheimer's Disease Detection from Structural MRI

Does longitudinal structural change improve Alzheimer's detection over a controlled
single-scan model — and does anatomy-guided auxiliary supervision improve the
representation — when cognitive test scores are kept out of the inference pathway and
subjects never cross a partition boundary?

**CMPE-258 Deep Learning, Fall 2026 · San Jose State University**
Jainil Narendrakumar Rana · Nirupam Narayana Maladi · Mohit Manoj Barade

---

## Status

**Pre-data.** The repository holds the proposal, the literature survey, the feasibility
audit, the experimental protocol, and the implementation scaffold. No model has been
trained and no result exists.

The critical path is OASIS-3 access, tracked in
[`docs/data/access-log.md`](docs/data/access-log.md). Every module under
`src/oasis_dl/` is a typed interface with a documented contract and an unimplemented
body — deliberately, so that the structure can be reviewed before the code commits to
it, and so nothing here can be mistaken for a working pipeline.

## The study

The predecessor project ([oasis-alzheimers-ml](https://github.com/JNR-10/oasis-alzheimers-ml),
CMPE-257) found that its tabular models were MMSE proxies: a model using **MMSE alone —
one feature — matched the full eight-feature baseline at 86.9% accuracy**. Adding 104
MRI-derived features cut MMSE's importance from 37.8% to 9.1% and still reached 85.7%
without it, which established that structural MRI carries independent signal. It left
two limitations, named in its own future work: hand-engineered features, and
cross-sectional modeling on a dataset that cannot support anything else.

This project takes both. It learns volumetric representations end-to-end and models
change across visits, on OASIS-3.

| Arm | Model | Role |
|---|---|---|
| B1 | 3D ResNet, single scan | Transparent baseline |
| B2 | 3DSC-TF | Recent 3D convolution–Transformer comparison |
| B3 | LongFormer | Recent longitudinal comparison |
| **PA** | Shared visit encoder + temporal module | Treatment for ablation 1 |
| **PB** | PA + anatomy-guided auxiliary head | Treatment for ablation 2 |

Two ablations carry the claim: **B1 vs PA** isolates longitudinal information, **PA vs
PB** isolates the auxiliary objective. Each varies one component with cohort, split,
preprocessing, and evaluation code held fixed.

The novelty claim is deliberately modest. Alzheimer's classification from MRI is a
crowded problem and LongFormer (WACV 2024) already does longitudinal 3D MRI. What is
offered here is the controlled comparison — see
[`docs/audits/`](docs/audits/ai_novelty_feasibility_audit.md).

## Three controls

1. **Cognitive scores never reach inference.** MMSE may inform labels; no model reads
   one when predicting. ([ADR-0002](docs/adr/0002-no-cognitive-scores-at-inference.md))
2. **Partitions are assigned by subject**, drawn once, never redrawn. No person's
   visits, derivatives, or normalization statistics cross a boundary.
   ([ADR-0001](docs/adr/0001-subject-level-partitioning.md))
3. **Elapsed time is modeled, not assumed.** Visit spacing is irregular and is given to
   the model as a continuous input rather than resampled onto a grid.
   ([ADR-0006](docs/adr/0006-elapsed-time-as-explicit-input.md))

## Layout

```text
├── GLOSSARY.md              Fixed vocabulary — code and docs use these words
├── docs/
│   ├── adr/                 Why it is built this way (8 records)
│   ├── data/                Dataset card, access log, cohort definition
│   ├── protocol/            What will be run, and how it is judged
│   ├── survey/              Literature and SOTA review
│   ├── audits/              Novelty and feasibility audit
│   └── proposal/            Course submission
├── src/oasis_dl/            The package — interface seams, unimplemented
│   ├── data/                Manifest, cohort construction, splits, datasets
│   ├── preprocessing/       MRI pipeline, anatomy target extraction
│   ├── models/              Visit encoder, temporal module, heads, arms
│   ├── training/            Loop, combined objective, seeding
│   ├── evaluation/          Metrics, subject-level bootstrap
│   └── cli/                 Hydra entrypoints
├── configs/                 Composable experiment configs
├── tests/                   Unit and integration (empty until there is behaviour)
├── experiments/             Per-run records: config, commit, seed, metrics
└── data/ results/           Gitignored. Nothing here is ever committed.
```

## Getting started

```bash
make setup                   # venv + editable install + pre-commit
export OASIS_ROOT=/path/to/local/oasis   # never inside the repo
make help                    # every pipeline entrypoint
```

An ablation is one override:

```bash
python -m oasis_dl.cli.train experiment=proposed_b_anatomy
```

If a comparison needs more than one override to express, the configs are hiding an
uncontrolled difference between the arms.

## Data policy

OASIS data is governed by a Data Use Agreement and is **not** redistributed here.
Imaging volumes, derivatives, subject-identifying manifests, and model weights never
enter version control — enforced by `.gitignore`, a pre-commit size hook, and a CI job
that fails on tracked scans or checkpoints. Reproducing this work means applying for
your own access; this repository ships the code and the cohort *definition*.

## Responsible use

Academic research, not a clinical diagnostic device. Results will be reported with
cohort details, leakage controls, uncertainty estimates, and limitations. Negative and
null results go in the report with the same prominence as positive ones — the ablations
are designed to be informative when they come back negative.

## Documents

- [Experimental protocol](docs/protocol/README.md) — the plan, written before results
- [Literature and SOTA survey](docs/survey/literature_sota_survey.md)
- [AI novelty and feasibility audit](docs/audits/ai_novelty_feasibility_audit.md)
- [Proposal PDF](docs/proposal/project_proposal.pdf) · [LaTeX source](docs/proposal/project_proposal.tex)
- [Predecessor project](https://github.com/JNR-10/oasis-alzheimers-ml) (CMPE-257)
