# Longitudinal Deep Learning for Alzheimer's Disease Detection from Structural MRI

Does longitudinal structural change improve Alzheimer's detection over a controlled
single-scan model — and does anatomy-guided auxiliary supervision improve the
representation — when cognitive test scores are kept out of the inference pathway and
subjects never cross a partition boundary?

**CMPE-258 Deep Learning, Fall 2026 · San Jose State University**
Jainil Narendrakumar Rana · Nirupam Narayana Maladi · Mohit Manoj Barade

---

## Status

**Pre-training.** Model, dataset, training-loop, and evaluation components exist and
can be exercised on synthetic data. No reportable trained model or result exists yet.

OASIS-3 access is granted ([`docs/data/access-log.md`](docs/data/access-log.md)).
The intended study uses a 300-subject sample at 96³ with up to 3 visits per
sequence ([ADR-0010](docs/adr/0010-subsampled-cohort-and-reduced-resolution.md)).
The label rule, cohort selection, and preprocessing must be resolved before real-data
training; the existing downloads are not the finalized cohort.

The training and evaluation command-line entry points are not implemented yet.

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
├── AGENTS.md                Instructions for coding agents (CLAUDE.md imports it)
├── GLOSSARY.md              Fixed vocabulary — code and docs use these words
├── docs/
│   ├── adr/                 Why it is built this way (8 records)
│   ├── data/                Dataset card, access log, cohort definition
│   ├── protocol/            What will be run, and how it is judged
│   ├── survey/              Literature and SOTA review
│   ├── audits/              Novelty and feasibility audit
│   └── proposal/            Course submission
├── src/oasis_dl/            The package — components and CLI entry points
│   ├── data/                Manifest, cohort construction, splits, datasets
│   ├── preprocessing/       MRI pipeline, anatomy target extraction
│   ├── models/              Visit encoder, temporal module, heads, arms
│   ├── training/            Loop, combined objective, seeding
│   ├── evaluation/          Metrics, subject-level bootstrap
│   └── cli/                 Hydra entrypoints
├── configs/                 Composable experiment configs
├── tests/                   Unit and integration tests
├── experiments/             Per-run records: config, commit, seed, metrics
└── data/ results/           Gitignored. Nothing here is ever committed.
```

## Getting started

```bash
make setup                   # venv + editable install + git hooks
cp .env.example .env         # then set OASIS_ROOT; never inside the repo
make check                   # lint, typecheck, test — the gates CI runs
make help                    # every pipeline entrypoint
```

Working on this with us? [`CONTRIBUTING.md`](CONTRIBUTING.md) is the short path:
branching, the checks to run before pushing, and the three controls a change must not
weaken. Coding agents read [`AGENTS.md`](AGENTS.md).

To exercise the single-scan baseline pipeline on generated data after installation:

```bash
python -m pytest tests/integration/test_pipeline_smoke.py -q
```

This CPU smoke test trains and evaluates ResNet and tiny ViT models on temporary
synthetic data. Its results are not research results and it does not use OASIS scans.
The configured training CLI is not yet runnable.

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
