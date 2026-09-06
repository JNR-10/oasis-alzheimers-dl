# Longitudinal Deep Learning for Alzheimer's Disease Detection from Structural MRI

## Team

- [Jainil]
- [Mohit]
- [Nirupam]

## Course and Track

**Deep Learning - Fall 2026**  
San Jose State University  
**Selected track:** Option 1 - Modern Deep Learning Pipeline (Training + Deployment)

## Project Status

This repository is the dedicated Fall 2026 deep-learning project repository. It currently contains the project proposal, literature review, feasibility audit, and planned implementation structure. Model implementation and training will begin after OASIS-3 access and cohort feasibility are confirmed.

## Abstract

This project studies Alzheimer's disease detection from longitudinal 3D structural MRI. The system will compare conventional volumetric baselines with longitudinal models that learn a shared 3D representation for each visit and use temporal attention or a Transformer to model change over time. Elapsed time between visits will be provided to the temporal module. Anatomy-guided auxiliary supervision will also be investigated using structural targets such as hippocampal and ventricular measurements. These auxiliary targets will be used during training only; MMSE and similar cognitive-test scores will not be used as inference features.

## Motivation

Earlier machine-learning work showed that structural MRI contains useful signal for Alzheimer's-related classification, while also exposing the risk of shortcut learning when a cognitive assessment such as MMSE is included as a predictive feature. This project keeps the inference pathway imaging-centered and addresses two further limitations: hand-engineered imaging representations and cross-sectional modeling that treats visits as unrelated examples.

The central question is whether longitudinal structural change provides reliable predictive value beyond a controlled single-scan model, and whether anatomy-guided training improves representation quality without introducing label or measurement leakage.

## Problem Formulation

- **Input:** One or more longitudinal 3D T1-weighted MRI volumes from the same subject, together with elapsed time between visits.
- **Output:** A probability that the subject shows Alzheimer's-related cognitive impairment at a target visit.
- **Inference restriction:** MMSE and similar cognitive-test scores will not be passed to the model as inference features. Clinical labels may be used as training targets.
- **Constraints:** OASIS-3 access and approval timing, irregular visits, missing labels, class imbalance, 3D GPU memory, and subject-level leakage must be handled explicitly.

## Proposed Approach

The proposed system will use:

1. A shared 3D CNN or 3D ResNet encoder for each MRI visit.
2. A temporal attention or Transformer module over visit embeddings.
3. Elapsed-time information so that irregular visit spacing is modeled explicitly.
4. An anatomy-guided multi-task head that predicts selected structural quantities during training.
5. A final MRI-based classification pathway that does not require MMSE at inference time.

The project will compare the proposed model with a conventional single-scan 3D baseline and recent volumetric or longitudinal approaches. The candidate methods and their code availability are documented in the [literature and SOTA survey](docs/literature_sota_survey.md).

## Dataset and Data Policy

The intended dataset is [OASIS-3](https://sites.wustl.edu/oasisbrains/home/oasis-3/), which provides longitudinal neuroimaging and associated clinical observations. Access, approval, and cohort-construction timing are explicit feasibility risks. Raw imaging data, protected health information, checkpoints, and large generated artifacts must not be committed to this repository.

All visits from a subject must remain in exactly one of the train, validation, or test partitions. Preprocessing is expected to include orientation standardization, registration, resampling, brain extraction, intensity normalization, fixed-size cropping or padding, and conservative 3D augmentation.

## Evaluation Plan

Primary metrics will be ROC-AUC, balanced accuracy, macro-F1, impaired-class sensitivity, specificity, and confusion matrices. The two core ablations are:

1. Single-scan model versus the same encoder with longitudinal temporal modeling.
2. Longitudinal model without anatomy-guided supervision versus the same model with auxiliary anatomical loss.

All principal comparisons will use the same subject-level held-out test set wherever possible. Repeated runs or bootstrap confidence intervals will be used when feasible so that small differences are not over-interpreted. The [AI novelty and feasibility audit](docs/ai_novelty_feasibility_audit.md) records the project's main technical, data, compute, and evaluation risks.

## Repository Structure

```text
oasis-alzheimers-dl/
├── README.md
├── dl/
│   ├── README.md                     # Planned implementation organization
│   ├── .gitignore                    # Imaging, weights, and generated-output rules
│   └── data/README.md                # OASIS-3 access and data-use notes
└── docs/
    ├── proposal/
    │   ├── project_proposal.pdf
    │   └── project_proposal.tex
    ├── literature_sota_survey.md
    └── ai_novelty_feasibility_audit.md
```

## Project Documents

- [Course proposal PDF](docs/proposal/project_proposal.pdf)
- [Course proposal LaTeX source](docs/proposal/project_proposal.tex)
- [Literature and SOTA survey](docs/literature_sota_survey.md)
- [AI novelty and feasibility audit](docs/ai_novelty_feasibility_audit.md)

## Responsible Use

This is an academic research project. The planned model is not a clinical diagnostic device. Results will be interpreted as research findings and reported with limitations, cohort details, leakage controls, and uncertainty estimates.
