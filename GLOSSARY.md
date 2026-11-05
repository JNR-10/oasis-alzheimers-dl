# Glossary

The vocabulary of this project. Terms here are fixed: code, configs, documents,
and the final report use these words and no synonyms. Where several words are in
circulation for one concept, the one listed is canonical and the alternatives are
marked `_Avoid_`.

This file defines meaning only. Decisions live in [`docs/adr/`](docs/adr/) and
procedures live in [`docs/protocol/`](docs/protocol/).

## Cohort and sampling

**Subject**:
One person in the dataset. The unit of partitioning — every artifact derived from
a subject belongs to exactly one partition.
_Avoid_: patient, participant, case

**Session**:
One occasion on which data was acquired for a subject, imaging or clinical. OASIS
numbers sessions by days elapsed from the subject's entry into the study.
_Avoid_: timepoint, event

**Visit**:
An imaging session that passed the inclusion rules and carries a resolvable label.
The unit the model consumes. Every visit is a session; most sessions are not visits.
_Avoid_: timepoint, observation, sample

**Scan**:
The 3D T1-weighted image acquired at a visit.
_Avoid_: image, volume, MRI (as a count noun)

**Visit sequence**:
A subject's visits in chronological order, as presented to the temporal module.
_Avoid_: trajectory, series, time series

**Target visit**:
The visit whose label the model predicts. A visit sequence ends at its target visit;
later visits are never shown to the model.

**Baseline visit**:
The earliest included visit for a subject. Elapsed time is measured from it.
_Avoid_: first scan, index visit

**Elapsed time**:
Days between two of a subject's visits. Supplied to the temporal module so that
irregular spacing is modeled rather than assumed away.
_Avoid_: interval, delta-t, age gap

**Cohort**:
The fixed set of subjects and visits that survives the inclusion rules. Fixed once
and recorded in a manifest; changing it invalidates every result computed before.

**Inclusion rule**:
A stated, checkable condition a session must meet to become a visit.
_Avoid_: filter, criterion

**Manifest**:
The file that pins the cohort — which subjects, which visits, which partition, which
label. The single source of truth for what was trained on.

## Labels and clinical variables

**CDR**:
Clinical Dementia Rating. The ordinal clinical score (0, 0.5, 1, 2, 3) that the
binary label is derived from.

**MMSE**:
Mini-Mental State Examination. A cognitive screening score, 0–30. Never an inference
feature in this project — see [ADR-0002](docs/adr/0002-no-cognitive-scores-at-inference.md).

**Label**:
The binary supervision target at a target visit: impaired or cognitively normal.
_Avoid_: class, diagnosis, ground truth

**Impaired**:
The positive class — a subject showing Alzheimer's-related cognitive impairment at
the target visit.
_Avoid_: demented, AD, positive, sick

**Cognitively normal**:
The negative class.
_Avoid_: healthy, control, normal (unqualified)

**Inference feature**:
A quantity the model is allowed to read when producing a prediction. The set is
deliberately restricted to imaging and elapsed time.

**Training target**:
A quantity the model is asked to predict during training. Clinical labels and
anatomy targets are training targets; neither is an inference feature. Keeping this
distinction is the central control of the project.

**Anatomy target**:
A structural quantity — hippocampal volume, ventricular volume and similar — used as
auxiliary supervision during training only.
_Avoid_: anatomical feature, ROI feature

**nWBV**:
Normalized whole-brain volume, an OASIS-supplied global atrophy summary.

**eTIV**:
Estimated total intracranial volume, used to normalize regional volumes for head size.

**ASF**:
Atlas scaling factor. Mathematically near-redundant with eTIV.

## Evaluation protocol

**Partition**:
One of train, validation, or test. A subject belongs to exactly one.
_Avoid_: fold, set, split (as a noun for the group)

**Subject-level split**:
Assignment of partitions by subject rather than by visit, so that no person's data
appears in more than one partition. The protocol's load-bearing control — see
[ADR-0001](docs/adr/0001-subject-level-partitioning.md).
_Avoid_: grouped split, patient-wise split

**Leakage**:
Any path by which information about a test subject reaches training. Includes shared
visits, shared derivatives, and normalization statistics fit on the full cohort.

**Shortcut**:
A feature that predicts the label without carrying the signal the project is after.
MMSE is the project's founding example: the predecessor found a single-feature MMSE
model matched its full baseline.
_Avoid_: spurious correlation, confound (which is a distinct idea)

**Circularity**:
The specific shortcut where the model predicts a diagnosis from a measurement that
itself defines that diagnosis.

**Ablation**:
A controlled comparison in which exactly one component changes and cohort, split,
preprocessing, and evaluation code are held fixed.

## Models

**Visit encoder**:
The 3D network that maps one scan to an embedding. Shared — the same weights encode
every visit, so the temporal module compares representations in one space.
_Avoid_: backbone, feature extractor, CNN

**Temporal module**:
The attention or Transformer component that combines a subject's visit embeddings,
conditioned on elapsed time, into one representation for the target visit.
_Avoid_: aggregator, fusion head, RNN

**Auxiliary head**:
The training-only output branch that predicts anatomy targets.

**Single-scan model**:
A model that sees exactly one scan — the target visit's. The control arm of the
first core ablation.
_Avoid_: cross-sectional model, 2D model

**Longitudinal model**:
A model that sees a visit sequence. The treatment arm of the first core ablation.

## Preprocessing

**Derivative**:
Anything computed from a scan rather than acquired — segmentations, registered
volumes, extracted volumes. OASIS ships some derivatives; this project computes others.

**Brain extraction**:
Removing non-brain tissue from a scan.
_Avoid_: skull stripping, defacing (which is a privacy operation, not this)

**Registration**:
Bringing scans into a common spatial frame so that voxel positions are comparable
across subjects and across a subject's visits.

**Intensity normalization**:
Putting voxel intensities on a comparable scale across scans. MRI intensities carry
no absolute units, so this is required rather than optional.
