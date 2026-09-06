# AI Novelty & Feasibility Audit

## Overall Assessment

The project is technically appropriate for a graduate deep-learning course, but its novelty should be described as a controlled extension rather than as a first-of-its-kind Alzheimer's MRI system. The strongest case is the continuity from the Spring 2026 ML project: the new work learns from 3D volumes, studies longitudinal progression, keeps MMSE out of inference, and tests anatomy-guided supervision under subject-level evaluation.

The minimum viable project is feasible only if dataset access and cohort construction begin early and the team keeps the core experiment small. A complete implementation of every recent model, a new pretraining run, and deployment should not be treated as mandatory.

## Red-Ocean / Saturation Risk

Plain Alzheimer's classification from MRI is a saturated problem. A project whose only claim is “we trained a CNN on MRI and classified Alzheimer's disease” would have weak novelty because many CNN, Transformer, hybrid, and multimodal systems already report this task. Reported accuracy can also be inflated by random image-level splits, leakage between visits, small cohorts, inconsistent preprocessing, or clinical variables that are close to the target label.

The project should not compete on a single headline accuracy number. It should compete on the quality of the question, the controls, the subject-level evaluation, and the evidence supplied by ablations.

## What Is Not Novel

- Using a 3D CNN or 3D ResNet to classify Alzheimer's-related status from structural MRI.
- Combining a 3D convolutional encoder with a Transformer.
- Using longitudinal MRI for Alzheimer's classification; LongFormer is an explicit recent precedent.
- Using hippocampal or ventricular measurements as informative anatomical signals.
- Reporting an improvement over a weak or inconsistently trained baseline.

The project must not claim to be the first to use longitudinal MRI for Alzheimer's classification.

## What Is Meaningful About This Project

The project has a credible course-level contribution when it does all of the following together:

1. Replaces handcrafted inference features with learned volumetric representations while retaining the previous project's imaging-centered motivation.
2. Models repeated visits and elapsed time explicitly instead of treating scans from one subject as independent examples.
3. Keeps MMSE and similar cognitive-test scores out of the inference feature set, preserving the earlier concern about shortcut or circular prediction.
4. Tests anatomy-guided multi-task learning using structural quantities connected to the prior feature analysis.
5. Uses subject-level train/validation/test splits so no person's visits cross partitions.
6. Proves the value of the two proposed changes through controlled ablations on the same held-out test subjects.

## Technical Novelty / Contribution

The intended architecture is a shared 3D encoder for each visit followed by a temporal attention or Transformer module with elapsed-time information. The second change is objective-level: an auxiliary head predicts selected structural quantities during training, while the inference pathway remains MRI-only.

This is not a claim that either component is individually new. The contribution is the controlled combination and evaluation in the continuation of the existing OASIS-based ML project. The study should report whether longitudinal information and anatomical supervision add value, not assume that they will.

## Dataset and Leakage Risks

- **OASIS-3 access and approval timing:** Access requirements and administrative timing may delay the project. Confirm the current OASIS process early and document the result.
- **Cohort shrinkage:** The number of subjects with usable T1-weighted scans, labels, and multiple visits may be much smaller than the headline dataset size.
- **Irregular visits:** Subjects have different visit counts and intervals. The model and batching strategy must handle missing visits without inventing uniform time points.
- **Missing clinical labels:** Label definitions and target-visit rules must be recorded before training.
- **Subject-level leakage:** Every scan and derived sample from one subject must remain in one partition. Random image-level splits are not acceptable for the main evaluation.
- **Confounding:** Age, scanner, acquisition protocol, site, and preprocessing quality may correlate with the label. Report available cohort balance and avoid presenting a shortcut as disease signal.
- **Label circularity:** Clinical labels may be based partly on cognitive assessments. MMSE should not be passed as an inference input, and the relationship between supervision labels and cognitive tests should be disclosed.

If OASIS-3 access is delayed, the team should document the delay. OASIS-2 may be considered as a clearly identified fallback only if necessary; the dataset change must not be hidden or treated as equivalent.

## Compute and Implementation Risks

- 3D volumes consume substantial GPU memory and make batch sizes small.
- Longitudinal sequences multiply the memory and data-loading cost.
- MRI preprocessing involves orientation, registration, brain extraction, resampling, and intensity normalization; failures can silently change the cohort.
- Public baselines may depend on old Python/PyTorch versions or unavailable checkpoints.
- Small datasets make overfitting likely, especially for Transformer components.
- Scanner/protocol heterogeneity can reduce generalization.
- Pretraining resources may have CT-to-MRI domain shift, restrictive licenses, or large storage requirements.

The first implementation should use a fixed, modest input size, mixed precision where supported, early stopping, deterministic subject-level manifests, and a conventional 3D baseline before adding the temporal or multi-task components.

## Evaluation Risks

Accuracy alone can hide class imbalance and asymmetric clinical error. The primary report should include ROC-AUC, balanced accuracy, macro-F1, impaired-class sensitivity, specificity, and confusion matrices.

The same held-out subject-level test set should be used for the main comparisons. The single-scan and longitudinal ablations should share preprocessing, labels, cohort inclusion rules, and evaluation code. Repeated runs or bootstrap confidence intervals should be used where feasible. A small numerical gain should not be described as meaningful without considering variability and cohort size.

## Semester Feasibility

### Minimum Viable Project

1. Obtain or prepare an approved longitudinal OASIS cohort.
2. Implement and document MRI preprocessing and subject-level splitting.
3. Train a conventional 3D CNN/ResNet baseline.
4. Evaluate at least two recent open-source candidates, prioritizing LongFormer and 3DSC-TF.
5. Implement the proposed shared-encoder longitudinal model.
6. Run the two core ablations: single-scan versus longitudinal, and without versus with anatomy-guided auxiliary loss.
7. Report controlled metrics, cohort counts, limitations, and reproducible configuration details.

### Stretch Goals

These are optional and should not appear to be required for project success:

- auxiliary severity prediction;
- uncertainty calibration;
- saliency or other interpretability analysis;
- pretrained versus randomly initialized encoder comparison;
- a lightweight inference or deployment demo;
- external validation on another dataset.

## Final AI Critique

The proposal is strongest when it is modest. LongFormer already weakens any claim that longitudinal MRI modeling itself is novel, and the general AD-from-MRI task is a red ocean. The defensible project claim is therefore: this work will test, under a leakage-resistant subject-level protocol, whether a learned 3D representation and explicit longitudinal modeling improve on the prior ML foundation, and whether anatomy-guided auxiliary supervision helps without reintroducing cognitive-test shortcuts.

The main failure mode is not insufficient novelty; it is an over-scoped semester plan that cannot secure data, reproduce baselines, and run controlled ablations. If OASIS-3 access, compute, or cohort size is inadequate, the team should reduce the number of baselines and preserve the core ablation design rather than quietly changing the question or overclaiming results.
