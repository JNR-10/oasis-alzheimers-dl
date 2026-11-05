# Experimental Protocol

The plan, the metrics, and what would falsify the hypothesis. Schedule, track ownership
and the scope gates live in [`implementation-plan.md`](implementation-plan.md).

Written before results exist. Amendments are appended with dates rather than edited in
place, so the distance between what was planned and what was run stays visible.

## The question

Does **longitudinal structural change** improve Alzheimer's detection over a controlled
single-scan model, and does **anatomy-guided auxiliary supervision** improve
representation quality — both measured without cognitive scores in the inference
pathway, and under subject-level partitioning?

The project does not claim longitudinal MRI modeling is new. LongFormer (WACV 2024)
precedes it. The claim is about what a *controlled* comparison shows.

## Arms

| Arm | Model | Role |
|---|---|---|
| B1 | 3D CNN / 3D ResNet, single scan | Transparent single-scan baseline |
| B2 | 3DSC-TF | Recent 3D convolution–Transformer comparison |
| B3 | LongFormer | Recent longitudinal comparison |
| **PA** | Shared visit encoder + temporal module | Treatment for ablation 1 |
| **PB** | PA + anatomy-guided auxiliary head | Treatment for ablation 2 |

## The two core ablations

Everything else is secondary. These two carry the project's claim.

**Ablation 1 — does longitudinal information help?** B1 versus PA. One variable changes:
the temporal module. Same encoder, same cohort, same partitions, same preprocessing,
same evaluation code. Restricted to subjects with two or more eligible visits, so the
architecture change is not confounded by a cohort change.

**Ablation 2 — does anatomy supervision help?** PA versus PB. One variable changes:
`λ · L_anatomy`. Note that PB's inference pathway is identical to PA's — the auxiliary
head is discarded — so any difference is attributable to the representation the
objective produced, not to extra information at test time.

## Metrics

Reported for every arm on the same held-out test subjects:

| Metric | Why it is here |
|---|---|
| ROC-AUC | Threshold-free ranking quality; the headline number |
| Balanced accuracy | Accuracy is uninformative under class imbalance |
| Macro-F1 | Weights the minority class equally |
| Sensitivity (impaired) | A missed impairment is the costly error |
| Specificity | Pairs with sensitivity; neither is readable alone |
| Confusion matrix | Shows where the errors actually fall |

Plain accuracy is not a headline metric. The predecessor's cohort was 76% negative, so
a model predicting one class reaches 76% while learning nothing.

## Uncertainty

A difference between arms is reported with its uncertainty or not at all. Repeated runs
across seeds where compute allows, bootstrap confidence intervals over test subjects
otherwise. A gap smaller than the spread across seeds is reported as "no detectable
difference," which is a real and publishable finding — the ablations are designed to be
informative when they come back negative.

## What would falsify the hypothesis

Stated in advance so the result cannot be reinterpreted after the fact:

- **Ablation 1 fails** if PA does not exceed B1 on ROC-AUC beyond run-to-run spread.
  Reported as: longitudinal information did not help *at this cohort size*, which is a
  claim about the experiment, not about longitudinal modeling in general.
- **Ablation 2 fails** if PB does not exceed PA by the same standard. Reported as: the
  auxiliary objective did not improve the representation at the tested `λ`.

## Guardrails

- The test partition is read **once per arm**, after that arm's training and tuning are
  complete. Every hyperparameter, including `λ` and the early-stopping point, is chosen
  on validation subjects.
- Partition assignment is drawn once and never redrawn.
- No arm gets preprocessing, augmentation, or a training budget another arm did not,
  except where the architecture makes it impossible — and that exception is reported.
- Negative and null results go in the report with the same prominence as positive ones.

## Scope under pressure

If access, compute, or cohort size fall short, cut in this order — baselines first,
core ablations last:

1. Drop B2 and B3 (the SOTA comparisons) and say so.
2. Drop the stretch goals entirely: pretraining transfer, calibration, interpretability,
   deployment demo, external validation.
3. Reduce input resolution and sequence length before reducing the number of arms.

The last thing to go is the controlled comparison, because a well-controlled negative
result is a finding and an uncontrolled positive one is not.
