---
status: accepted
date: 2026-11-22
---

# A pretrained 3D ViT-B/16 encoder, and the OASIS overlap it carries

[ADR-0005](0005-shared-encoder-over-deformation-fusion.md) fixed that one encoder is
shared across every visit and every arm, but left *which* encoder open. The placeholder
was a randomly initialized 3D ResNet-18.

With [ADR-0010](0010-subsampled-cohort-and-reduced-resolution.md) setting the cohort at
300 subjects — roughly 450 training sequences — a randomly initialized encoder is the
weakest part of the design. 33.2M parameters fitted from scratch on 450 sequences is a
capacity-to-data ratio no amount of augmentation repairs, and the obvious response
(shrink the model) buys generalization by giving up representational power.

Self-supervised pretraining answers both. A 3D encoder pretrained on tens of thousands
of brain MRIs arrives with features that 450 sequences could never induce, and the
labels it was fitted without are exactly the ones we are predicting.

## Decision

**The shared encoder is a 3D ViT-B/16 initialized from BrainIAC's self-supervised
weights.** It is selected once in the `encoder` config group at the root, so every arm
receives the same one.

| | Parameters | Pretraining |
|---|---|---|
| `vit_base_brainiac` **(default)** | 117.1M | BrainIAC, SimCLR over ~32k brain MRIs |
| `vit_base_random` | 117.1M | none — the initialization control |
| `resnet18` | 33.2M | none — the from-scratch convolutional reference |
| `vit_tiny_smoke` | 2.5M | none — smoke runs only, never a reported result |

Parameter counts are measured at the ADR-0010 input shape by
`tests/unit/test_encoder.py::TestMemoryFootprint`, not typed.

BrainIAC's published backbone is a ViT-B with 16³ patches over a 96³ volume. That is
already the ADR-0010 input shape, so adopting it costs **nothing at training time**:
the stored volume is the same 96³ it would otherwise have been, and ADR-0010's compute
argument survives intact. What changes is how that 96³ is produced — see Consequences.

## The overlap, stated plainly

**BrainIAC's pretraining corpus includes OASIS-3 and OASIS-2.** The encoder has
therefore seen scans belonging to subjects who will land in our test partition.

This is a real limitation and it is not worked around, because it cannot be:

- It was **self-supervised**. No CDR, MMSE, or diagnosis entered the pretraining
  objective, so it is not a label leak and
  [ADR-0002](0002-no-cognitive-scores-at-inference.md) is untouched.
- It is **constant across arms**. Both core ablations are within-cohort contrasts
  between arms sharing one encoder, so the overlap shifts every arm by the same unknown
  amount and cancels in the difference. The contrasts are what this project claims.
- It **does** inflate absolute numbers, which are therefore not comparable with
  literature that trained from scratch. The report says so wherever an absolute figure
  appears.

Avoiding it entirely was investigated and no option survives — see below.

## Considered options

**Stay with a randomly initialized ResNet-18.** No overlap, nothing to disclose, and
every absolute number directly comparable. Rejected as the default because it leaves
the largest available improvement on the table for a caveat that does not touch the
contrasts we actually claim. Kept as `resnet18`, and still run: it is the number a
reader expects.

**BrainSegFounder** (Swin UNETR, UK Biobank + BraTS/ATLAS). The one genuinely
OASIS-free option found. Rejected on licensing: the weights are distributed under the
UK Biobank Material Transfer Agreement and restricted to approved UK Biobank
researchers, which this team is not. It was also pretrained on paired T1+T2, and only
T1w is downloaded (ADR-0010).

**Brain2Vec** (masked autoencoder, 74,425 T1w scans from iSTAGING). Rejected twice
over: iSTAGING includes OASIS, so the overlap is not avoided, and the weights are
CC BY-NC-ND — the no-derivatives term is a poor fit for fine-tuning.

**MONAI's CT-pretrained Swin UNETR SSL weights.** No OASIS, licence fine. Rejected
because CT-to-T1 transfer is an open question rather than a free win, which is what the
original encoder docstring already said.

**Pretrain our own encoder on the training partition only.** Clean by construction and
the right answer with a different budget. 300 subjects is three orders of magnitude
short of what contrastive pretraining needs.

**A larger ResNet (34 or 50).** More capacity with none of the benefit; the problem was
never that 33M parameters were too few.

## Consequences

- **The encoder is chosen once, at the config root, for all arms.** Previously it lived
  inside each model config, where B1 and PA could have drifted apart and confounded
  ablation 1. `tests/unit/test_configs.py` now asserts the arms share an encoder and
  that B1 and PA differ in the temporal module alone.
- **A third comparison becomes available for one extra run.** `vit_base_brainiac`
  against `vit_base_random` on arm PA, identical in every other respect, measures what
  foundation-model pretraining is worth on a cohort this size. Run on PA only; running
  it on all five arms doubles the budget to answer a question once.
- **Preprocessing must target BrainIAC's expected input.** Skull-stripping, N4 bias
  correction, rigid registration to the BrainIAC template, then resize to 96³ — rather
  than resampling to 2 mm. This changes the work in
  [#15](https://github.com/JNR-10/oasis-alzheimers-dl/issues/15) and interacts with the
  registration decision in
  [#16](https://github.com/JNR-10/oasis-alzheimers-dl/issues/16): the within-subject
  alignment that ADR-0010's successor must settle still applies, with the template fixed
  by this choice rather than chosen freely.
- **Weights never enter git.** The checkpoint is read from `BRAINIAC_CHECKPOINT`, the
  same way imaging is read from `OASIS_ROOT`. The encoder will not download it: a
  recorded run must not depend on a URL that can change under it.
- **Licence is research-only.** Non-commercial academic use, which this project is.
  Recorded in `src/oasis_dl/models/encoder.py` beside the checkpoint entry so it travels
  with the weights.
- **The overlap is announced at run time**, not only here. Loading a checkpoint whose
  corpus overlaps the evaluation cohort emits a warning naming the datasets, so a run
  log carries the caveat even if nobody read this file.
- **A ViT fixes its input shape at construction.** Unlike the ResNet it replaces, it
  cannot accept another volume size. The config interpolates the shape from the data
  config so the two cannot drift, and the encoder rejects a mismatch rather than
  resizing input a recorded run depends on.
- **CPU training is no longer plausible for the real arms.** `vit_base_16` at 96³ is
  over a second per forward pass on a laptop CPU. `vit_tiny_smoke` exists so that the
  training loop can still be proven end to end on any machine; production runs move to
  whoever has the GPU.

## Sources

- BrainIAC — [Nature Neuroscience (2026)](https://www.nature.com/articles/s41593-026-02202-6),
  [preprint](https://www.medrxiv.org/content/10.1101/2024.12.02.24317992v1.full),
  [code](https://github.com/AIM-KannLab/BrainIAC)
- Brain2Vec — [arXiv:2610.28411](https://arxiv.org/abs/2610.28411), [code](https://github.com/CBICA/Brain2Vec)
- BrainSegFounder — [weights and licence](https://huggingface.co/smilelab/BrainSegFounder)
