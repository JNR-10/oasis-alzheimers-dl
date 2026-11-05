# Literature and SOTA Survey

This survey supports the Fall 2026 project proposal. It focuses on recent work relevant to 3D structural MRI, longitudinal Alzheimer's disease classification, volumetric CNN-Transformer models, and 3D medical-image representation learning. The project is not claiming that any listed paper is directly reproducible on OASIS-3 without additional preprocessing or label work.

## Verification note

Paper metadata and public code/model links were checked on 2026-09-06 against the linked paper pages and repositories. A public repository is not the same as a drop-in baseline: several projects use older dependencies, different datasets, segmentation rather than classification, or a different split protocol. Those limitations are recorded below.

## 1. LongFormer: Longitudinal Transformer for Alzheimer's Disease Classification with Structural MRIs

- **Year / venue:** 2024, IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3575-3584.
- **Paper:** [arXiv:2302.00901](https://arxiv.org/abs/2302.00901) and [WACV open-access record](https://openaccess.thecvf.com/content/WACV2024/html/Chen_Longformer_Longitudinal_Transformer_for_Alzheimers_Disease_Classification_With_Structural_MRIs_WACV_2024_paper.html).
- **Problem addressed:** Alzheimer's versus normal-control classification using longitudinal structural MRI.
- **Core method:** A hybrid 3D CNN and Transformer framework that uses a current scan with a prior-scan optical-flow or deformation-field representation. The paper describes a query-based Transformer with deformable cross-attention to fuse spatiotemporal features.
- **Datasets:** ADNI, OASIS-2, and AIBL. The paper evaluates on longitudinal data and reports OASIS-2 rather than OASIS-3.
- **Code / model availability:** The paper links an author implementation at [Qybc/LongFormer](https://github.com/Qybc/LongFormer). The repository should be checked for its current environment, data-preparation assumptions, and checkpoint availability before reuse.
- **Why it matters:** It is the closest recent precedent for the proposed longitudinal 3D MRI direction.
- **How we may use it:** Primary longitudinal comparison and architecture inspiration, subject to reimplementation or adaptation for the course cohort.
- **Limitations / project gap:** It already establishes that longitudinal MRI modeling is not a new idea. Its pair-and-flow formulation also differs from the proposed shared encoder plus temporal attention design, so any comparison must keep the subject-level split and preprocessing protocol controlled.

## 2. 3DSC-TF: Classification of Alzheimer's Disease by Jointing 3D Depthwise Separable Convolutional Neural Network and Transformer

- **Year / venue:** 2025, *Expert Systems with Applications*, vol. 286, article 127720, DOI [10.1016/j.eswa.2025.127720](https://doi.org/10.1016/j.eswa.2025.127720).
- **Problem addressed:** Alzheimer's disease classification from structural MRI.
- **Core method:** A 3D depthwise-separable convolutional network combined with Transformer-based feature modeling, with an accompanying interpretability/visual-patch component.
- **Datasets:** The paper's implementation and experiments need to be read alongside its data instructions; the public repository exposes a `Dataset` directory and train/test scripts.
- **Code / model availability:** Public implementation at [NWPU-903PR/3DSC-TF](https://github.com/NWPU-903PR/3DSC-TF). The repository documents Python 3.5.2, PyTorch 1.10.1, and GPU-oriented experiments; no claim is made here that its environment is immediately compatible with the current project.
- **Why it matters:** It is a recent AD-specific 3D CNN-Transformer candidate and satisfies the course requirement to evaluate a recent open-source model relevant to the task.
- **How we may use it:** Second SOTA comparison or architecture reference for convolution-Transformer fusion.
- **Limitations / project gap:** The public implementation is small, old in its dependency assumptions, and not designed around longitudinal subject sequences. Reproducing it may require a carefully documented compatibility layer or a faithful reimplementation.

## 3. Joint Transformer Architecture in Brain 3D MRI Classification: Its Application in Alzheimer's Disease Classification

- **Year / venue:** 2024, *Scientific Reports*, 14, 8996, DOI [10.1038/s41598-024-59578-3](https://doi.org/10.1038/s41598-024-59578-3).
- **Problem addressed:** Binary and multiclass Alzheimer's disease classification using ADNI T1-weighted MRI.
- **Core method:** ViT-derived features from multiple anatomical planes are arranged as sequences and processed by a time-series Transformer. The use of “time series” refers to feature sequences across slices/planes in this work, not necessarily repeated visits from the same subject.
- **Datasets:** ADNI collections described as Complete 1Yr 1.5T and Complete 3Yr 3T.
- **Code / model availability:** An author-linked public implementation is available at [tami64/Joint-Transformer-in-AD-MRI-Classification](https://github.com/tami64/Joint-Transformer-in-AD-MRI-Classification). It documents CAT12 preprocessing, ViT feature extraction, and 10-fold classification scripts.
- **Why it matters:** It shows that Transformer-based 3D MRI classification is already crowded and that very high reported accuracy can depend strongly on preprocessing and split design.
- **How we may use it:** Architecture inspiration and a leakage/split-protocol comparison, not an automatic claim of superiority.
- **Limitations / project gap:** The paper describes random splits and slice/feature sequences, which are not equivalent to subject-level longitudinal evaluation. Our project must avoid allowing visits or correlated derived samples from one subject to cross partitions.

## 4. VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis

- **Year / venue:** 2024, IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22873-22882, DOI [10.1109/CVPR52733.2024.02158](https://doi.org/10.1109/CVPR52733.2024.02158).
- **Problem addressed:** Self-supervised representation learning for 3D medical images when high-level annotations are limited.
- **Core method:** Volume Contrast pretraining uses contextual position priors. Base crops represent different regions, and randomly sampled sub-volumes are trained to identify their contextual region.
- **Datasets:** The public implementation describes 10k and 160k pretraining variants, primarily involving CT data and multiple downstream 3D medical tasks.
- **Code / model availability:** Public repository and pretrained-weight instructions at [Luffy03/VoCo](https://github.com/Luffy03/VoCo). The README notes that pretraining code is available and that downstream fine-tuning code is more limited.
- **Why it matters:** It is a recent, concrete option for testing whether 3D pretraining can help when the labeled OASIS cohort is small.
- **How we may use it:** Pretraining comparison or representation-learning inspiration, only after checking modality transfer and licensing.
- **Limitations / project gap:** VoCo is not an Alzheimer's classifier and its released pretraining data/checkpoints are not OASIS T1 MRI. Any transfer result would be a separate experiment, not a direct SOTA comparison.

## 5. An OpenMind for 3D Medical Vision Self-Supervised Learning

- **Year / status:** 2024 arXiv preprint, arXiv [2412.17041](https://arxiv.org/abs/2412.17041).
- **Problem addressed:** Lack of standardization in 3D medical-image self-supervised learning caused by different datasets, architectures, and downstream protocols.
- **Core method:** A benchmark and framework for comparing 3D SSL methods under common architectures; the paper describes a large public pretraining resource containing 114k 3D brain MRI volumes.
- **Datasets:** OpenMind pretraining data and downstream 3D medical-imaging tasks.
- **Code / model availability:** Public framework at [MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl). The repository includes ResEnc-L and Primus-M architectures, multiple SSL methods, and links to OpenMind data/checkpoint resources.
- **Why it matters:** The brain-MRI pretraining domain is closer to OASIS than CT-only resources, and the framework highlights the need for fair, common evaluation rather than isolated accuracy claims.
- **How we may use it:** Optional pretrained-encoder comparison or a principled source of 3D representation-learning baselines.
- **Limitations / project gap:** It is a framework and pretraining resource, not a ready-made longitudinal Alzheimer's classifier. Storage, compute, licensing, and checkpoint transfer must be checked before adding it to the minimum scope.

## 6. How Well Do Supervised 3D Models Transfer to Medical Imaging Tasks? (SuPreM)

- **Year / venue:** 2024, International Conference on Learning Representations (ICLR), oral presentation.
- **Paper:** [ICLR paper record](https://proceedings.iclr.cc/paper_files/paper/2024/file/47360925c45a166c96f652589265dee9-Paper-Conference.pdf).
- **Problem addressed:** Whether large-scale supervised pretraining can transfer effectively across 3D medical-imaging tasks.
- **Core method:** Supervised pretrained 3D backbones using large annotated volumetric datasets, with transfer experiments across medical segmentation and related tasks.
- **Datasets:** The released SuPreM models are trained primarily on large CT/abdominal datasets such as AbdomenAtlas, not on OASIS brain MRI.
- **Code / model availability:** Public code, training/fine-tuning examples, and model-weight links are maintained at [MrGiovanni/SuPreM](https://github.com/MrGiovanni/SuPreM).
- **Why it matters:** It provides a useful contrast to self-supervised pretraining and makes the domain-shift question explicit.
- **How we may use it:** Optional pretrained-versus-random initialization experiment if a compatible 3D encoder and licensing allow it.
- **Limitations / project gap:** CT-to-T1 MRI transfer may be weak or require substantial adaptation. It should remain a stretch comparison rather than a required baseline.

## 7. Med3DInsight: Enhancing 3D Medical Image Understanding with Pretraining Aided by 2D Multimodal Large Language Models

- **Year / status:** 2025 arXiv preprint, arXiv [2509.09064](https://arxiv.org/abs/2509.09064); the arXiv record notes acceptance by the *IEEE Journal of Biomedical and Health Informatics*.
- **Problem addressed:** Improving semantic representation learning for 3D medical volumes by using signals from 2D multimodal language models.
- **Core method:** A 3D image encoder is connected to a 2D multimodal model through a plane-slice-aware Transformer and partial optimal-transport alignment.
- **Datasets:** Public CT and MRI datasets across classification and segmentation downstream tasks.
- **Code / model availability:** A public repository is available at [Qybc/Med3DInsight](https://github.com/Qybc/Med3DInsight). The paper describes source code, generated data, and pretrained models as release resources; their completeness and current compatibility should be confirmed before use.
- **Why it matters:** It represents a recent direction in 3D medical representation learning and could inform a future pretrained-encoder or semantic-auxiliary experiment.
- **How we may use it:** Stretch architecture/pretraining comparison, not part of the minimum semester implementation.
- **Limitations / project gap:** It is not specific to Alzheimer's or longitudinal visits and may add a substantial engineering and compute burden. The proposed project should not depend on it for feasibility.

## Synthesis and selection rationale

The literature makes the novelty boundary clear. A generic “MRI to Alzheimer's classifier” is already a crowded problem, and LongFormer shows that longitudinal structural MRI modeling predates this project. The project therefore uses a conventional 3D CNN/ResNet to establish a transparent single-scan baseline, evaluates LongFormer and 3DSC-TF as recent open-source candidates, and treats VoCo/OpenMind/SuPreM/Med3DInsight as optional representation-learning resources rather than guaranteed plug-ins.

The proposed contribution is a controlled extension of the prior ML work: learned volumetric representations, explicit longitudinal modeling, anatomy-guided auxiliary supervision, no MMSE inference input, subject-level splitting, and ablations that isolate the value of each change. The result will be meaningful only if those controls are maintained and any improvement is consistent across held-out subjects.
