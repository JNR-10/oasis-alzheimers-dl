"""Build the cohort manifest and draw the partitions. Run once.

``make cohort``

The partitions this writes are drawn once and never redrawn (ADR-0001). Running this
a second time against an existing manifest path fails rather than overwriting, because
the old manifest may already have produced results.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from oasis_dl.cli import CONFIG_PATH
from oasis_dl.data.cohort import (
    InclusionRules,
    build_sequences,
    build_visits,
    sample_subjects,
    sampling_pool,
    summarize_cohort,
)
from oasis_dl.data.manifest import write_manifest
from oasis_dl.data.oasis3 import Oasis3CohortSource, Oasis3SourceConfig
from oasis_dl.data.splits import partition_subjects
from oasis_dl.types import Partition, SubjectId

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Apply inclusion rules, build sequences, draw partitions, write the manifest.

    Order (ADR-0010): sessions -> eligible subjects -> *sample* -> preprocess ->
    visits -> partition. The sample is drawn before preprocessing, so the manifest
    freezes the sampled subject list and the eligibility audit; scan paths may point
    at raw or future-preprocessed locations until #15 runs.
    """
    oasis_root = os.environ.get("OASIS_ROOT")
    if not oasis_root:
        raise RuntimeError(
            "OASIS_ROOT is not set. Set it in your environment (and .env, which is "
            "gitignored) to the OASIS-3 data root containing OASIS3_data_files_full_set."
        )

    manifest_path = Path(cfg.data.manifest_path)
    rules = InclusionRules(
        max_assessment_gap_days=int(cfg.data.inclusion.max_assessment_gap_days),
        min_visits_for_longitudinal=int(cfg.data.inclusion.min_visits_for_longitudinal),
        require_preprocessing_success=bool(cfg.data.inclusion.require_preprocessing_success),
    )
    max_subjects = cfg.data.sampling.max_subjects
    sample_seed = int(cfg.data.sampling.sample_seed)
    split_seed = int(cfg.data.split_seed)
    split_ratios = tuple(cfg.data.split_ratios)
    max_sequence_length = int(cfg.data.max_sequence_length)

    log.info("building cohort from OASIS_ROOT=%s", oasis_root)
    log.info("inclusion rules: %s", rules)
    log.info("writing manifest to %s", manifest_path)
    log.info("sample seed: %s, max_subjects: %s", sample_seed, max_subjects)
    log.info("partition seed: %s, ratios: %s", split_seed, split_ratios)

    source = Oasis3CohortSource(Oasis3SourceConfig(root=Path(oasis_root)))
    visits, audit = build_visits(source, rules)

    # ADR-0010 samples from subjects with two or more labeled visits only.
    subject_labels = sampling_pool(visits, min_labeled_visits=rules.min_visits_for_longitudinal)
    if not subject_labels:
        raise RuntimeError(
            f"no subject has {rules.min_visits_for_longitudinal} or more labeled "
            f"visits; there is no longitudinal cohort to sample"
        )

    sampled = sample_subjects(subject_labels, max_subjects=max_subjects, seed=sample_seed)
    sampled_set = set(sampled)
    log.info(
        "sampled %d of %d subjects with >= %d labeled visits (stratified by label)",
        len(sampled),
        len(subject_labels),
        rules.min_visits_for_longitudinal,
    )

    # Keep only sampled subjects' visits; build sequences; partition the sampled pool.
    sampled_visits = [v for v in visits if v.subject_id in sampled_set]
    sequences = build_sequences(sampled_visits, max_length=max_sequence_length)
    sampled_subject_labels = {subject: subject_labels[subject] for subject in sampled}
    assignment = partition_subjects(sampled_subject_labels, ratios=split_ratios, seed=split_seed)

    manifest_frame = _build_manifest_frame(
        sampled=sampled_set,
        assignment=assignment,
        audit=audit,
    )
    write_manifest(manifest_frame, manifest_path)
    log.info("wrote manifest: %d rows to %s", len(manifest_frame), manifest_path)

    summary = summarize_cohort(sequences, assignment=assignment)
    summary_path = manifest_path.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    log.info("wrote cohort summary: %d rows to %s", len(summary), summary_path)


def _build_manifest_frame(
    sampled: set[SubjectId],
    assignment: dict[SubjectId, Partition],
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """One row per input session: survivors carry partition + is_target_visit;
    sampled-out and excluded sessions carry their exclusion_reason."""
    rows: list[dict[str, object]] = []
    for record in audit.to_dict(orient="records"):
        subject_id = SubjectId(str(record["subject_id"]))
        reason = str(record["exclusion_reason"])
        raw_label = record["label"]
        label_is_na = bool(pd.isna(raw_label))
        if reason:
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "session_id": record["session_id"],
                    "partition": "",
                    "days_from_entry": record["days_from_entry"],
                    "scan_path": record["scan_path"],
                    "label": "" if label_is_na else int(float(raw_label)),
                    "is_target_visit": False,
                    "exclusion_reason": reason,
                }
            )
            continue
        in_sample = subject_id in sampled
        if not in_sample:
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "session_id": record["session_id"],
                    "partition": "",
                    "days_from_entry": record["days_from_entry"],
                    "scan_path": record["scan_path"],
                    "label": "" if label_is_na else int(float(raw_label)),
                    "is_target_visit": False,
                    "exclusion_reason": "sampled_out",
                }
            )
            continue
        rows.append(
            {
                "subject_id": record["subject_id"],
                "session_id": record["session_id"],
                "partition": assignment[subject_id].value,
                "days_from_entry": record["days_from_entry"],
                "scan_path": record["scan_path"],
                "label": "" if label_is_na else int(float(raw_label)),
                "is_target_visit": not label_is_na,
                "exclusion_reason": "",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
