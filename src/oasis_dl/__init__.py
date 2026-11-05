"""Longitudinal deep learning for Alzheimer's disease detection from structural MRI.

The vocabulary used throughout this package is fixed in ``GLOSSARY.md`` at the repo
root. The decisions it encodes are recorded in ``docs/adr/``; where a module comment
says "see ADR-NNNN", that record holds the reasoning and this code holds only the
consequence.
"""

from oasis_dl.types import Label, Partition, SessionId, SubjectId, Visit, VisitSequence

__version__ = "0.1.0"

__all__ = [
    "Label",
    "Partition",
    "SessionId",
    "SubjectId",
    "Visit",
    "VisitSequence",
    "__version__",
]
