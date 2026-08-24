"""Package-owned real-study executors built on explicit external evidence authority."""

from .kumar2024 import Kumar2024StudyConfig, fingerprint_raw_dataset
from .kumar2024_execution import run_kumar2024_study, run_kumar2024_subject

__all__ = [
    "Kumar2024StudyConfig",
    "fingerprint_raw_dataset",
    "run_kumar2024_study",
    "run_kumar2024_subject",
]
