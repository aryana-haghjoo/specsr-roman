"""Evaluation: frozen prediction caches, metrics, figures, and audits."""

from .cache import CacheConfig, build_prediction_cache, load_prediction_cache
from .metrics import (
                      RECOVERABILITY_BINS,
                      line_amplitude_recovery,
                      line_snr_from_spectrum,
                      per_line_amplitude_recovery,
                      redshift_summary,
)

__all__ = [
    "CacheConfig", "build_prediction_cache", "load_prediction_cache",
    "line_amplitude_recovery", "per_line_amplitude_recovery",
    "redshift_summary", "line_snr_from_spectrum",
    "RECOVERABILITY_BINS",
]
