"""Evaluation: frozen prediction caches, metrics, figures, and audits."""

from .cache import CacheConfig, build_prediction_cache, load_prediction_cache
from .metrics import (
                      HLSS_LINE_FLUX_LIMIT,
                      MIN_BEST_LINE_SNR,
                      RECOVERABILITY_BINS,
                      detected_line_mask,
                      hlss_cosmology_sample,
                      line_amplitude_recovery,
                      line_snr_from_spectrum,
                      per_line_amplitude_recovery,
                      redshift_summary,
                      true_line_flux_cgs,
)

__all__ = [
    "CacheConfig", "build_prediction_cache", "load_prediction_cache",
    "line_amplitude_recovery", "per_line_amplitude_recovery",
    "redshift_summary", "line_snr_from_spectrum",
    "RECOVERABILITY_BINS", "true_line_flux_cgs", "hlss_cosmology_sample",
    "HLSS_LINE_FLUX_LIMIT", "MIN_BEST_LINE_SNR", "detected_line_mask",

]
