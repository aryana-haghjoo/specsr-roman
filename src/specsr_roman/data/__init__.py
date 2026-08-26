"""Dataset, splits, augmentation, and the resampling primitives."""

from .augment import SpectrumAugmentor, calzetti_k, find_line_segments
from .datasets import RomanFixedGridDataset
from .photometry import apply_phot_noise, band_names, select_bands, standardization_stats
from .splits import filter_split_min_lines, get_or_make_group_split, get_or_make_split, hash_file
from .transforms import fluxconserve_resample, interp_ascending, normalize, smooth_to_grism

__all__ = [
    "RomanFixedGridDataset",
    "get_or_make_group_split", "get_or_make_split", "filter_split_min_lines",
    "hash_file",
    "SpectrumAugmentor", "calzetti_k", "find_line_segments",
    "normalize", "fluxconserve_resample", "smooth_to_grism", "interp_ascending",
    "select_bands", "apply_phot_noise", "band_names", "standardization_stats",
]
