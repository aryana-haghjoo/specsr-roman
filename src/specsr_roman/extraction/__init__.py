"""Building training data from Roman simulation products.

OpenUniverse2024 provides direct images, truth catalogues and Diffsky SEDs but
no grism images, so this package disperses the scene with grizli and extracts
it back --- making input and target self-consistent by construction.

Requires the extraction extra (``pip install specsr-roman[extract]``): grizli,
photutils, h5py and pyarrow are imported lazily so the rest of the package
works without them.
"""

from .batch import ExtractionConfig, merge, run_batch, run_worker
from .catalog import ab_h158, detect_and_relabel, load_truth_index
from .extract import ExtractionFailure, extract_target, to_fixed_grids
from .frames import prepare_frames
from .seds import SEDLibrary
from .simulate import add_grism_noise, disperse_scene

__all__ = [
    "ExtractionConfig", "run_batch", "run_worker", "merge",
    "prepare_frames", "load_truth_index", "detect_and_relabel", "ab_h158",
    "SEDLibrary", "disperse_scene", "add_grism_noise",
    "extract_target", "to_fixed_grids", "ExtractionFailure",
]
