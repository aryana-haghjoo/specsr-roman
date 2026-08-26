"""Rest-frame spectral features used across the pipeline.

Three line sets appear here, and they are not interchangeable:

``LINE_LIST_REST_AA``
    The full 98-feature list that SR2's line-token transformer attends over.
    Breadth matters more than in-band coverage: a token whose line falls
    outside the grism band is gated off by ``in_range`` in the line branch,
    so an over-complete list costs a little compute and buys robustness
    across the redshift range.

``SR1_LINES_AA``
    The ten redshift-carrying lines SR1 supervises directly (line-flux L1 and
    the anti-hallucination penalty), and the lines the dataset computes
    per-row integrated S/N labels for. **Order is load-bearing** --- the
    ``line_snr`` column of :class:`specsr_roman.data.RomanFixedGridDataset` follows
    this tuple, and the losses index into it positionally.

``STRONG_LINES_AA``
    The four lines that decide whether a redshift is *identifiable* at all.
    Two or more in band means the redshift is line-*pair* constrained; one is
    alias-degenerate (Ha/[OIII]/[OII] all look alike on their own); zero
    (z < 0.52 for the Roman band) means unconstrained. Used by the
    ``min_strong_lines`` split filter.
"""

from __future__ import annotations

import numpy as np

# Rest-frame features (Angstrom), mixing emission lines with the continuum
# breaks and absorption features that carry redshift information for quiescent
# galaxies. Deduplicated: identical wavelengths would waste attention capacity.
LINE_LIST_REST_AA: list[tuple[str, float]] = [
    # --- Continuum breaks / edges ---
    ("Lyman_limit_912", 912.0),
    ("LyA_1216", 1215.670),
    ("Balmer_break_3646", 3646.0),
    ("D4000_break_4000", 4000.0),

    # --- UV emission lines (AGN + star-forming) ---
    ("NV_1240", 1240.810),
    ("SiII_1260", 1260.422),
    ("OI_1302", 1302.168),
    ("SiII_1304", 1304.370),
    ("CII_1335", 1335.708),
    ("SiIV_1394", 1393.755),
    ("SiIV_1403", 1402.770),
    ("OIV]_1402", 1402.06),
    ("NIV]_1486", 1486.50),
    ("SiII_1527", 1526.707),
    ("CIV_1548", 1548.204),
    ("CIV_1551", 1550.781),
    ("HeII_1640", 1640.420),
    ("OIII]_1661", 1660.809),
    ("OIII]_1666", 1666.150),
    ("AlII_1671", 1670.788),
    ("SiII_1808", 1808.013),
    ("AlIII_1855", 1854.716),
    ("AlIII_1863", 1862.789),
    ("SiIII]_1892", 1892.03),
    ("CIII]_1907", 1906.680),
    ("CIII]_1909", 1908.734),
    ("FeII_UV_2344", 2344.214),
    ("FeII_UV_2374", 2374.461),
    ("FeII_UV_2382", 2382.765),
    ("MnII_2577", 2576.877),
    ("MnII_2594", 2594.499),
    ("MnII_2606", 2606.462),
    ("FeII_UV_2586", 2586.650),
    ("FeII_UV_2600", 2600.173),
    ("MgII_2796", 2796.352),
    ("MgII_2803", 2803.531),
    ("MgI_2853", 2852.964),

    # --- Optical strong nebular emission (galaxies) ---
    ("[OII]_3726", 3726.032),
    ("[OII]_3729", 3728.815),
    ("Htheta_3798", 3797.900),
    ("Heta_3835", 3835.386),
    ("[NeIII]_3869", 3868.760),
    ("H8_3889", 3889.064),
    ("CaK_3934", 3933.663),
    ("[NeIII]_3968", 3967.470),
    ("CaH_3969", 3968.468),
    ("Hepsilon_3970", 3970.075),
    ("Hdelta_4102", 4101.734),
    ("Gband_4304", 4304.0),
    ("Hgamma_4341", 4340.472),
    ("[OIII]_4363", 4363.210),
    ("HeI_4471", 4471.479),
    ("FeII_opt_blend_4570", 4570.0),
    ("HeII_4686", 4685.710),
    ("Hbeta_4861", 4861.333),
    ("[OIII]_4959", 4958.911),
    ("[OIII]_5007", 5006.843),
    ("Mg_b_5167", 5167.321),
    ("Mg_b_5173", 5172.684),
    ("Mg_b_5184", 5183.604),
    ("[NI]_5198", 5197.902),
    ("[NI]_5200", 5200.257),
    ("FeII_opt_blend_5350", 5350.0),
    ("DIB_5780", 5780.5),
    ("HeI_5876", 5875.624),
    ("NaD_5890", 5889.951),
    ("NaD_5896", 5895.924),
    ("TiO_6159", 6159.0),
    ("DIB_6284", 6283.8),
    ("[OI]_6300", 6300.304),
    ("[OI]_6364", 6363.776),
    ("[NII]_6548", 6548.050),
    ("Halpha_6563", 6562.800),
    ("[NII]_6583", 6583.450),
    ("TiO_6651", 6651.0),
    ("HeI_6678", 6678.151),
    ("[SII]_6716", 6716.440),
    ("[SII]_6731", 6730.820),
    ("[ArIII]_7136", 7135.790),
    ("[OII]_7320", 7319.990),
    ("[OII]_7330", 7330.190),

    # --- NIR ---
    ("CaII_triplet_8498", 8498.020),
    ("CaII_triplet_8542", 8542.090),
    ("CaII_triplet_8662", 8662.140),
    ("[SIII]_9069", 9068.600),
    ("Pa10_9015", 9014.910),
    ("Pa9_9229", 9229.014),
    ("[SIII]_9531", 9530.600),
    ("Pa8_9546", 9545.969),
    ("Pa7_10049", 10049.37),
    ("HeI_10830", 10830.340),
    ("Pa_gamma_10941", 10941.09),
    ("[FeII]_12567", 12566.770),
    ("Pa_beta_12818", 12818.08),
    ("[FeII]_16435", 16435.0),
    ("Br11_16811", 16811.0),
    ("Br10_17363", 17363.0),
    ("Br_gamma_21661", 21661.0),
]

LINE_NAMES: list[str] = [n for (n, _) in LINE_LIST_REST_AA]
LINE_WAVELENGTHS_AA: list[float] = [w for (_, w) in LINE_LIST_REST_AA]

# The four lines that decide whether a redshift is identifiable at all.
STRONG_LINES_AA: tuple[float, ...] = (6563.0, 5007.0, 4861.0, 3727.0)

# Lines SR1 supervises. The strong-line set extended so every redshift in
# 0 < z < 2 has at least one member in the Roman band (bright z < 0.5 galaxies
# land on the NIR set). ORDER IS LOAD-BEARING: the per-row line-S/N labels in
# RomanFixedGridDataset follow this tuple and the losses index it positionally.
SR1_LINES_AA: tuple[float, ...] = (
    3727.0,   # [OII]
    4861.0,   # Hb
    5007.0,   # [OIII]
    6563.0,   # Ha
    6725.0,   # [SII] doublet (blended at grism R)
    9069.0,   # [SIII]
    9532.0,   # [SIII]
    10830.0,  # HeI
    12818.0,  # Pab
    18751.0,  # Paa
)

# Lines worth labelling on a figure: the ones a reader looks for.
LABEL_LINES_AA: tuple[tuple[str, float], ...] = (
    (r"[OII]", 3727.0),
    (r"H$\beta$", 4861.0),
    (r"[OIII]", 5007.0),
    (r"H$\alpha$+[NII]", 6563.0),
    (r"[SII]", 6725.0),
    (r"[SIII]", 9531.0),
    (r"HeI", 10830.0),
)


def angstrom_to_micron(x) -> np.ndarray:
    """Angstrom -> micron, as float32 (the dtype every model buffer uses)."""
    return np.asarray(x, dtype=np.float32) * 1e-4


def count_strong_lines(z, lam_min_aa: float, lam_max_aa: float) -> np.ndarray:
    """Number of :data:`STRONG_LINES_AA` inside ``[lam_min_aa, lam_max_aa]`` at ``z``."""
    z = np.asarray(z, dtype=np.float64)
    c = np.zeros(z.shape, dtype=np.int64)
    for w in STRONG_LINES_AA:
        lam = (1.0 + z) * w
        c += ((lam > lam_min_aa) & (lam < lam_max_aa)).astype(np.int64)
    return c
