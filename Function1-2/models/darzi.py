import math
from dataclasses import dataclass
from typing import Optional
from models.sof import encode_sof


# ============================================================
# Darzi et al. (2019) - PGA coefficients
# ============================================================

DARZI_PGA_COEFFICIENTS = {
    "c1": 0.549,
    "m1": 0.702,
    "m2": -0.023,
    "h": 6.629,
    "r1": -1.356,
    "sII": -0.007,
    "sIII": 0.054,
    "fRV": -0.023,
    "fSS": -0.032,

    # Uncertainty components
    "phi": 0.226,
    "tau1": 0.207,
    "tau2": 0.143,
    "sigma": 0.267,
}


@dataclass
class DarziResult:
    """Output of the Darzi PGA calculation."""

    log10_pga: float
    pga_cm_s2: float
    pga_g: float

    magnitude_term: float
    distance_term: float
    site_term: float
    sof_term: float

    sigma: float


# ============================================================
# Hypocentral Distance
# ============================================================

def calculate_hypocentral_distance(
    epicentral_distance_km: float,
    depth_km: float
) -> float:
    """
    Calculate hypocentral distance.

    R_hyp = sqrt(R_epi^2 + Depth^2)
    """

    # --------------------------------------------------------
    # Validate epicentral distance
    # --------------------------------------------------------

    if not isinstance(epicentral_distance_km, (int, float)):
        raise TypeError(
            "Epicentral distance must be a numeric value in kilometers. "
            f"Received {epicentral_distance_km!r} "
            f"of type {type(epicentral_distance_km).__name__}."
        )

    if epicentral_distance_km < 0:
        raise ValueError(
            "Epicentral distance cannot be negative. "
            f"Received {epicentral_distance_km} km."
        )

    # --------------------------------------------------------
    # Validate depth
    # --------------------------------------------------------

    if not isinstance(depth_km, (int, float)):
        raise TypeError(
            "Earthquake depth must be a numeric value in kilometers. "
            f"Received {depth_km!r} "
            f"of type {type(depth_km).__name__}."
        )

    if depth_km < 0:
        raise ValueError(
            "Earthquake depth cannot be negative. "
            f"Received {depth_km} km."
        )

    return math.sqrt(
        epicentral_distance_km**2 +
        depth_km**2
    )


# ============================================================
# Darzi PGA Calculation
# ============================================================

def calculate_darzi_pga(
    magnitude: float,
    hypocentral_distance_km: float,
    site_class: int = 1,
    mechanism: Optional[str] = None
) -> DarziResult:
    """
    Calculate PGA using the Darzi et al. (2019) GMPE.

    The implemented distance measure is hypocentral distance
    (R_hyp).

    PGA is returned in:
        cm/s^2
        g
    """

    # ========================================================
    # Input Validation
    # ========================================================

    # --------------------------------------------------------
    # 1. Validate magnitude type
    # --------------------------------------------------------

    if magnitude is None:
        raise ValueError(
            "Magnitude is required and cannot be None."
        )

    if not isinstance(magnitude, (int, float)):
        raise TypeError(
            "Magnitude must be a numeric value. "
            f"Received {magnitude!r} "
            f"of type {type(magnitude).__name__}."
        )

    # --------------------------------------------------------
    # 2. Validate magnitude value
    # --------------------------------------------------------

    if magnitude <= 0:
        raise ValueError(
            "Magnitude must be greater than zero. "
            f"Received Mw = {magnitude}."
        )

    # --------------------------------------------------------
    # 3. Validate hypocentral distance type
    # --------------------------------------------------------

    if hypocentral_distance_km is None:
        raise ValueError(
            "Hypocentral distance is required and cannot be None."
        )

    if not isinstance(hypocentral_distance_km, (int, float)):
        raise TypeError(
            "Hypocentral distance must be a numeric value in kilometers. "
            f"Received {hypocentral_distance_km!r} "
            f"of type {type(hypocentral_distance_km).__name__}."
        )

    # --------------------------------------------------------
    # 4. Validate hypocentral distance value
    # --------------------------------------------------------

    if hypocentral_distance_km < 0:
        raise ValueError(
            "Hypocentral distance cannot be negative. "
            f"Received {hypocentral_distance_km} km."
        )

    # --------------------------------------------------------
    # 5. Validate site class type
    # --------------------------------------------------------

    if not isinstance(site_class, int):
        raise TypeError(
            "Site class must be an integer: 1, 2, or 3. "
            f"Received {site_class!r} "
            f"of type {type(site_class).__name__}."
        )

    # --------------------------------------------------------
    # 6. Validate site class value
    # --------------------------------------------------------

    if site_class not in (1, 2, 3):
        raise ValueError(
            "Invalid site class. "
            "Site class must be 1 (Rock), 2, or 3. "
            f"Received site_class = {site_class}."
        )

    # --------------------------------------------------------
    # 7. Validate mechanism
    # --------------------------------------------------------

    valid_mechanisms = {
        "reverse",
        "strike-slip",
        "normal"
    }

    if mechanism not in valid_mechanisms:
        raise ValueError(
            "Invalid faulting mechanism. "
            "Mechanism must be one of: "
            "'reverse', 'strike-slip', or 'normal'. "
            f"Received {mechanism!r}."
        )

    # ========================================================
    # Coefficients
    # ========================================================

    c1 = DARZI_PGA_COEFFICIENTS["c1"]
    m1 = DARZI_PGA_COEFFICIENTS["m1"]
    m2 = DARZI_PGA_COEFFICIENTS["m2"]

    h = DARZI_PGA_COEFFICIENTS["h"]
    r1 = DARZI_PGA_COEFFICIENTS["r1"]

    sII = DARZI_PGA_COEFFICIENTS["sII"]
    sIII = DARZI_PGA_COEFFICIENTS["sIII"]

    fRV = DARZI_PGA_COEFFICIENTS["fRV"]
    fSS = DARZI_PGA_COEFFICIENTS["fSS"]

    sigma = DARZI_PGA_COEFFICIENTS["sigma"]

    # ========================================================
    # 1. Magnitude term
    # ========================================================

    magnitude_term = (
        c1
        + m1 * magnitude
        + m2 * magnitude**2
    )

    # ========================================================
    # 2. Distance term
    # ========================================================

    effective_distance = math.sqrt(
        hypocentral_distance_km**2 +
        h**2
    )

    distance_term = (
        r1 *
        math.log10(effective_distance)
    )

    # ========================================================
    # 3. Site term
    # ========================================================

    if site_class == 1:
        II = 0
        III = 0

    elif site_class == 2:
        II = 1
        III = 0

    else:
        II = 0
        III = 1

    site_term = (
        sII * II +
        sIII * III
    )

    # ========================================================
    # 4. Style of Faulting term
    # ========================================================

    rv, ss = encode_sof(mechanism)

    sof_term = (
        fRV * rv +
        fSS * ss
    )

    # ========================================================
    # 5. Final logarithmic PGA
    # ========================================================

    log10_pga = (
        magnitude_term
        + distance_term
        + site_term
        + sof_term
    )

    # ========================================================
    # 6. Convert PGA
    # ========================================================

    pga_cm_s2 = 10 ** log10_pga

    pga_g = pga_cm_s2 / 981.0

    # ========================================================
    # Return result
    # ========================================================

    return DarziResult(
        log10_pga=log10_pga,
        pga_cm_s2=pga_cm_s2,
        pga_g=pga_g,

        magnitude_term=magnitude_term,
        distance_term=distance_term,
        site_term=site_term,
        sof_term=sof_term,

        sigma=sigma,
    )