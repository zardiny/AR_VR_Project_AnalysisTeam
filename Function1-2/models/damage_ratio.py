from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional



# JICA MMI shifts
MMI_SHIFTS = {
    1: 0.0,
    2: 0.5,
    3: 0.4,
    4: 0.5,
    5: -0.5,
    6: -0.7,
    7: -1.0,
    8: -1.2,
    9: -2.0,
}

# JICA base damage curve parameters

JICA_MU = 7.7093
JICA_SIGMA = 0.65779

# Validation

def _validate_mmi(MMI: float) -> float:
    """Validate and convert MMI to a finite float."""

    if MMI is None:
        raise ValueError(
            "MMI cannot be None."
        )

    try:
        MMI = float(MMI)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MMI must be numeric. Received: {MMI!r}"
        ) from exc

    if not math.isfinite(MMI):
        raise ValueError(
            f"MMI must be finite. Received: {MMI!r}"
        )

    return MMI


def _validate_building_class(
    building_class: int,
) -> None:
    """Validate JICA building class."""

    if building_class is None:
        return

    if isinstance(building_class, bool):
        raise ValueError(
            "building_class must be an integer between 1 and 9."
        )

    if not isinstance(building_class, int):
        raise ValueError(
            "building_class must be an integer between 1 and 9."
        )

    if building_class not in MMI_SHIFTS:
        raise ValueError(
            "building_class must be an integer between 1 and 9."
        )



# Main damage ratio function

def calculate_damage_ratio(
    MMI: float,
    building_class: Optional[int],
) -> Optional[float]:
    """
    Calculate building damage ratio (%) based on MMI
    and JICA building class.

    Parameters
    ----------
    MMI : float
        Modified Mercalli Intensity.

    building_class : int or None
        JICA building class (1-9).

        None indicates that the building could not be
        reliably classified.

    Returns
    -------
    float or None
        Damage ratio in percent [0, 100].

        None is returned when building_class is None.
    """

    # Unknown building class

    if building_class is None:
        return None

    # Validate inputs

    MMI = _validate_mmi(MMI)

    _validate_building_class(
        building_class
    )

    # Apply class-specific MMI shift

    shift = MMI_SHIFTS[building_class]

    adjusted_MMI = MMI - shift


    # Calculate damage ratio

    z = (
        adjusted_MMI - JICA_MU
    ) / JICA_SIGMA

    damage_ratio = (
        100.0 * NormalDist().cdf(z)
    )

    # Numerical safety
    
    damage_ratio = max(
        0.0,
        min(100.0, damage_ratio)
    )

    return damage_ratio