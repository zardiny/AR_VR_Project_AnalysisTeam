import numpy as np
import pandas as pd

from models.darzi import calculate_darzi_pga
from models.validation import check_darzi_validity

def calculate_pga_for_grid(
    grid,
    magnitude,
    depth_km,
    mechanism=None
):
    """
    Calculate Darzi PGA for every grid point
    using the site class assigned to each point.
    """

    if "R_hyp_km" not in grid.columns:
        raise ValueError(
            "Grid must contain 'R_hyp_km'."
        )

    if "site_class" not in grid.columns:
        raise ValueError(
            "Grid must contain 'site_class'."
        )

    result = grid.copy()

    pga_results = []

    for distance, site_class in zip(
        result["R_hyp_km"].values,
        result["site_class"].values
    ):

        if np.isnan(site_class):

            pga_results.append(None)

            continue

        pga = calculate_darzi_pga(
            magnitude=magnitude,
            hypocentral_distance_km=distance,
            site_class=int(site_class),
            mechanism=mechanism
        )

        pga_results.append(pga)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result["pga_cm_s2"] = [
        r.pga_cm_s2 if r is not None else np.nan
        for r in pga_results
    ]

    result["pga_g"] = [
        r.pga_g if r is not None else np.nan
        for r in pga_results
    ]

    result["log10_pga"] = [
        r.log10_pga if r is not None else np.nan
        for r in pga_results
    ]

    result["magnitude_term"] = [
        r.magnitude_term if r is not None else np.nan
        for r in pga_results
    ]

    result["distance_term"] = [
        r.distance_term if r is not None else np.nan
        for r in pga_results
    ]

    result["site_term"] = [
        r.site_term if r is not None else np.nan
        for r in pga_results
    ]

    result["sof_term"] = [
        r.sof_term if r is not None else np.nan
        for r in pga_results
    ]

    result["sigma"] = [
        r.sigma if r is not None else np.nan
        for r in pga_results
    ]

    result["magnitude_valid"] = [
    check_darzi_validity(
        magnitude=magnitude,
        distance_km=d
    )["magnitude_valid"]
    for d in result["R_hyp_km"]
    ]

    result["distance_valid"] = [
    check_darzi_validity(
        magnitude=magnitude,
        distance_km=d
    )["distance_valid"]
    for d in result["R_hyp_km"]
    ]

    result["extrapolated"] = [
    check_darzi_validity(
        magnitude=magnitude,
        distance_km=d
    )["extrapolated"]
    for d in result["R_hyp_km"]
    ]

    return result