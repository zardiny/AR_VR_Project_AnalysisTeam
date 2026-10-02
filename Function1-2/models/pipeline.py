from models.site_effect import sample_vs30, add_site_class
from models.pga_map import calculate_pga_for_grid
from models.mmi_map import calculate_mmi_for_grid
import math


def run_earthquake_scenario(
    grid,
    magnitude,
    depth_km,
    vs30_raster,
    mechanism="reverse",
    calculate_mmi=True
):
    """
    Run the complete earthquake ground-motion workflow.

    Parameters
    ----------
    grid : GeoDataFrame
        Computational grid.

    magnitude : float
        Earthquake moment magnitude (Mw).

    depth_km : float
        Earthquake focal depth in km.

    vs30_raster : str
        Path to VS30 raster.

    mechanism : str
        Earthquake mechanism:
        "reverse", "strike-slip", or "normal".

    calculate_mmi : bool
        Whether to calculate MMI.

    Returns
    -------
    GeoDataFrame
        Final result containing PGA and optionally MMI.
    """

    # -----------------------------------------------------
    # 1. Extract VS30
    # -----------------------------------------------------

    result = sample_vs30(
        grid=grid,
        vs30_raster=vs30_raster
    )

    # -----------------------------------------------------
    # 2. Classify site conditions
    # -----------------------------------------------------

    result = add_site_class(result)

    # -----------------------------------------------------
    # 3. Calculate PGA
    # -----------------------------------------------------

    result = calculate_pga_for_grid(
        grid=result,
        magnitude=magnitude,
        depth_km=depth_km,
        mechanism=mechanism
    )

    # -----------------------------------------------------
    # 4. Calculate MMI
    # -----------------------------------------------------

    if calculate_mmi:

        result = calculate_mmi_for_grid(
            grid=result,
            magnitude=magnitude
        )

    # --------------------------------------------------------
    # 6. Final validation
    # --------------------------------------------------------

    if result["R_hyp_km"].isna().any():
        raise ValueError(
            "Hypocentral distance contains NaN values."
        )

    if not result["R_hyp_km"].apply(
        math.isfinite
    ).all():
        raise ValueError(
            "Hypocentral distance contains non-finite values."
        )

    if result["pga_g"].isna().any():
        raise ValueError(
            "PGA contains NaN values."
        )

    if not result["pga_g"].apply(
        math.isfinite
    ).all():
        raise ValueError(
            "PGA contains non-finite values."
        )

    if (result["pga_g"] <= 0).any():
        raise ValueError(
            "PGA values must be positive."
        )
    
    if (result["log10_pga"] <= 0).any():
        raise ValueError(
            "log10_pga values must be positive."
        )

    if result["mmi"].isna().any():
        raise ValueError(
            "MMI contains NaN values."
        )

    if not result["mmi"].apply(
        math.isfinite
    ).all():
        raise ValueError(
            "MMI contains non-finite values."
        )
    return result