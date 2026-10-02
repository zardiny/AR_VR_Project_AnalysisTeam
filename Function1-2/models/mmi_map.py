import numpy as np

from models.gmice import calculate_mmi


def calculate_mmi_for_grid(
    grid,
    magnitude,
    pga_column="pga_g",
    distance_column="R_hyp_km"
):
    """
    Calculate MMI for each grid point using the
    appropriate GMICE.

    Parameters
    ----------
    grid : GeoDataFrame
        Computational grid containing PGA and distance.

    magnitude : float
        Earthquake moment magnitude (Mw).

    pga_column : str
        Name of PGA column in g.

    distance_column : str
        Name of source-to-site distance column in km.

    Returns
    -------
    GeoDataFrame
        Grid with MMI and GMICE model columns.
    """

    result = grid.copy()

    mmi_values = []
    gmice_models = []

    for _, row in result.iterrows():

        pga = row[pga_column]
        distance = row[distance_column]

        # Missing PGA
        if (
            pga is None
            or not np.isfinite(pga)
            or pga <= 0
        ):
            mmi_values.append(np.nan)
            gmice_models.append("Not_available")
            continue

        # Missing distance
        if (
            distance is None
            or not np.isfinite(distance)
            or distance < 0
        ):
            mmi_values.append(np.nan)
            gmice_models.append("Not_available")
            continue

        mmi, model = calculate_mmi(
            pga_g=pga,
            magnitude=magnitude,
            distance_km=distance
        )

        mmi_values.append(mmi)
        gmice_models.append(model)

    result["mmi"] = mmi_values
    result["gmice_model"] = gmice_models

    return result