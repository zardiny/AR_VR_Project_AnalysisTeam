import numpy as np
import rasterio


def sample_vs30(grid, vs30_raster):
    """
    Extract VS30 from a raster for each grid point.
    """

    if grid.crs is None:
        raise ValueError("Grid CRS is not defined.")

    result = grid.copy()

    with rasterio.open(vs30_raster) as src:

        # Transform grid to raster CRS
        grid_raster_crs = result.to_crs(src.crs)

        coordinates = [
            (point.x, point.y)
            for point in grid_raster_crs.geometry
        ]

        sampled_values = list(
            src.sample(coordinates)
        )

        values = np.array(
            [value[0] for value in sampled_values],
            dtype=float
        )

        # Convert raster NoData to NaN
        if src.nodata is not None:
            values[values == src.nodata] = np.nan

    result["vs30_m_s"] = values

    return result


def classify_vs30(vs30):
    """
    Darzi site classification based on VS30.

    Class I:
        VS30 > 750 m/s

    Class II:
        375 < VS30 <= 750 m/s

    Class III:
        VS30 <= 375 m/s
    """

    if np.isnan(vs30):
        return np.nan

    if vs30 > 750:
        return 1

    elif vs30 > 375:
        return 2

    else:
        return 3


def add_site_class(grid):
    """
    Add Darzi site class to the grid.
    """

    result = grid.copy()

    result["site_class"] = (
        result["vs30_m_s"]
        .apply(classify_vs30)
    )

    return result