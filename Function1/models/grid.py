import numpy as np
import geopandas as gpd

from shapely.geometry import box, Point
from pyproj import Geod


# WGS84 ellipsoid
GEOD = Geod(ellps="WGS84")


def create_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    spacing_m: float = 500.0
) -> gpd.GeoDataFrame:
    """
    Create a regular computational grid over a geographic
    bounding box.

    Parameters
    ----------
    min_lon, min_lat : float
        Lower-left corner of study area.

    max_lon, max_lat : float
        Upper-right corner of study area.

    spacing_m : float
        Approximate grid spacing in meters.

    Returns
    -------
    GeoDataFrame
        Grid points in WGS84.
    """

    if min_lon >= max_lon:
        raise ValueError("min_lon must be smaller than max_lon.")

    if min_lat >= max_lat:
        raise ValueError("min_lat must be smaller than max_lat.")

    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive.")

    # Approximate conversion from meters to degrees
    lat_center = (min_lat + max_lat) / 2

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = (
        111_320.0 *
        np.cos(np.radians(lat_center))
    )

    dlat = spacing_m / meters_per_degree_lat
    dlon = spacing_m / meters_per_degree_lon

    lats = np.arange(
        min_lat,
        max_lat + dlat,
        dlat
    )

    lons = np.arange(
        min_lon,
        max_lon + dlon,
        dlon
    )

    points = [
        Point(lon, lat)
        for lat in lats
        for lon in lons
    ]

    grid = gpd.GeoDataFrame(
        {
            "latitude": [p.y for p in points],
            "longitude": [p.x for p in points],
        },
        geometry=points,
        crs="EPSG:4326"
    )

    return grid