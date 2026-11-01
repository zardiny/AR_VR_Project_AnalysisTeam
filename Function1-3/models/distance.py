import numpy as np
from pyproj import Geod


GEOD = Geod(ellps="WGS84")


def calculate_epicentral_distance(
    site_latitudes,
    site_longitudes,
    earthquake_latitude,
    earthquake_longitude
):
    """
    Calculate epicentral distance using WGS84 geodesic distance.

    Returns distance in kilometers.
    """

    site_latitudes = np.asarray(site_latitudes)
    site_longitudes = np.asarray(site_longitudes)

    earthquake_latitudes = np.full_like(
        site_latitudes,
        earthquake_latitude,
        dtype=float
    )

    earthquake_longitudes = np.full_like(
        site_longitudes,
        earthquake_longitude,
        dtype=float
    )

    _, _, distance_m = GEOD.inv(
        earthquake_longitudes,
        earthquake_latitudes,
        site_longitudes,
        site_latitudes
    )

    return np.asarray(distance_m) / 1000.0


def calculate_hypocentral_distance_vectorized(
    epicentral_distance_km,
    depth_km
):
    """
    Calculate hypocentral distance for all grid points.

    R_hyp = sqrt(R_epi^2 + Depth^2)
    """

    epicentral_distance_km = np.asarray(
        epicentral_distance_km,
        dtype=float
    )

    return np.sqrt(
        epicentral_distance_km**2 +
        depth_km**2
    )