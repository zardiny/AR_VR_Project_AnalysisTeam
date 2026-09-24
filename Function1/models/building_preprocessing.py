from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd

from models.pipeline import run_earthquake_scenario
from models.distance import (
    calculate_epicentral_distance,
    calculate_hypocentral_distance_vectorized,
)


# ============================================================
# Configuration
# ============================================================

WGS84_CRS = "EPSG:4326"

# UTM Zone 40N is appropriate for the study area.
# This CRS is used only for geometric operations such as
# representative point and footprint area.
PROJECTED_CRS = "EPSG:32640"


# ============================================================
# Validation
# ============================================================

def _validate_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    building_id_col: str,
) -> None:

    if not isinstance(
        buildings_gdf,
        gpd.GeoDataFrame,
    ):
        raise TypeError(
            "Input must be a GeoDataFrame."
        )

    if buildings_gdf.empty:
        raise ValueError(
            "Input building GeoDataFrame is empty."
        )

    if buildings_gdf.crs is None:
        raise ValueError(
            "Input GeoDataFrame has no CRS."
        )

    # --------------------------------------------------------
    # Building ID
    # --------------------------------------------------------

    if building_id_col not in buildings_gdf.columns:
        raise ValueError(
            f"Required column '{building_id_col}' "
            "was not found in the GeoDataFrame."
        )

    if buildings_gdf[building_id_col].isna().any():
        raise ValueError(
            f"Column '{building_id_col}' contains null values."
        )

    if buildings_gdf[building_id_col].duplicated().any():
        raise ValueError(
            f"Column '{building_id_col}' must contain "
            "unique building IDs."
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    if buildings_gdf.geometry.isna().any():
        raise ValueError(
            "Some buildings have null geometry."
        )

    invalid_geometry = (
        ~buildings_gdf.geometry.is_valid
    )

    if invalid_geometry.any():

        count = int(
            invalid_geometry.sum()
        )

        raise ValueError(
            f"{count} building geometries are invalid. "
            "Please repair them before preprocessing."
        )

    allowed_types = {
        "Polygon",
        "MultiPolygon",
    }

    geometry_types = set(
        buildings_gdf.geometry.geom_type.unique()
    )

    invalid_types = (
        geometry_types - allowed_types
    )

    if invalid_types:
        raise ValueError(
            "Building geometries must be Polygon or MultiPolygon. "
            f"Invalid types: {sorted(invalid_types)}"
        )


# ============================================================
# Main preprocessing function
# ============================================================

def preprocess_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    earthquake_latitude: float,
    earthquake_longitude: float,
    magnitude: float,
    depth_km: float,
    vs30_raster: Union[str, Path],
    mechanism: str,
    building_id_col: str = "TARGET_FID",
    calculate_mmi: bool = True,
) -> gpd.GeoDataFrame:
    """
    Prepare building polygons for Function 2.

    A representative point is generated for each building.
    VS30 is sampled from the supplied raster at these points.
    PGA and optionally MMI are then calculated.

    The original building polygons are preserved.

    Returns
    -------
    geopandas.GeoDataFrame
        Original building polygons enriched with:

        - building_id
        - latitude
        - longitude
        - R_epi_km
        - R_hyp_km
        - footprint_area
        - vs30
        - pga_g
        - mmi (if calculate_mmi=True)
    """


    # 1. Validate input
    _validate_buildings(
        buildings_gdf=buildings_gdf,
        building_id_col=building_id_col,
    )

    # 2. Copy original GeoDataFrame
    buildings = buildings_gdf.copy()

    if building_id_col != "building_id":

        buildings = buildings.rename(
            columns={
                building_id_col: "building_id"
            }
        )

    # 3. Project buildings
    buildings_projected = buildings.to_crs(
        PROJECTED_CRS
    )

    # 4. Footprint area
    buildings_projected["footprint_area"] = (
        buildings_projected.geometry.area
    )

    # 5. Representative points
    representative_points = (
        buildings_projected
        .geometry
        .representative_point()
    )

    points_gdf = gpd.GeoDataFrame(
        buildings_projected[
            [
                "building_id",
                "footprint_area",
            ]
        ].copy(),
        geometry=representative_points,
        crs=PROJECTED_CRS,
    )

    # 6. Convert points to WGS84
    points_wgs84 = points_gdf.to_crs(
        WGS84_CRS
    )

    # 7. Coordinates
    points_wgs84["longitude"] = (
        points_wgs84.geometry.x
    )

    points_wgs84["latitude"] = (
        points_wgs84.geometry.y
    )

    # 8. Epicentral distance
    points_wgs84["R_epi_km"] = (
        calculate_epicentral_distance(
            site_latitudes=points_wgs84[
                "latitude"
            ].values,
            site_longitudes=points_wgs84[
                "longitude"
            ].values,
            earthquake_latitude=earthquake_latitude,
            earthquake_longitude=earthquake_longitude,
        )
    )

    # 9. Hypocentral distance
    points_wgs84["R_hyp_km"] = (
        calculate_hypocentral_distance_vectorized(
            epicentral_distance_km=points_wgs84[
                "R_epi_km"
            ].values,
            depth_km=depth_km,
        )
    )

    # 10. Prepare Function 1 input
    function1_input = points_wgs84.copy()

    # 11. Run Function 1
    result = run_earthquake_scenario(
        grid=function1_input,
        magnitude=magnitude,
        depth_km=depth_km,
        vs30_raster=Path(vs30_raster),
        mechanism=mechanism,
        calculate_mmi=calculate_mmi,
    )

    # Standardize VS30 column name
    if "vs30" not in result.columns:

        if "vs30_m_s" in result.columns:
            result = result.rename(
                columns={"vs30_m_s": "vs30"}
            )

        else:
            raise RuntimeError(
                "Function 1 output does not contain a VS30 column."
            )
    # 12. Validate Function 1 output
    if not isinstance(
        result,
        pd.DataFrame,
    ):
        raise TypeError(
            "Function 1 must return a pandas DataFrame "
            "or GeoDataFrame."
        )

    required_columns = [
        "pga_g",
        "vs30",
    ]

    if calculate_mmi:
        required_columns.append("mmi")

    missing_columns = [
        col
        for col in required_columns
        if col not in result.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Function 1 output is missing required columns: "
            f"{missing_columns}"
        )


    # 13. Ensure building IDs exist
    if "building_id" not in result.columns:

        if len(result) != len(function1_input):

            raise RuntimeError(
                "Function 1 output length does not match "
                "the number of input buildings."
            )

        result = result.copy()

        result.insert(
            0,
            "building_id",
            function1_input[
                "building_id"
            ].values,
        )


    # 14. Check uniqueness
    if result["building_id"].duplicated().any():

        raise RuntimeError(
            "Function 1 returned duplicate building IDs."
        )

    # 15. Select seismic outputs
    seismic_columns = [
        "building_id",
        "vs30",
        "pga_g",
    ]

    if calculate_mmi:
        seismic_columns.append(
            "mmi"
        )

    seismic_result = result[
        seismic_columns
    ].copy()


    # 16. Validate VS30
    if seismic_result["vs30"].isna().any():

        raise RuntimeError(
            "Some buildings do not have a valid VS30 value."
        )

    if not pd.api.types.is_numeric_dtype(
        seismic_result["vs30"]
    ):

        raise RuntimeError(
            "VS30 values must be numeric."
        )

    if (
        ~seismic_result["vs30"]
        .apply(pd.notna)
        .all()
    ):

        raise RuntimeError(
            "VS30 contains invalid values."
        )

    # 17. Validate PGA / MMI
    if seismic_result["pga_g"].isna().any():

        raise RuntimeError(
            "Some buildings do not have a valid PGA value."
        )

    if calculate_mmi:

        if seismic_result["mmi"].isna().any():

            raise RuntimeError(
                "Some buildings do not have a valid MMI value."
            )


    # 18. Join geometric information
    buildings = buildings.merge(
        points_wgs84[
            [
                "building_id",
                "latitude",
                "longitude",
                "R_epi_km",
                "R_hyp_km",
                "footprint_area",
            ]
        ],
        on="building_id",
        how="left",
        validate="one_to_one",
    )

    # 19. Join seismic information
    buildings = buildings.merge(
        seismic_result,
        on="building_id",
        how="left",
        validate="one_to_one",
    )


    # 20. Restore GeoDataFrame
    buildings = gpd.GeoDataFrame(
        buildings,
        geometry="geometry",
        crs=buildings_gdf.crs,
    )

    # 21. Final geometry validation
    geometry_types = set(
        buildings.geometry.geom_type.unique()
    )

    invalid_types = (
        geometry_types
        - {
            "Polygon",
            "MultiPolygon",
        }
    )

    if invalid_types:

        raise RuntimeError(
            "Original building polygon geometry was not "
            f"preserved. Found: {sorted(invalid_types)}"
        )

    return buildings