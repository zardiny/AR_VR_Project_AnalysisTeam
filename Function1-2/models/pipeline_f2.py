from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd

from models.classification import classify_building
from models.damage_ratio import calculate_damage_ratio


REQUIRED_COLUMNS = {
    "type",
    "year",
    "floors",
    "mmi",
}


def run_building_damage_assessment(
    buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Run Function 2: JICA-based building damage assessment.

    Input:
        Building polygons containing:
        - type
        - year
        - floors
        - mmi

    Output:
        Original building polygons with:
        - building_class
        - damage_ratio
        - damage_status
    """

    # Validation
    if not isinstance(buildings_gdf, gpd.GeoDataFrame):
        raise TypeError(
            "buildings_gdf must be a GeoDataFrame."
        )

    if buildings_gdf.empty:
        raise ValueError(
            "Input GeoDataFrame is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(buildings_gdf.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input GeoDataFrame is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if buildings_gdf.geometry.isna().any():
        raise ValueError(
            "Input GeoDataFrame contains null geometries."
        )


    # Copy input
    result = buildings_gdf.copy()

    # Calculate building class and damage ratio
    building_classes = []
    damage_ratios = []
    damage_statuses = []

    for _, row in result.iterrows():

        building_class = classify_building(
            building_type=row["type"],
            year=row["year"],
            floors=row["floors"],
        )

        damage_ratio = calculate_damage_ratio(
            MMI=row["mmi"],
            building_class=building_class,
        )

        # Damage status(based on report ATC-13)
        
        if damage_ratio is None:
            damage_status = None

        elif damage_ratio == 0:
            damage_status = "none"

        elif damage_ratio < 1:
            damage_status = "slight"

        elif damage_ratio < 10:
            damage_status = "light"

        elif damage_ratio < 30:
            damage_status = "moderate"

        elif damage_ratio < 60:
            damage_status = "heavy"

        elif damage_ratio < 100:
            damage_status = "major"

        else:
            damage_status = "destroyed"

        building_classes.append(building_class)
        damage_ratios.append(damage_ratio)
        damage_statuses.append(damage_status)

    # Add results
    result["building_class"] = pd.Series(
        building_classes,
        index=result.index,
        dtype="Int64",
    )

    result["damage_ratio"] = pd.Series(
        damage_ratios,
        index=result.index,
        dtype="float64",
    )

    result["damage_status"] = damage_statuses


    # Final validation
    valid_damage = result["damage_ratio"].notna()

    if valid_damage.any():

        if (
            result.loc[valid_damage, "damage_ratio"] < 0
        ).any():

            raise ValueError(
                "Damage ratio contains values below 0."
            )

        if (
            result.loc[valid_damage, "damage_ratio"] > 100
        ).any():

            raise ValueError(
                "Damage ratio contains values above 100."
            )

        if not result.loc[
            valid_damage, "damage_ratio"
        ].apply(math.isfinite).all():

            raise ValueError(
                "Damage ratio contains non-finite values."
            )

    return result