from pathlib import Path

import geopandas as gpd

from models.output import (
    save_csv,
    save_geojson,
)

from models.pipeline import run_earthquake_scenario

from models.grid import create_grid

from models.distance import (
    calculate_epicentral_distance,
    calculate_hypocentral_distance_vectorized,
)

from models.building_preprocessing import preprocess_buildings
from models.pipeline_f2 import run_building_damage_assessment


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BUILDINGS_FILE = (
    BASE_DIR / "shp" / "UTM40.shp"
)

VS30_RASTER = (
    BASE_DIR / "vs30_mosaic.tif"
)

OUTPUT_DIR = (
    BASE_DIR / "export_sample"
)


# ============================================================
# Output files
# ============================================================

GRID_CSV = (
    OUTPUT_DIR / "earthquake_result.csv"
)

GRID_GEOJSON = (
    OUTPUT_DIR / "earthquake_result.geojson"
)

BUILDING_GPKG = (
    OUTPUT_DIR / "building_damage.gpkg"
)

BUILDING_LAYER = "building_damage"


# ============================================================
# Earthquake scenario
# ============================================================

EARTHQUAKE_LATITUDE = 36.75
EARTHQUAKE_LONGITUDE = 54.65

MAGNITUDE = 6.0
DEPTH_KM = 10.0

MECHANISM = "reverse"


# ============================================================
# Grid configuration
# ============================================================

GRID_MIN_LON = 54.751251
GRID_MIN_LAT = 36.895583

GRID_MAX_LON = 54.754489
GRID_MAX_LAT = 36.898588

GRID_SPACING_M = 100


# ============================================================
# Building configuration
# ============================================================

BUILDING_ID_COL = "TARGET_FID"


# ============================================================
# Main workflow
# ============================================================

def main():

    print("=" * 70)
    print("EARTHQUAKE ANALYSIS WORKFLOW")
    print("=" * 70)

    print("\nEarthquake scenario:")
    print(f"  Latitude   : {EARTHQUAKE_LATITUDE}")
    print(f"  Longitude  : {EARTHQUAKE_LONGITUDE}")
    print(f"  Magnitude  : Mw {MAGNITUDE}")
    print(f"  Depth      : {DEPTH_KM} km")
    print(f"  Mechanism  : {MECHANISM}")

    # ========================================================
    # 1. Check input files
    # ========================================================

    print("\n" + "=" * 70)
    print("[1/6] Checking input files...")
    print("=" * 70)

    if not BUILDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Building file not found:\n{BUILDINGS_FILE}"
        )

    if not VS30_RASTER.exists():
        raise FileNotFoundError(
            f"VS30 raster not found:\n{VS30_RASTER}"
        )

    print(f"Building file : {BUILDINGS_FILE}")
    print(f"VS30 raster   : {VS30_RASTER}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 2. Grid-based earthquake analysis
    # ========================================================

    print("\n" + "=" * 70)
    print("[2/6] Creating analysis grid...")
    print("=" * 70)

    grid = create_grid(
        min_lon=GRID_MIN_LON,
        min_lat=GRID_MIN_LAT,
        max_lon=GRID_MAX_LON,
        max_lat=GRID_MAX_LAT,
        spacing_m=GRID_SPACING_M,
    )

    print(
        f"Grid cells/points created: "
        f"{len(grid):,}"
    )

    # --------------------------------------------------------
    # Epicentral distance
    # --------------------------------------------------------

    print("\nCalculating epicentral distances...")

    grid["R_epi_km"] = calculate_epicentral_distance(
        site_latitudes=grid["latitude"].values,
        site_longitudes=grid["longitude"].values,
        earthquake_latitude=EARTHQUAKE_LATITUDE,
        earthquake_longitude=EARTHQUAKE_LONGITUDE,
    )

    # --------------------------------------------------------
    # Hypocentral distance
    # --------------------------------------------------------

    print("Calculating hypocentral distances...")

    grid["R_hyp_km"] = calculate_hypocentral_distance_vectorized(
        epicentral_distance_km=grid["R_epi_km"].values,
        depth_km=DEPTH_KM,
    )

    # --------------------------------------------------------
    # Function 1 on grid
    # --------------------------------------------------------

    print("\nRunning Function 1 on grid...")
    print("  VS30 → PGA → MMI")

    grid_result = run_earthquake_scenario(
        grid=grid,

        magnitude=MAGNITUDE,
        depth_km=DEPTH_KM,

        vs30_raster=VS30_RASTER,

        mechanism=MECHANISM,

        calculate_mmi=True,
    )

    print(
        f"Grid analysis completed: "
        f"{len(grid_result):,} points"
    )

    # --------------------------------------------------------
    # Save grid results
    # --------------------------------------------------------

    print("\nSaving grid results...")

    save_csv(
        grid_result,
        GRID_CSV,
    )

    save_geojson(
        grid_result,
        GRID_GEOJSON,
    )

    print(f"CSV saved     : {GRID_CSV}")
    print(f"GeoJSON saved : {GRID_GEOJSON}")

    # ========================================================
    # 3. Load building data
    # ========================================================

    print("\n" + "=" * 70)
    print("[3/6] Loading building data...")
    print("=" * 70)

    buildings = gpd.read_file(
        BUILDINGS_FILE
    )

    print(
        f"Buildings loaded: "
        f"{len(buildings):,}"
    )

    # ========================================================
    # 4. Function 1 on buildings
    # ========================================================

    print("\n" + "=" * 70)
    print("[4/6] Running Function 1 for buildings...")
    print("=" * 70)

    print("  VS30 → PGA → MMI")

    seismic_result = preprocess_buildings(
        buildings_gdf=buildings,

        earthquake_latitude=EARTHQUAKE_LATITUDE,
        earthquake_longitude=EARTHQUAKE_LONGITUDE,

        magnitude=MAGNITUDE,
        depth_km=DEPTH_KM,

        vs30_raster=VS30_RASTER,

        mechanism=MECHANISM,

        building_id_col=BUILDING_ID_COL,

        calculate_mmi=True,
    )

    print(
        f"Function 1 completed: "
        f"{len(seismic_result):,} buildings"
    )

    # --------------------------------------------------------
    # Validate Function 1 output
    # --------------------------------------------------------

    required_seismic_columns = {
        "building_id",
        "vs30",
        "pga_g",
        "mmi",
    }

    missing = (
        required_seismic_columns
        - set(seismic_result.columns)
    )

    if missing:
        raise RuntimeError(
            "Function 1 output is missing columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # 5. Function 2
    # ========================================================

    print("\n" + "=" * 70)
    print("[5/6] Running Function 2...")
    print("=" * 70)

    print("  Building classification")
    print("  JICA damage ratio")
    print("  ATC-13 damage status")

    final_result = run_building_damage_assessment(
        seismic_result
    )

    print(
        f"Function 2 completed: "
        f"{len(final_result):,} buildings"
    )

    # ========================================================
    # 6. Final summary and save
    # ========================================================

    print("\n" + "=" * 70)
    print("[6/6] Final results")
    print("=" * 70)

    print(
        f"\nTotal buildings: "
        f"{len(final_result):,}"
    )

    # --------------------------------------------------------
    # VS30
    # --------------------------------------------------------

    if "vs30" in final_result.columns:

        print("\nVS30:")
        print(
            final_result["vs30"].describe()
        )

    # --------------------------------------------------------
    # PGA
    # --------------------------------------------------------

    if "pga_g" in final_result.columns:

        print("\nPGA (g):")
        print(
            final_result["pga_g"].describe()
        )

    # --------------------------------------------------------
    # MMI
    # --------------------------------------------------------

    if "mmi" in final_result.columns:

        print("\nMMI:")
        print(
            final_result["mmi"].describe()
        )

    # --------------------------------------------------------
    # Building classes
    # --------------------------------------------------------

    if "building_class" in final_result.columns:

        print("\nBuilding classes:")

        print(
            final_result[
                "building_class"
            ].value_counts(
                dropna=False
            ).sort_index()
        )

    # --------------------------------------------------------
    # Damage ratio
    # --------------------------------------------------------

    if "damage_ratio" in final_result.columns:

        valid_damage = (
            final_result[
                "damage_ratio"
            ].notna()
        )

        print(
            f"\nBuildings with damage ratio: "
            f"{valid_damage.sum():,}"
        )

        print(
            f"Buildings without damage ratio: "
            f"{(~valid_damage).sum():,}"
        )

        if valid_damage.any():

            damage = final_result.loc[
                valid_damage,
                "damage_ratio",
            ]

            print("\nDamage ratio (%):")

            print(
                damage.describe()
            )

    # --------------------------------------------------------
    # Damage status
    # --------------------------------------------------------

    if "damage_status" in final_result.columns:

        print("\nATC-13 damage status:")

        print(
            final_result[
                "damage_status"
            ].value_counts(
                dropna=False
            )
        )

    # --------------------------------------------------------
    # Save building results
    # --------------------------------------------------------

    final_result.to_file(
        BUILDING_GPKG,
        layer=BUILDING_LAYER,
        driver="GPKG",
    )

    print(
        f"\nBuilding results saved to:"
    )

    print(
        BUILDING_GPKG.resolve()
    )

    # ========================================================
    # Completion
    # ========================================================

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nOutputs:")

    print(f"  Grid CSV:")
    print(f"    {GRID_CSV}")

    print(f"\n  Grid GeoJSON:")
    print(f"    {GRID_GEOJSON}")

    print(f"\n  Building damage GeoPackage:")
    print(f"    {BUILDING_GPKG}")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    main()