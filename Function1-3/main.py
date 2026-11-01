
from pathlib import Path
from models.distance import calculate_epicentral_distance, calculate_hypocentral_distance_vectorized
from models.grid import create_grid
from models.pipeline import run_earthquake_scenario
from models.building_preprocessing import preprocess_buildings
from models.pipeline_f2 import run_building_damage_assessment
from models.pipeline_f3 import run_function3
from models.database_building_population import load_buildings_from_database, load_database_config

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_CONFIG_FILE = BASE_DIR / "database_config.yaml"
VS30_RASTER = BASE_DIR / "vs30_mosaic.tif"

OUTPUT_DIR = BASE_DIR / "export_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_CSV = OUTPUT_DIR / "earthquake_result.csv"
GRID_GEOJSON = OUTPUT_DIR / "earthquake_result.geojson"

BUILDING_SHP = OUTPUT_DIR / "building_damage.shp"

BUILDING_CASUALTIES_CSV = OUTPUT_DIR / "building_casualties.csv"
BLOCK_CASUALTIES_CSV = OUTPUT_DIR / "block_casualties.csv"


# ============================================================
# EARTHQUAKE SCENARIO
# ============================================================

EARTHQUAKE_LATITUDE = 36.75
EARTHQUAKE_LONGITUDE = 54.65

MAGNITUDE = 6.0
DEPTH_KM = 10.0
MECHANISM = "reverse"
EARTHQUAKE_TIME = "7:30"


# ============================================================
# GRID SETTINGS
# ============================================================

GRID_MIN_LON = 54.364163
GRID_MIN_LAT = 36.790849

GRID_MAX_LON = 54.492416
GRID_MAX_LAT = 36.867695

GRID_SPACING_M = 100


# ============================================================
# PREPARE BUILDING INPUTS
# ============================================================

def prepare_building_inputs(buildings):
    """Prepare fields required by Function 2."""

    buildings = buildings.copy()

    buildings["type"] = buildings["type_code"]

    return buildings


# ============================================================
# FUNCTION 2 SUMMARY
# ============================================================

def print_building_summary(result):
    """Print Function 2 damage assessment summary."""

    print("\n" + "=" * 70)
    print("BUILDING DAMAGE ASSESSMENT SUMMARY")
    print("=" * 70)

    print(f"Total buildings: {len(result):,}")

    summary_fields = {
        "vs30": ("VS30", ""),
        "pga_g": ("PGA", " g"),
        "mmi": ("MMI", ""),
        "damage_ratio": ("DAMAGE RATIO", "%"),
    }

    for field, (label, unit) in summary_fields.items():

        if field not in result.columns:
            continue

        values = result[field].dropna()

        print(f"\n--- {label} ---")
        print(f"Valid values: {len(values):,}")

        if len(values) > 0:
            print(f"Mean: {values.mean():.6f}{unit}")
            print(f"Std : {values.std():.6f}")
            print(f"Min : {values.min():.6f}{unit}")
            print(f"Max : {values.max():.6f}{unit}")

    if "damage_status" in result.columns:
        print("\n--- DAMAGE STATUS ---")
        print(
            result["damage_status"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "building_class" in result.columns:
        print("\n--- BUILDING CLASS ---")
        print(
            result["building_class"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )


# ============================================================
# EXPORT BUILDING DAMAGE RESULT
# ============================================================

def export_building_result(final_result):
    """Export Function 2 building damage result as Shapefile."""

    result = final_result.copy()

    # Geometry preparation
    if "geom" in result.columns:
        result = result.set_geometry("geom")

    if "geometry" in result.columns and "geom" in result.columns:
        result = result.drop(columns=["geometry"])

    if "geom" in result.columns:
        result = result.rename_geometry("geometry")

    if result.geometry is None:
        raise RuntimeError("No active geometry column found.")

    # Geometry diagnostics
    print("\n" + "=" * 70)
    print("GEOMETRY CHECK BEFORE SHAPEFILE EXPORT")
    print("=" * 70)

    print(f"Active geometry : {result.geometry.name}")
    print(f"CRS             : {result.crs}")

    print("\nGeometry types:")
    print(result.geometry.geom_type.value_counts(dropna=False).to_string())

    print(f"\nNull geometries  : {result.geometry.isna().sum():,}")
    print(f"Empty geometries : {result.geometry.is_empty.sum():,}")
    print(f"Invalid geometries: {(~result.geometry.is_valid).sum():,}")

    # Shapefile field mapping
    field_mapping = {
        "id_building": "bldg_id",
        "source_fid": "src_fid",
        "source_dataset": "src_data",
        "type_name": "type_name",
        "type_code": "type_code",
        "roof_type": "roof_type",
        "facade_type": "facade",
        "description": "descr",
        "area_source": "area_src",
        "jica_class": "jica_cls",
        "footprint_area": "fp_area",
        "building_class": "bldg_cls",
        "damage_ratio": "dmg_ratio",
        "damage_status": "dmg_state",
    }

    field_mapping = {
        old: new
        for old, new in field_mapping.items()
        if old in result.columns
    }

    shapefile_result = result.rename(columns=field_mapping)

    print("\n" + "=" * 70)
    print("EXPORTING BUILDING DAMAGE SHAPEFILE")
    print("=" * 70)

    print(f"Output: {BUILDING_SHP}")

    shapefile_result.to_file(
        BUILDING_SHP,
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )

    print("\nBuilding Shapefile successfully created.")

    print("\nShapefile fields:")
    print([
        col
        for col in shapefile_result.columns
        if col != "geometry"
    ])

    return shapefile_result


# ============================================================
# PREPARE FUNCTION 2 OUTPUT FOR FUNCTION 3
# ============================================================

def prepare_function3_input(function2_result):
    """
    Normalize Function 2 field names for function3_pipeline.py.

    This function only changes column names.
    It does not infer or modify damage states.
    """

    result = function2_result.copy()

    aliases = {
        "id_building": "building_id",
        "damage_ratio": "dmg_ratio",
        "damage_status": "dmg_state",
    }

    rename_mapping = {}

    for old_name, new_name in aliases.items():

        if old_name in result.columns and new_name not in result.columns:
            rename_mapping[old_name] = new_name

    result = result.rename(columns=rename_mapping)

    required_fields = [
        "building_id",
        "type_code",
        "year",
        "floors",
        "dmg_ratio",
        "dmg_state",
    ]

    missing = [
        field
        for field in required_fields
        if field not in result.columns
    ]

    if missing:
        raise ValueError(
            "Function 2 output is missing fields required by Function 3: "
            f"{missing}\n"
            f"Available fields: {result.columns.tolist()}"
        )

    if result["building_id"].isna().any():
        raise ValueError(
            "Function 2 output contains missing building IDs."
        )

    if result["building_id"].duplicated().any():
        raise ValueError(
            "Function 2 output contains duplicate building IDs."
        )

    print("\nFunction 3 input fields successfully prepared.")

    print(
        "Building ID field : building_id\n"
        "Damage ratio field: dmg_ratio\n"
        "Damage state field: dmg_state"
    )

    return result


# ============================================================
# FUNCTION 3 - CASUALTY ESTIMATION
# ============================================================

def execute_function3(final_result, config, earthquake_time):
    """
    Execute the independent Function 3 pipeline.

    Function 3 handles:
    - Loading temporal population data from PostgreSQL
    - Joining Function 2 damage results with population data
    - Deterministic casualty estimation
    - Block-level fallback estimation
    - Returning results for local CSV export (no casualty DB writes)
    """

    print("\n" + "=" * 70)
    print("FUNCTION 3 - CASUALTY ESTIMATION")
    print("=" * 70)

    function3_input = prepare_function3_input(final_result)

    building_output, block_output, validation = run_function3(
        function2_result=function3_input,
        config=config,
        earthquake_time=earthquake_time,
    )

    # Export CSV outputs
    building_output.to_csv(
        BUILDING_CASUALTIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    block_output.to_csv(
        BLOCK_CASUALTIES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFunction 3 completed successfully.")

    print(f"Building casualties CSV: {BUILDING_CASUALTIES_CSV}")
    print(f"Block casualties CSV   : {BLOCK_CASUALTIES_CSV}")

    print("\nBuildings with casualty outputs:")
    print(f"{len(building_output):,}")

    print("\nBlocks with casualty outputs:")
    print(f"{len(block_output):,}")

    print("\nFunction 3 validation:")
    print(validation)

    return building_output, block_output, validation


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("EARTHQUAKE ASSESSMENT SYSTEM")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load database configuration
    # --------------------------------------------------------

    config = load_database_config(DATABASE_CONFIG_FILE)

    # --------------------------------------------------------
    # 2. Load building data
    # --------------------------------------------------------

    buildings = load_buildings_from_database(config)

    buildings = prepare_building_inputs(buildings)

    # --------------------------------------------------------
    # 3. Function 1 - Grid-based ground-motion assessment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUNCTION 1 - GRID-BASED GROUND MOTION")
    print("=" * 70)

    grid = create_grid(
        min_lon=GRID_MIN_LON,
        min_lat=GRID_MIN_LAT,
        max_lon=GRID_MAX_LON,
        max_lat=GRID_MAX_LAT,
        spacing_m=GRID_SPACING_M,
    )

    grid["R_epi_km"] = calculate_epicentral_distance(
        site_latitudes=grid["latitude"].values,
        site_longitudes=grid["longitude"].values,
        earthquake_latitude=EARTHQUAKE_LATITUDE,
        earthquake_longitude=EARTHQUAKE_LONGITUDE,
    )

    grid["R_hyp_km"] = calculate_hypocentral_distance_vectorized(
        epicentral_distance_km=grid["R_epi_km"].values,
        depth_km=DEPTH_KM,
    )

    grid_result = run_earthquake_scenario(
        grid=grid,
        magnitude=MAGNITUDE,
        depth_km=DEPTH_KM,
        vs30_raster=VS30_RASTER,
        mechanism=MECHANISM,
        calculate_mmi=True,
    )

    grid_result.to_csv(
        GRID_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    grid_result.to_file(
        GRID_GEOJSON,
        driver="GeoJSON",
    )

    print("\nFunction 1 completed successfully.")
    print(f"Grid CSV    : {GRID_CSV}")
    print(f"Grid GeoJSON: {GRID_GEOJSON}")

    # --------------------------------------------------------
    # 4. Function 1 - Building-level preprocessing
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUNCTION 1 - BUILDING-LEVEL PREPROCESSING")
    print("=" * 70)

    seismic_result = preprocess_buildings(
        buildings_gdf=buildings,
        earthquake_latitude=EARTHQUAKE_LATITUDE,
        earthquake_longitude=EARTHQUAKE_LONGITUDE,
        magnitude=MAGNITUDE,
        depth_km=DEPTH_KM,
        vs30_raster=VS30_RASTER,
        mechanism=MECHANISM,
        building_id_col="id_building",
        calculate_mmi=True,
    )

    print("\nBuilding-level seismic preprocessing completed.")
    print(f"Processed buildings: {len(seismic_result):,}")

    # --------------------------------------------------------
    # 5. Function 2 - Building damage assessment
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUNCTION 2 - BUILDING DAMAGE ASSESSMENT")
    print("=" * 70)

    final_result = run_building_damage_assessment(seismic_result)

    if final_result is None or final_result.empty:
        raise RuntimeError(
            "Function 2 returned an empty result."
        )

    print_building_summary(final_result)
    print(final_result.head())

    # --------------------------------------------------------
    # 6. Export building damage result
    # --------------------------------------------------------

    export_building_result(final_result)

    # --------------------------------------------------------
    # 7. Function 3 - Casualty estimation
    # --------------------------------------------------------

    building_casualties, block_casualties, validation = execute_function3(
        final_result=final_result,
        config=config,
        earthquake_time=EARTHQUAKE_TIME,
    )
    print(building_casualties)
    print(block_casualties)
    # --------------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EARTHQUAKE ASSESSMENT COMPLETED")
    print("=" * 70)

    print("\nScenario:")
    print(f"Latitude  : {EARTHQUAKE_LATITUDE}")
    print(f"Longitude : {EARTHQUAKE_LONGITUDE}")
    print(f"Magnitude : {MAGNITUDE}")
    print(f"Depth     : {DEPTH_KM} km")
    print(f"Mechanism : {MECHANISM}")
    print(f"Occurrence time: {EARTHQUAKE_TIME}")

    print("\nOutput files:")
    print(f"Grid CSV                 : {GRID_CSV}")
    print(f"Grid GeoJSON             : {GRID_GEOJSON}")
    print(f"Building damage Shapefile: {BUILDING_SHP}")
    print(f"Building casualties CSV  : {BUILDING_CASUALTIES_CSV}")
    print(f"Block casualties CSV     : {BLOCK_CASUALTIES_CSV}")

    print("\nFinal record counts:")
    print(f"Buildings assessed : {len(final_result):,}")
    print(f"Building casualties: {len(building_casualties):,}")
    print(f"Block casualties   : {len(block_casualties):,}")

    print("\nAll assessment functions completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
