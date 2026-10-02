from pathlib import Path

import geopandas as gpd
import yaml

from models.distance import (
    calculate_epicentral_distance,
    calculate_hypocentral_distance_vectorized,
)

from models.grid import create_grid
from models.pipeline import run_earthquake_scenario
from models.building_preprocessing import preprocess_buildings
from models.pipeline_f2 import run_building_damage_assessment


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


# ============================================================
# EARTHQUAKE SCENARIO
# ============================================================

EARTHQUAKE_LATITUDE = 36.75
EARTHQUAKE_LONGITUDE = 54.65
MAGNITUDE = 6.0
DEPTH_KM = 10.0
MECHANISM = "reverse"


# ============================================================
# GRID SETTINGS
# ============================================================

GRID_MIN_LON = 54.364163
GRID_MIN_LAT = 36.790849

GRID_MAX_LON = 54.492416
GRID_MAX_LAT = 36.867695

GRID_SPACING_M = 100


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

def load_database_config(config_path):
    """
    Load PostgreSQL/PostGIS configuration from YAML.
    """

    if not config_path.exists():
        raise FileNotFoundError(
            f"Database configuration file not found:\n{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Database configuration file must contain a YAML dictionary."
        )

    if "database" not in config:
        raise KeyError(
            "Missing 'database' section in database_config.yaml"
        )

    if "schema" not in config:
        raise KeyError(
            "Missing 'schema' section in database_config.yaml"
        )

    required_database_keys = [
        "host",
        "port",
        "name",
        "user",
        "password",
    ]

    for key in required_database_keys:
        if key not in config["database"]:
            raise KeyError(
                f"Missing database configuration key: database.{key}"
            )

    required_schema_keys = [
        "name",
        "buildings_table",
    ]

    for key in required_schema_keys:
        if key not in config["schema"]:
            raise KeyError(
                f"Missing schema configuration key: schema.{key}"
            )

    return config


# ============================================================
# LOAD BUILDINGS FROM POSTGIS
# ============================================================

def load_buildings_from_database(config):
    """
    Load building polygons from PostgreSQL/PostGIS.

    Database column names are preserved exactly as stored in DB.
    """

    db_config = config["database"]
    schema_config = config["schema"]

    host = db_config["host"]
    port = db_config["port"]
    database = db_config["name"]
    user = db_config["user"]
    password = db_config["password"]

    schema_name = schema_config["name"]
    table_name = schema_config["buildings_table"]

    connection_string = (
        f"postgresql+psycopg://"
        f"{user}:{password}@{host}:{port}/{database}"
    )

    sql = f'''
        SELECT
            id_building,
            source_fid,
            source_dataset,
            type_name,
            type_code,
            year,
            floors,
            landuse,
            roof_type,
            facade_type,
            noorgir,
            description,
            area_source,
            jica_class,
            geom
        FROM "{schema_name}"."{table_name}"
    '''

    print("=" * 70)
    print("LOADING BUILDINGS FROM POSTGIS")
    print("=" * 70)

    print(f"Database : {database}")
    print(f"Schema   : {schema_name}")
    print(f"Table    : {table_name}")

    buildings = gpd.read_postgis(
        sql,
        connection_string,
        geom_col="geom",
    )

    print(f"Number of buildings: {len(buildings):,}")
    print(f"CRS: {buildings.crs}")

    if buildings.empty:
        raise RuntimeError(
            "No building records were returned from the database."
        )

    return buildings


# ============================================================
# PREPARE BUILDING INPUTS
# ============================================================

def prepare_building_inputs(buildings):
    """
    Prepare internal fields required by Function 2.

    Original database columns are preserved.
    """

    buildings = buildings.copy()

    # Function 2 expects a generic 'type' field.
    # The original database field 'type_code' remains untouched.
    buildings["type"] = buildings["type_code"]

    return buildings


# ============================================================
# PRINT BUILDING SUMMARY
# ============================================================

def print_building_summary(result):
    """
    Print final Function 2 summary.
    """

    print("\n" + "=" * 70)
    print("BUILDING DAMAGE ASSESSMENT SUMMARY")
    print("=" * 70)

    print(f"Total buildings       : {len(result):,}")

    print("\n--- VS30 ---")

    if "vs30" in result.columns:
        vs30 = result["vs30"].dropna()

        print(f"Count                 : {len(vs30):,}")

        if len(vs30) > 0:
            print(f"Mean                  : {vs30.mean():.6f}")
            print(f"Std                   : {vs30.std():.6f}")
            print(f"Min                   : {vs30.min():.6f}")
            print(f"Max                   : {vs30.max():.6f}")

    print("\n--- PGA ---")

    if "pga_g" in result.columns:
        pga = result["pga_g"].dropna()

        print(f"Count                 : {len(pga):,}")

        if len(pga) > 0:
            print(f"Mean                  : {pga.mean():.6f} g")
            print(f"Std                   : {pga.std():.6f}")
            print(f"Min                   : {pga.min():.6f}")
            print(f"Max                   : {pga.max():.6f}")

    print("\n--- MMI ---")

    if "mmi" in result.columns:
        mmi = result["mmi"].dropna()

        print(f"Count                 : {len(mmi):,}")

        if len(mmi) > 0:
            print(f"Mean                  : {mmi.mean():.6f}")
            print(f"Std                   : {mmi.std():.6f}")
            print(f"Min                   : {mmi.min():.6f}")
            print(f"Max                   : {mmi.max():.6f}")

    print("\n--- BUILDING CLASS ---")

    if "building_class" in result.columns:
        print(
            result["building_class"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    print("\n--- DAMAGE RATIO ---")

    if "damage_ratio" in result.columns:

        damage = result["damage_ratio"].dropna()

        print(f"Valid values          : {len(damage):,}")
        print(
            f"Missing values        : "
            f"{result['damage_ratio'].isna().sum():,}"
        )

        if len(damage) > 0:
            print(f"Mean                  : {damage.mean():.6f}%")
            print(f"Std                   : {damage.std():.6f}")
            print(f"Min                   : {damage.min():.6f}%")
            print(f"Max                   : {damage.max():.6f}%")

    print("\n--- DAMAGE STATUS ---")

    if "damage_status" in result.columns:
        print(
            result["damage_status"]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# EXPORT BUILDINGS
# ============================================================

def export_building_result(final_result):
    """
    Export final building damage assessment as an ESRI Shapefile.

    Shapefile field names are limited to 10 characters, so selected
    attribute names are shortened only in the exported file.
    """

    final_result = final_result.copy()

    # --------------------------------------------------------
    # Geometry preparation
    # --------------------------------------------------------

    if "geom" in final_result.columns:
        final_result = final_result.set_geometry("geom")

    if (
        "geometry" in final_result.columns
        and "geom" in final_result.columns
    ):
        final_result = final_result.drop(columns=["geometry"])

    if "geom" in final_result.columns:
        final_result = final_result.rename_geometry("geometry")

    # --------------------------------------------------------
    # Geometry diagnostics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GEOMETRY CHECK BEFORE SHAPEFILE EXPORT")
    print("=" * 70)

    print(f"Active geometry : {final_result.geometry.name}")
    print(f"CRS             : {final_result.crs}")

    print("\nGeometry types:")
    print(
        final_result.geometry
        .geom_type
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        f"\nNull geometries  : "
        f"{final_result.geometry.isna().sum():,}"
    )

    print(
        f"Empty geometries : "
        f"{final_result.geometry.is_empty.sum():,}"
    )

    print(
        f"Invalid geometries: "
        f"{(~final_result.geometry.is_valid).sum():,}"
    )

    # --------------------------------------------------------
    # Prepare Shapefile attributes
    # --------------------------------------------------------

    shapefile_result = final_result.copy()

    # Shapefile field names must be <= 10 characters.
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
        "damage_status": "dmg_stat",
    }

    # Rename only fields that actually exist.
    field_mapping = {
        old: new
        for old, new in field_mapping.items()
        if old in shapefile_result.columns
    }

    shapefile_result = shapefile_result.rename(
        columns=field_mapping
    )

    # --------------------------------------------------------
    # Remove unsupported / unnecessary columns
    # --------------------------------------------------------

    # Keep only one geometry column.
    if "geometry" not in shapefile_result.columns:
        raise RuntimeError(
            "Geometry column not found before Shapefile export."
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

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
    print(
        [
            col
            for col in shapefile_result.columns
            if col != "geometry"
        ]
    )

    return shapefile_result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("EARTHQUAKE DAMAGE ASSESSMENT WORKFLOW")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATABASE CONFIGURATION
    # ========================================================

    config = load_database_config(
        DATABASE_CONFIG_FILE
    )

    # ========================================================
    # 2. LOAD BUILDINGS
    # ========================================================

    buildings = load_buildings_from_database(
        config
    )

    # ========================================================
    # 3. PREPARE BUILDING INPUTS
    # ========================================================

    buildings = prepare_building_inputs(
        buildings
    )

    print("\nBuilding columns:")
    print(buildings.columns.tolist())

    # ========================================================
    # 4. GRID-BASED EARTHQUAKE ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FUNCTION 1 - GRID EARTHQUAKE ANALYSIS")
    print("=" * 70)

    grid = create_grid(
        min_lon=GRID_MIN_LON,
        min_lat=GRID_MIN_LAT,
        max_lon=GRID_MAX_LON,
        max_lat=GRID_MAX_LAT,
        spacing_m=GRID_SPACING_M,
    )

    print(f"Grid points: {len(grid):,}")

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

    grid_result = run_earthquake_scenario(
        grid=grid,
        magnitude=MAGNITUDE,
        depth_km=DEPTH_KM,
        vs30_raster=VS30_RASTER,
        mechanism=MECHANISM,
        calculate_mmi=True,
    )

    # --------------------------------------------------------
    # Save grid CSV
    # --------------------------------------------------------

    grid_result.to_csv(
        GRID_CSV,
        index=False,
    )

    print(f"\nGrid CSV saved to:")
    print(GRID_CSV)

    # --------------------------------------------------------
    # Save grid GeoJSON
    # --------------------------------------------------------

    grid_result.to_file(
        GRID_GEOJSON,
        driver="GeoJSON",
    )

    print("Grid GeoJSON saved to:")
    print(GRID_GEOJSON)

    # ========================================================
    # 5. FUNCTION 1 - BUILDING SEISMIC PREPROCESSING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FUNCTION 1 - BUILDING SEISMIC PREPROCESSING")
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

    print("\nFunction 1 completed successfully.")

    # ========================================================
    # 6. FUNCTION 2 - BUILDING DAMAGE ASSESSMENT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FUNCTION 2 - BUILDING DAMAGE ASSESSMENT")
    print("=" * 70)

    final_result = run_building_damage_assessment(
        seismic_result
    )

    print("\nFunction 2 completed successfully.")

    # ========================================================
    # 7. FINAL SUMMARY
    # ========================================================

    print_building_summary(
        final_result
    )

    # ========================================================
    # 8. EXPORT FINAL BUILDING RESULT
    # ========================================================

    final_result = export_building_result(
        final_result
    )

    # ========================================================
    # 9. FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nBuildings processed : {len(final_result):,}")
    print(f"Grid points         : {len(grid_result):,}")

    print("\nOutputs:")
    print(f"  Grid CSV       : {GRID_CSV}")
    print(f"  Grid GeoJSON   : {GRID_GEOJSON}")
    print(f"  Building SHP   : {BUILDING_SHP}")

    print("\n")

    print("\n" + "=" * 70)
    print("FINAL GEODATAFRAME")
    print("=" * 70)

    print(final_result.head())
    print("\nColumns:")
    print(final_result.columns.tolist())

    print("\nGeoDataFrame info:")
    print(final_result.info())

    print("\nCRS:")
    print(final_result.crs)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()