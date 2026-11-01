from __future__ import annotations
import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2 import sql
import yaml

def load_database_config(config_path):
    """Load and validate PostgreSQL/PostGIS configuration."""

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
        raise KeyError("Missing 'database' section in database_config.yaml")

    if "schema" not in config:
        raise KeyError("Missing 'schema' section in database_config.yaml")

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
                f"Missing database configuration key: schema.{key}"
            )

    return config

def get_connection(config):
    db = config["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["name"],
        user=db["user"],
        password=db["password"],
    )

def load_buildings_from_database(config):
    """Load building polygons from PostgreSQL/PostGIS."""

    db_config = config["database"]
    schema_config = config["schema"]

    schema_name = schema_config["name"]
    table_name = schema_config["buildings_table"]

    query = sql.SQL("""
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
            ST_AsBinary(geom) AS geom_wkb,
            ST_SRID(geom) AS geom_srid
        FROM {}.{}
    """).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
    )

    print("\n" + "=" * 70)
    print("LOADING BUILDINGS FROM POSTGIS")
    print("=" * 70)

    print(f"Database : {db_config['name']}")
    print(f"Schema   : {schema_name}")
    print(f"Table    : {table_name}")

    connection = None

    try:
        connection = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            dbname=db_config["name"],
            user=db_config["user"],
            password=db_config["password"],
        )

        with connection.cursor() as cursor:
            cursor.execute(query)

            column_names = [
                description[0]
                for description in cursor.description
            ]

            records = cursor.fetchall()

        wkb_index = column_names.index("geom_wkb")
        srid_index = column_names.index("geom_srid")

        attributes = [
            {
                name: record[index]
                for index, name in enumerate(column_names)
                if name not in ("geom_wkb", "geom_srid")
            }
            for record in records
        ]

        from shapely import wkb

        geometries = [
            wkb.loads(bytes(record[wkb_index]))
            if record[wkb_index] is not None
            else None
            for record in records
        ]

        srid = next(
            (
                record[srid_index]
                for record in records
                if record[srid_index]
            ),
            None,
        )

        crs = f"EPSG:{srid}" if srid else None

        buildings = gpd.GeoDataFrame(
            attributes,
            geometry=geometries,
            crs=crs,
        )

        buildings = buildings.rename_geometry("geom")

    except Exception as exc:
        raise RuntimeError(
            "Failed to load buildings from PostgreSQL/PostGIS. "
            f"Original error: {exc}"
        ) from exc

    finally:
        if connection is not None:
            connection.close()

    if buildings.empty:
        raise RuntimeError(
            "No building records were returned from the database."
        )

    print(f"Number of buildings: {len(buildings):,}")
    print(f"CRS: {buildings.crs}")

    return buildings


def load_function3_population(config, selected_period):
    """
    Read only the selected temporal population scenario from PostgreSQL.

    selected_period must be one of: day, night, transit.
    Source tables remain read-only.
    """
    if selected_period not in {"day", "night", "transit"}:
        raise ValueError(
            "selected_period must be one of: day, night, transit."
        )

    schema_config = config["schema"]
    schema_name = schema_config["name"]

    buildings_table = schema_config["buildings_table"]
    building_population_table = schema_config.get(
        "building_population_table",
        "building_population",
    )
    block_population_table = schema_config.get(
        "block_population_table",
        "block_population_temporal",
    )

    building_population_column = f"{selected_period}_population"
    block_res_column = f"{selected_period}_res_population"
    block_nonres_column = f"{selected_period}_nonres_population"
    block_outdoor_column = f"{selected_period}_outdoor_population"
    block_unallocated_column = f"{selected_period}_unallocated_population"

    building_sql = sql.SQL("""
        SELECT
            b.id_building,
            b.block_id,
            bp.{} AS selected_population,
            bp.allocation_status
        FROM {} AS b
        LEFT JOIN {} AS bp
            ON bp.id_building = b.id_building
        ORDER BY b.id_building
    """).format(
        sql.Identifier(building_population_column),
        sql.Identifier(schema_name, buildings_table),
        sql.Identifier(schema_name, building_population_table),
    )

    block_sql = sql.SQL("""
        SELECT
            block_id,
            COALESCE({}, 0) AS res_selected,
            COALESCE({}, 0) AS non_selected,
            COALESCE({}, 0) + COALESCE({}, 0) AS out_selected
        FROM {}
        ORDER BY block_id
    """).format(
        sql.Identifier(block_res_column),
        sql.Identifier(block_nonres_column),
        sql.Identifier(block_outdoor_column),
        sql.Identifier(block_unallocated_column),
        sql.Identifier(schema_name, block_population_table),
    )

    connection = get_connection(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(building_sql)
            building_columns = [
                description[0] for description in cursor.description
            ]
            building_rows = cursor.fetchall()

            cursor.execute(block_sql)
            block_columns = [
                description[0] for description in cursor.description
            ]
            block_rows = cursor.fetchall()

        buildings = pd.DataFrame(building_rows, columns=building_columns)
        blocks = pd.DataFrame(block_rows, columns=block_columns)
    finally:
        connection.close()

    if buildings.empty:
        raise RuntimeError(
            "No building records were returned from the database population query."
        )

    if blocks.empty:
        raise RuntimeError(
            "No block population records were returned from the database query."
        )

    if buildings["id_building"].isna().any():
        raise ValueError("Database population input contains null id_building values.")

    if blocks["block_id"].isna().any():
        raise ValueError("Temporal block population input contains null block_id values.")

    if buildings["id_building"].duplicated().any():
        raise ValueError(
            "Duplicate id_building values in database population input. "
            f"Check the uniqueness of {schema_name}.{building_population_table}.id_building."
        )

    if blocks["block_id"].duplicated().any():
        raise ValueError(
            "Duplicate block_id values in temporal block population input."
        )

    buildings["selected_population"] = pd.to_numeric(
        buildings["selected_population"], errors="coerce"
    )
    buildings["population_record_missing"] = (
        buildings["selected_population"].isna()
    )

    print("\\nFunction 3 - Database input summary")
    print(f"Selected population scenario: {selected_period}")
    print(f"Building rows: {len(buildings):,}")
    print(f"Block rows:    {len(blocks):,}")
    print(
        "Buildings without population values for selected scenario: "
        f"{buildings['population_record_missing'].sum():,}"
    )

    return buildings, blocks
