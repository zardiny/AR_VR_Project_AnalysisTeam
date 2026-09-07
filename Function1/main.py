from pathlib import Path
from models.output import (
    save_csv,
    save_geojson
)

from models.pipeline import run_earthquake_scenario

from models.pipeline import run_earthquake_scenario
from models.grid import create_grid
from models.distance import (
    calculate_epicentral_distance,
    calculate_hypocentral_distance_vectorized
)

dir = Path(__file__).resolve().parent

#main_sample
earthquake_latitude = 36.75
earthquake_longitude = 54.65
depth_km = 15


grid = create_grid(
    min_lon=54.751251,
    min_lat=36.895583,
    max_lon=54.754489,
    max_lat=36.898588,
    spacing_m=100
)

grid["R_epi_km"] = calculate_epicentral_distance(
    site_latitudes=grid["latitude"].values,
    site_longitudes=grid["longitude"].values,
    earthquake_latitude=earthquake_latitude,
    earthquake_longitude=earthquake_longitude
)

grid["R_hyp_km"] = calculate_hypocentral_distance_vectorized(
    epicentral_distance_km=grid["R_epi_km"].values,
    depth_km=depth_km
)

# you can paste your desired directory here
VS30_RASTER = dir / "vs30_mosaic.tif"


result = run_earthquake_scenario(
    grid=grid,
    magnitude=6.0,
    depth_km=10.0,
    vs30_raster=VS30_RASTER,
    mechanism="reverse",
    calculate_mmi=True
)


# Current project directory
output_dir = Path(__file__).resolve().parent
print(output_dir)
save_csv(
    result,
    dir / r"export_sample\earthquake_result.csv"
)

save_geojson(
    result,
    dir / r"export_sample\earthquake_result.geojson"
)

print("Outputs saved successfully.")

