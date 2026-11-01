import math
import time
import tracemalloc

import pandas as pd

from models.pga_map import calculate_pga_for_grid
from models.gmice import calculate_mmi


# ============================================================
# PERFORMANCE TEST
# Earthquake Scenario → Grid → PGA → GMICE → MMI
# ============================================================


# ============================================================
# 1. Test Configuration
# ============================================================

EARTHQUAKE = {
    "magnitude": 6.0,
    "latitude": 36.70,
    "longitude": 54.60,
    "depth": 10.0,
    "mechanism": "reverse",
}


# Grid sizes = number of points
GRID_SIZES = [
    100,
    500,
    1_000,
    5_000,
    10_000,
]


# Number of sequential requests
REQUEST_COUNTS = [
    1,
    5,
    10,
    20,
]


# ============================================================
# 2. Create Synthetic Grid
# ============================================================

def create_test_grid(n_points):
    """
    Create a synthetic grid containing n_points.

    The grid is generated around the earthquake epicenter.
    """

    rows = []

    for i in range(n_points):

        lat_offset = (i % 100) * 0.001
        lon_offset = (i // 100) * 0.001

        latitude = EARTHQUAKE["latitude"] + lat_offset
        longitude = EARTHQUAKE["longitude"] + lon_offset

        site_classes = [1, 2, 3]
        site_class = site_classes[i % 3]

        rows.append({
            "latitude": latitude,
            "longitude": longitude,
            "site_class": site_class,
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. Calculate Hypocentral Distance
# ============================================================

from math import radians, sin, cos, sqrt, atan2


def calculate_hypocentral_distance(
    lat1,
    lon1,
    lat2,
    lon2,
    depth_km
):
    """
    Calculate hypocentral distance in km.
    """

    R_EARTH = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    epicentral_distance_km = R_EARTH * c

    return sqrt(
        epicentral_distance_km ** 2
        + depth_km ** 2
    )


# ============================================================
# 4. Prepare Grid
# ============================================================

def prepare_grid(n_points):
    """
    Create grid and calculate hypocentral distances.
    """

    grid = create_test_grid(n_points)

    grid["R_hyp_km"] = grid.apply(
        lambda row: calculate_hypocentral_distance(
            row["latitude"],
            row["longitude"],
            EARTHQUAKE["latitude"],
            EARTHQUAKE["longitude"],
            EARTHQUAKE["depth"],
        ),
        axis=1,
    )

    return grid


# ============================================================
# 5. Execute One Complete Request
# ============================================================

def execute_request(grid):
    """
    Execute the complete earthquake processing pipeline:

        Grid
          ↓
        PGA
          ↓
        PGA → cm/s²
          ↓
        MMI
    """

    # --------------------------------------------------------
    # PGA
    # --------------------------------------------------------

    result = calculate_pga_for_grid(
        grid=grid,
        magnitude=EARTHQUAKE["magnitude"],
        depth_km=EARTHQUAKE["depth"],
        mechanism=EARTHQUAKE["mechanism"],
    )

    # --------------------------------------------------------
    # PGA unit conversion
    # --------------------------------------------------------

    result["pga_cm_s2"] = result["pga_g"] * 981.0

    # --------------------------------------------------------
    # MMI
    # --------------------------------------------------------

    mmi_values = []

    for _, row in result.iterrows():

        mmi_result = calculate_mmi(
            pga_g=row["pga_g"],
            magnitude=EARTHQUAKE["magnitude"],
            distance_km=row["R_hyp_km"],
        )

        mmi_values.append(mmi_result[0])

    result["mmi"] = mmi_values

    return result


# ============================================================
# 6. Performance Test for One Grid Size
# ============================================================

def benchmark_grid_size(
    n_points,
    n_requests
):
    """
    Benchmark a given grid size and number of requests.
    """

    print(
        f"\nTesting "
        f"{n_points:,} points "
        f"× "
        f"{n_requests} requests"
    )

    # --------------------------------------------------------
    # Prepare grid
    # --------------------------------------------------------

    grid = prepare_grid(n_points)

    # --------------------------------------------------------
    # Warm-up request
    # --------------------------------------------------------

    execute_request(grid)

    # --------------------------------------------------------
    # Start memory tracking
    # --------------------------------------------------------

    tracemalloc.start()

    start_time = time.perf_counter()

    request_times = []

    for _ in range(n_requests):

        request_start = time.perf_counter()

        result = execute_request(grid)

        request_end = time.perf_counter()

        request_times.append(
            request_end - request_start
        )

    end_time = time.perf_counter()

    # --------------------------------------------------------
    # Memory statistics
    # --------------------------------------------------------

    current_memory, peak_memory = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_time = end_time - start_time

    mean_request_time = sum(request_times) / n_requests

    min_request_time = min(request_times)

    max_request_time = max(request_times)

    total_points = n_points * n_requests

    throughput = total_points / total_time

    # --------------------------------------------------------
    # Basic output validation
    # --------------------------------------------------------

    assert isinstance(
        result,
        pd.DataFrame
    ), "Result must be a DataFrame."

    assert len(result) == n_points

    assert "pga_g" in result.columns

    assert "mmi" in result.columns

    assert result["pga_g"].notna().all()

    assert result["mmi"].notna().all()

    assert result["pga_g"].apply(math.isfinite).all()

    assert result["mmi"].apply(math.isfinite).all()

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Total time       : {total_time:.4f} s"
    )

    print(
        f"Mean request     : {mean_request_time:.4f} s"
    )

    print(
        f"Min request      : {min_request_time:.4f} s"
    )

    print(
        f"Max request      : {max_request_time:.4f} s"
    )

    print(
        f"Throughput       : {throughput:.2f} points/s"
    )

    print(
        f"Peak memory      : "
        f"{peak_memory / (1024 ** 2):.2f} MB"
    )

    return {
        "grid_points": n_points,
        "requests": n_requests,
        "total_points": total_points,
        "total_time_s": total_time,
        "mean_request_time_s": mean_request_time,
        "min_request_time_s": min_request_time,
        "max_request_time_s": max_request_time,
        "throughput_points_per_s": throughput,
        "peak_memory_mb": peak_memory / (1024 ** 2),
    }


# ============================================================
# 7. Main Performance Experiment
# ============================================================

print("=" * 80)
print("EARTHQUAKE PGA → MMI PERFORMANCE TEST")
print("=" * 80)

print("\nEarthquake Scenario")
print("-" * 80)

print(
    f"Magnitude       : Mw = "
    f"{EARTHQUAKE['magnitude']}"
)

print(
    f"Latitude        : "
    f"{EARTHQUAKE['latitude']}"
)

print(
    f"Longitude       : "
    f"{EARTHQUAKE['longitude']}"
)

print(
    f"Depth           : "
    f"{EARTHQUAKE['depth']} km"
)

print(
    f"Mechanism       : "
    f"{EARTHQUAKE['mechanism']}"
)


# ============================================================
# 8. Run Benchmark
# ============================================================

results = []

for n_points in GRID_SIZES:

    for n_requests in REQUEST_COUNTS:

        benchmark_result = benchmark_grid_size(
            n_points=n_points,
            n_requests=n_requests,
        )

        results.append(
            benchmark_result
        )


# ============================================================
# 9. Results DataFrame
# ============================================================

performance_df = pd.DataFrame(results)


# ============================================================
# 10. Display Results
# ============================================================

print("\n")
print("=" * 80)
print("PERFORMANCE TEST RESULTS")
print("=" * 80)

print(
    performance_df.to_string(
        index=False,
        formatters={
            "total_time_s": "{:.4f}".format,
            "mean_request_time_s": "{:.4f}".format,
            "min_request_time_s": "{:.4f}".format,
            "max_request_time_s": "{:.4f}".format,
            "throughput_points_per_s": "{:.2f}".format,
            "peak_memory_mb": "{:.2f}".format,
        }
    )
)


# ============================================================
# 11. Save Results
# ============================================================

performance_df.to_csv(
    "performance_results.csv",
    index=False
)


print(
    "\nPerformance results saved to:"
)

print(
    "performance_results.csv"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE TEST COMPLETED")
print("=" * 80)