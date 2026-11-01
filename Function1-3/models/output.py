from pathlib import Path


def save_csv(result, output_path):
    """
    Save numerical results as CSV.
    """

    output_path = Path(output_path)

    df = result.copy()

    # Geometry is not needed in CSV
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")

    df.to_csv(
        output_path,
        index=False
    )


def save_geojson(result, output_path):
    """
    Save spatial results as GeoJSON.
    """

    output_path = Path(output_path)

    result.to_file(
        output_path,
        driver="GeoJSON"
    )
