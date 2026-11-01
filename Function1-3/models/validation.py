def check_darzi_validity(
    magnitude,
    distance_km
):
    """
    Check Darzi calibration range.
    """

    magnitude_valid = (
        4.5 <= magnitude <= 7.4
    )

    distance_valid = (
        distance_km <= 200
    )

    if not magnitude_valid and not distance_valid:

        reason = "magnitude_and_distance"

    elif not magnitude_valid:

        reason = "magnitude"

    elif not distance_valid:

        reason = "distance"

    else:

        reason = None

    return {
        "magnitude_valid": magnitude_valid,
        "distance_valid": distance_valid,
        "extrapolated": reason is not None,
        "extrapolation_reason": reason
    }