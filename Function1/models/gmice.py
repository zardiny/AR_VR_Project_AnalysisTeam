import math


def ahmadzadeh_refined(pga_g, magnitude, distance_km):
    """
    Ahmadzadeh refined GMICE.

    PGA is converted from g to cm/s^2 before applying
    the published equation if required by the reference.
    """

    pga_cm_s2 = 981.0 * pga_g

    c0 = 2.182
    c1 = 1.305
    c2 = 0.575
    c3 = -1.597
    c4 = 0.010
    R0 = 15.0

    mmi = (
        c0
        + c1 * math.log10(pga_cm_s2)
        + c2 * magnitude
        + c3 * math.log10(distance_km + R0)
        + c4 * distance_km
    )

    return mmi


def ahmadzadeh_simple(pga_g):
    """
    Ahmadzadeh simple GMICE.
    """

    pga_cm_s2 = 981.0 * pga_g

    c0 = -0.58
    c1 = 3.47

    mmi = (
        c0
        + c1 * math.log10(pga_cm_s2)
    )

    return mmi


def zare(pga_g):
    """
    Zare GMICE.
    """

    pga_cm_s2 = 981.0 * pga_g

    c0 = -1.726
    c1 = 4.07

    mmi = (
        c0
        + c1 * math.log10(pga_cm_s2)
    )

    return mmi


def worden_global(pga_g):
    """
    Worden et al. (2012) global GMICE.

    PGA input to the published relation is in cm/s^2.
    """

    pga_cm_s2 = 981.0 * pga_g

    log_pga = math.log10(pga_cm_s2)

    if log_pga <= 1.57:

        mmi = (
            1.78
            + 1.55 * log_pga
        )

    else:

        mmi = (
            -1.60
            + 3.70 * log_pga
        )

    return mmi

def select_gmice(magnitude, distance_km):
    """
    Select the appropriate GMICE according to the
    decision logic defined in the project.

    Returns
    -------
    str
        GMICE model name.
    """

    # Ahmadzadeh refined
    if (
        5.1 <= magnitude <= 7.3
        and 5.0 <= distance_km <= 153.0
    ):
        return "Ahmadzadeh_refined"

    # Ahmadzadeh simple
    elif (
        5.1 <= magnitude <= 7.3
    ):
        return "Ahmadzadeh_simple"

    # Zare
    elif (
        4.6 <= magnitude < 5.1
        or 7.3 < magnitude <= 7.8
    ):
        return "Zare"

    # Global
    else:
        return "Worden2012_global"

def calculate_mmi(
    pga_g,
    magnitude,
    distance_km
):
    """
    Select GMICE and calculate MMI.
    """

    model = select_gmice(
        magnitude=magnitude,
        distance_km=distance_km
    )

    if model == "Ahmadzadeh_refined":

        mmi = ahmadzadeh_refined(
            pga_g=pga_g,
            magnitude=magnitude,
            distance_km=distance_km
        )

    elif model == "Ahmadzadeh_simple":

        mmi = ahmadzadeh_simple(
            pga_g=pga_g
        )

    elif model == "Zare":

        mmi = zare(
            pga_g=pga_g
        )

    elif model == "Worden2012_global":

        mmi = worden_global(
            pga_g=pga_g
        )

    else:

        raise ValueError(
            f"Unknown GMICE model: {model}"
        )

    return mmi, model