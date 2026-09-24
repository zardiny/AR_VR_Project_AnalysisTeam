from __future__ import annotations

from typing import Optional


# ============================================================
# Database building type codes
# ============================================================

TYPE_BRICK_STEEL = 1
TYPE_BRICK = 2
TYPE_CONCRETE = 3
TYPE_STEEL = 4
TYPE_COMPOSITE = 5
TYPE_UNKNOWN = 6


# ============================================================
# JICA building classification
# ============================================================

def classify_building(
    building_type: int,
    year: int,
    floors: float,
) -> Optional[int]:
    """
    Classify a building according to the JICA 9-class
    building classification.

    Parameters
    ----------
    building_type : int
        Building structural type from the database.

        1 = Brick & Steel
        2 = Brick
        3 = Concrete
        4 = Steel
        5 = Composite
        6 = Unknown

    year : int
        Construction year.

    floors : float
        Number of floors.

    Returns
    -------
    int or None
        JICA building class (1-9).

        None is returned when the building cannot be
        reliably classified, such as composite or
        unknown building types.

    Raises
    ------
    ValueError
        If required building attributes are invalid.
    """

    # 1. Validate building type

    if building_type is None:
        return None

    try:
        building_type = int(building_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid building type: {building_type!r}"
        ) from exc

    if building_type not in {
        TYPE_BRICK_STEEL,
        TYPE_BRICK,
        TYPE_CONCRETE,
        TYPE_STEEL,
        TYPE_COMPOSITE,
        TYPE_UNKNOWN,
    }:
        raise ValueError(
            f"Unknown building type code: {building_type}"
        )


    # 2. Unknown / composite buildings

    # These buildings are intentionally not classified because
    # assigning them to a JICA class would introduce additional
    # assumptions and uncertainty.

    if building_type in {
        TYPE_COMPOSITE,
        TYPE_UNKNOWN,
    }:
        return None

    # 3. Validate year

    if year is None:
        raise ValueError(
            "Construction year is missing."
        )

    try:
        year = int(year)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid construction year: {year!r}"
        ) from exc

    # 4. Validate floors

    if floors is None:
        raise ValueError(
            "Number of floors is missing."
        )

    try:
        floors = float(floors)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid number of floors: {floors!r}"
        ) from exc

    if floors <= 0:
        raise ValueError(
            f"Number of floors must be positive. "
            f"Received: {floors}"
        )


    # 5. Brick & Steel

    if building_type == TYPE_BRICK_STEEL:
        return 1


    # 6. Brick

    if building_type == TYPE_BRICK:
        return 8


    # 7. Steel buildings
    #
    # Class 2:
    # Steel-1
    # Built >= 1371 and 1-3 floors
    #
    # Class 3:
    # Steel-2
    # Built < 1370 OR 4+ floors

    if building_type == TYPE_STEEL:

        if year <= 1370 or floors >= 4:
            return 3

        if year >= 1371 and floors < 4:
            return 2

        raise ValueError(
            "Steel building falls outside the defined "
            "JICA classification boundaries."
        )


    # 8. Reinforced concrete buildings
    #
    # Class 4:
    # RC-0, 6+ floors
    #
    # Class 5:
    # RC-1, built >= 1370 and 1-2 floors
    #
    # Class 6:
    # RC-2, built < 1370 OR 3+ floors

    if building_type == TYPE_CONCRETE:

        if floors >= 6:
            return 4

        if year < 1370 or floors >= 3:
            return 6

        if year >= 1370 and floors in {1, 2}:
            return 5

        raise ValueError(
            "Concrete building falls outside the defined "
            "JICA classification boundaries."
        )

    raise ValueError(
        f"Building could not be classified: "
        f"type={building_type}, "
        f"year={year}, "
        f"floors={floors}"
    )