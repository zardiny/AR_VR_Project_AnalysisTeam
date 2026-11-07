from __future__ import annotations

import numpy as np
import pandas as pd


# Key: (Firuzi building type, damage state); value: (NHS1 triple, NHS2 triple, NHS3 triple)
CASUALTY_RATES = {
    ('ST-LR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-LR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-LR-LQ', 'Extensive'): ((20.0, 30.0, 40.0), (0.0, 5.0, 10.0), (5.0, 7.5, 10.0)),
    ('ST-LR-LQ', 'Collapse'): ((30.0, 40.0, 50.0), (25.0, 35.0, 40.0), (15.0, 20.0, 25.0)),
    ('ST-LR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-LR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-LR-MQ', 'Extensive'): ((15.0, 25.0, 35.0), (0.0, 2.5, 5.0), (2.5, 5.0, 7.5)),
    ('ST-LR-MQ', 'Collapse'): ((25.0, 35.0, 45.0), (20.0, 30.0, 35.0), (10.0, 15.0, 20.0)),
    ('ST-LR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-LR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-LR-HQ', 'Extensive'): ((10.0, 20.0, 30.0), (0.0, 2.5, 5.0), (0.0, 2.5, 5.0)),
    ('ST-LR-HQ', 'Collapse'): ((20.0, 30.0, 40.0), (15.0, 20.0, 25.0), (5.0, 10.0, 15.0)),
    ('ST-MR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-MR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-MR-LQ', 'Extensive'): ((25.0, 35.0, 45.0), (0.0, 5.0, 10.0), (5.0, 7.5, 10.0)),
    ('ST-MR-LQ', 'Collapse'): ((35.0, 45.0, 55.0), (25.0, 30.0, 35.0), (20.0, 25.0, 30.0)),
    ('ST-MR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-MR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-MR-MQ', 'Extensive'): ((20.0, 30.0, 40.0), (0.0, 2.5, 5.0), (2.5, 5.0, 7.5)),
    ('ST-MR-MQ', 'Collapse'): ((30.0, 40.0, 50.0), (20.0, 25.0, 30.0), (15.0, 20.0, 25.0)),
    ('ST-MR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-MR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-MR-HQ', 'Extensive'): ((15.0, 25.0, 35.0), (0.0, 2.5, 5.0), (0.0, 2.5, 5.0)),
    ('ST-MR-HQ', 'Collapse'): ((25.0, 35.0, 45.0), (15.0, 20.0, 25.0), (10.0, 15.0, 20.0)),
    ('ST-HR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-HR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-HR-LQ', 'Extensive'): ((30.0, 40.0, 50.0), (0.0, 5.0, 10.0), (7.5, 10.0, 12.5)),
    ('ST-HR-LQ', 'Collapse'): ((40.0, 50.0, 60.0), (30.0, 35.0, 40.0), (25.0, 30.0, 35.0)),
    ('ST-HR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-HR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-HR-MQ', 'Extensive'): ((25.0, 35.0, 45.0), (0.0, 2.5, 5.0), (5.0, 7.5, 10.0)),
    ('ST-HR-MQ', 'Collapse'): ((35.0, 45.0, 45.0), (25.0, 30.0, 35.0), (20.0, 25.0, 30.0)),
    ('ST-HR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('ST-HR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('ST-HR-HQ', 'Extensive'): ((20.0, 30.0, 40.0), (0.0, 2.5, 5.0), (2.5, 5.0, 7.5)),
    ('ST-HR-HQ', 'Collapse'): ((30.0, 40.0, 50.0), (20.0, 30.0, 35.0), (15.0, 20.0, 25.0)),
    ('RC-LR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-LR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-LR-LQ', 'Extensive'): ((25.0, 35.0, 45.0), (0.0, 5.0, 10.0), (2.5, 7.5, 12.5)),
    ('RC-LR-LQ', 'Collapse'): ((45.0, 55.0, 65.0), (35.0, 45.0, 55.0), (20.0, 25.0, 30.0)),
    ('RC-LR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-LR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-LR-MQ', 'Extensive'): ((20.0, 30.0, 40.0), (0.0, 2.5, 5.0), (0.0, 5.0, 10.0)),
    ('RC-LR-MQ', 'Collapse'): ((40.0, 50.0, 60.0), (30.0, 35.0, 40.0), (15.0, 20.0, 25.0)),
    ('RC-LR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-LR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-LR-HQ', 'Extensive'): ((15.0, 25.0, 35.0), (0.0, 2.5, 5.0), (0.0, 2.5, 5.0)),
    ('RC-LR-HQ', 'Collapse'): ((35.0, 45.0, 45.0), (25.0, 30.0, 35.0), (10.0, 15.0, 20.0)),
    ('RC-MR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-MR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-MR-LQ', 'Extensive'): ((30.0, 40.0, 50.0), (0.0, 5.0, 10.0), (2.5, 7.5, 12.5)),
    ('RC-MR-LQ', 'Collapse'): ((50.0, 65.0, 80.0), (40.0, 45.0, 50.0), (25.0, 30.0, 35.0)),
    ('RC-MR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-MR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-MR-MQ', 'Extensive'): ((25.0, 35.0, 45.0), (0.0, 2.5, 5.0), (0.0, 5.0, 10.0)),
    ('RC-MR-MQ', 'Collapse'): ((45.0, 55.0, 65.0), (35.0, 45.0, 55.0), (20.0, 25.0, 30.0)),
    ('RC-MR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-MR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-MR-HQ', 'Extensive'): ((20.0, 30.0, 40.0), (0.0, 2.5, 5.0), (0.0, 2.5, 5.0)),
    ('RC-MR-HQ', 'Collapse'): ((40.0, 50.0, 60.0), (30.0, 35.0, 40.0), (15.0, 20.0, 25.0)),
    ('RC-HR-LQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-HR-LQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-HR-LQ', 'Extensive'): ((35.0, 45.0, 55.0), (0.0, 5.0, 10.0), (5.0, 10.0, 15.0)),
    ('RC-HR-LQ', 'Collapse'): ((55.0, 70.0, 85.0), (45.0, 50.0, 55.0), (30.0, 35.0, 40.0)),
    ('RC-HR-MQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-HR-MQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-HR-MQ', 'Extensive'): ((30.0, 40.0, 50.0), (0.0, 2.5, 5.0), (2.5, 7.5, 12.5)),
    ('RC-HR-MQ', 'Collapse'): ((50.0, 65.0, 80.0), (40.0, 45.0, 50.0), (25.0, 30.0, 35.0)),
    ('RC-HR-HQ', 'Slight'): ((0.0, 2.5, 5.0), (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
    ('RC-HR-HQ', 'Moderate'): ((0.0, 5.0, 10.0), (0.0, 2.5, 5.0), (0.0, 0.0, 0.0)),
    ('RC-HR-HQ', 'Extensive'): ((25.0, 35.0, 45.0), (0.0, 5.0, 10.0), (0.0, 5.0, 10.0)),
    ('RC-HR-HQ', 'Collapse'): ((45.0, 55.0, 65.0), (35.0, 45.0, 55.0), (20.0, 25.0, 30.0)),
    ('M', 'Slight'): ((5.0, 10.0, 15.0), (0.0, 5.0, 10.0), (0.0, 0.0, 0.0)),
    ('M', 'Moderate'): ((10.0, 15.0, 20.0), (5.0, 10.0, 15.0), (0.0, 5.0, 10.0)),
    ('M', 'Extensive'): ((50.0, 70.0, 90.0), (10.0, 20.0, 30.0), (0.0, 10.0, 20.0)),
    ('M', 'Collapse'): ((70.0, 85.0, 100.0), (25.0, 40.0, 55.0), (20.0, 35.0, 50.0)),
}

PERIODS = {
    "day": ("b_resday", "b_nonday", "res_day", "non_day", "out_day"),
    "night": ("b_resnig", "b_nonnig", "res_nig", "non_nig", "out_nig"),
    "transit": ("b_restrn", "b_nontrn", "res_trn", "non_trn", "out_trn"),
}
SEVERITIES = ("NHS1", "NHS2", "NHS3")

PERIODS = {
    "day": ("b_resday", "b_nonday", "res_day", "non_day", "out_day"),
    "night": ("b_resnig", "b_nonnig", "res_nig", "non_nig", "out_nig"),
    "transit": ("b_restrn", "b_nontrn", "res_trn", "non_trn", "out_trn"),
}

SEVERITIES = ("NHS1", "NHS2", "NHS3")


def classify_population_period(earthquake_time):
    """
    Classify local earthquake time into PAGER population scenario.

    Day:     10:00 <= time < 17:00
    Night:   22:00 <= time < 24:00 or 00:00 <= time < 05:00
    Transit: all remaining times.
    """
    from datetime import datetime, time as time_type

    if isinstance(earthquake_time, datetime):
        event_time = earthquake_time.time()
    elif isinstance(earthquake_time, time_type):
        event_time = earthquake_time
    elif isinstance(earthquake_time, str):
        try:
            event_time = datetime.strptime(
                earthquake_time.strip(), "%H:%M"
            ).time()
        except ValueError as exc:
            raise ValueError(
                "Earthquake time must use 24-hour HH:MM format, e.g. 14:30."
            ) from exc
    else:
        raise TypeError(
            "earthquake_time must be a string in HH:MM format or a time object."
        )

    minutes = event_time.hour * 60 + event_time.minute

    if 10 * 60 <= minutes < 17 * 60:
        return "day"
    if minutes >= 22 * 60 or minutes < 5 * 60:
        return "night"
    return "transit"


def norm_text(value):
    """Normalize damage-state text without changing its meaning."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def detect_damage_state(df, requested=None):
    """
    Find and normalize a Firuzi-compatible damage-state field.

    Damage Ratio is deliberately never used to infer Damage State.
    """
    candidates = [requested] if requested else []
    candidates += ["damage_state", "damage_st", "dmg_state", "damage_status"]

    field = next((name for name in candidates if name and name in df.columns), None)
    if field is None:
        raise ValueError(
            "No Firuzi-compatible Damage State field was found. "
            "Expected one of: damage_state, damage_st, dmg_state, damage_status."
        )

    mapping = {
        "slight": "Slight",
        "minor": "Slight",
        "moderate": "Moderate",
        "extensive": "Extensive",
        "collapse": "Collapse",
        "collapsed": "Collapse",
        "complete": "Collapse",
    }

    raw = df[field].map(norm_text)
    normalized = raw.map(mapping)

    invalid = raw.notna() & normalized.isna()
    if invalid.any():
        examples = (
            df.loc[invalid, field]
            .astype(str)
            .value_counts()
            .head(8)
            .to_dict()
        )
        raise ValueError(
            f"Unsupported values in {field}; expected "
            f"Slight/Moderate/Extensive/Collapse. Examples: {examples}"
        )

    return field, normalized


def derive_firuzi_type_vectorized(buildings):
    """Derive Firuzi type from material code, number of floors, and year."""
    code = pd.to_numeric(
        buildings.get("type_code", pd.Series(np.nan, index=buildings.index)),
        errors="coerce",
    )

    if "type" in buildings.columns:
        fallback = pd.to_numeric(buildings["type"], errors="coerce")
        code = code.fillna(fallback)

    floors = pd.to_numeric(
        buildings.get("floors", pd.Series(np.nan, index=buildings.index)),
        errors="coerce",
    )
    year = pd.to_numeric(
        buildings.get("year", pd.Series(np.nan, index=buildings.index)),
        errors="coerce",
    )

    material = pd.Series(index=buildings.index, dtype="object")
    material.loc[code.isin([1, 2])] = "M"
    material.loc[code.eq(3)] = "RC"
    material.loc[code.eq(4)] = "ST"

    height = pd.Series(index=buildings.index, dtype="object")
    height.loc[floors.gt(0) & floors.le(3)] = "LR"
    height.loc[floors.gt(3) & floors.le(7)] = "MR"
    height.loc[floors.gt(7)] = "HR"

    quality = pd.Series(index=buildings.index, dtype="object")
    quality.loc[year.le(1364)] = "LQ"
    quality.loc[year.gt(1364) & year.le(1385)] = "MQ"
    quality.loc[year.gt(1385)] = "HQ"

    valid = material.notna() & height.notna() & quality.notna()
    result = pd.Series(None, index=buildings.index, dtype="object")

    # The finalized casualty-rate table uses one generic masonry class.
    masonry = valid & material.eq("M")
    result.loc[masonry] = "M"

    other = valid & material.isin(["RC", "ST"])
    result.loc[other] = (
        material.loc[other] + "-" + height.loc[other] + "-" + quality.loc[other]
    )

    return result


def deterministic_rate_table():
    """Convert the three-point casualty-rate ranges to mean proportions."""
    table = {}

    for key, severity_triples in CASUALTY_RATES.items():
        rates = np.asarray(severity_triples, dtype=float).mean(axis=1) / 100.0

        total_rate = rates.sum()
        if total_rate > 1.0:
            rates = rates / total_rate

        table[key] = rates

    return table


def round_severity_counts(values, population_capacity):
    """
    Round three severity estimates to whole people without exceeding capacity.

    Rows containing non-finite estimates or capacity remain NaN.
    Largest-remainder allocation preserves the rounded row total.
    """
    values = np.asarray(values, dtype=float)
    capacity = np.asarray(population_capacity, dtype=float)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("values must have shape (n, 3).")

    if capacity.ndim != 1 or len(capacity) != len(values):
        raise ValueError("population_capacity must have shape (n,).")

    output = np.full(values.shape, np.nan, dtype=float)

    valid = np.isfinite(values).all(axis=1) & np.isfinite(capacity)
    if not valid.any():
        return output

    x = np.maximum(values[valid], 0.0)
    cap = np.floor(np.maximum(capacity[valid], 0.0) + 1e-9).astype(np.int64)

    sums = x.sum(axis=1)
    over = sums > cap
    nonzero_over = over & (sums > 0)
    x[nonzero_over] *= (
        cap[nonzero_over] / sums[nonzero_over]
    )[:, None]

    sums = x.sum(axis=1)
    base = np.floor(x).astype(np.int64)
    target = np.minimum(
        np.floor(sums + 0.5).astype(np.int64),
        cap,
    )

    remaining = np.maximum(target - base.sum(axis=1), 0)
    fractions = x - base

    for i, count in enumerate(remaining):
        count = int(count)
        if count:
            order = np.argsort(-fractions[i], kind="stable")
            for j in order[:min(count, x.shape[1])]:
                base[i, j] += 1

    output[valid] = base.astype(float)
    return output


def add_population_periods(buildings, selected_period=None):
    """Create total building population for the selected temporal scenario."""
    buildings = buildings.copy()
    periods = (
        {selected_period: PERIODS[selected_period]}
        if selected_period is not None
        else PERIODS
    )

    for period, (res_col, non_col, *_rest) in periods.items():
        required = (res_col, non_col)
        missing = [column for column in required if column not in buildings.columns]

        if missing:
            raise ValueError(
                f"Building population input is missing required fields: {missing}"
            )

        for column in required:
            values = pd.to_numeric(buildings[column], errors="coerce")

            if values.isna().any():
                raise ValueError(
                    f"Building population field '{column}' contains null or "
                    "non-numeric values."
                )

            if (values < 0).any():
                raise ValueError(
                    f"Building population field '{column}' contains negative values."
                )

            buildings[column] = values.astype(float)

        buildings[f"pop_{period}"] = (
            buildings[res_col] + buildings[non_col]
        )

    return buildings


def estimate_casualties_vectorized(
    buildings, blocks, block_id_col="block_id", selected_period=None
):
    """
    Estimate deterministic casualties at building and block levels.

    Building-level estimates remain unrounded expected counts. They are
    aggregated using these exact expectations, then block estimates are
    rounded once. Blocks without assigned buildings use the global
    population-weighted casualty-rate fallback.
    """
    periods = (
        {selected_period: PERIODS[selected_period]}
        if selected_period is not None
        else PERIODS
    )
    if buildings.empty:
        raise ValueError("No buildings were supplied to casualty estimation.")

    if blocks.empty:
        raise ValueError("No blocks were supplied to casualty estimation.")

    required_building_fields = [
        block_id_col,
        "firuzi_type",
        "damage_state",
        *[f"pop_{period}" for period in periods],
    ]
    missing_building = [
        column for column in required_building_fields
        if column not in buildings.columns
    ]
    if missing_building:
        raise ValueError(
            f"Building input is missing required fields: {missing_building}"
        )

    required_block_fields = [
        block_id_col,
        *[
            column
            for period_values in periods.values()
            for column in period_values[2:]
        ],
    ]
    missing_block = [
        column for column in required_block_fields
        if column not in blocks.columns
    ]
    if missing_block:
        raise ValueError(
            f"Block input is missing required fields: {missing_block}"
        )

    block_ids_numeric = pd.to_numeric(blocks[block_id_col], errors="coerce")
    if block_ids_numeric.isna().any():
        raise ValueError("Block population input contains null/non-numeric block IDs.")

    if (block_ids_numeric % 1 != 0).any():
        raise ValueError("Block IDs must be integer-valued.")

    block_ids = block_ids_numeric.astype("int64")
    if block_ids.duplicated().any():
        duplicates = block_ids[block_ids.duplicated()].head(10).tolist()
        raise ValueError(f"Duplicate block IDs in block input: {duplicates}")

    # Buildings with a missing block_id are retained for building-level
    # casualty estimation, but cannot be assigned to a block-level total.
    # Likewise, a valid block_id that is absent from the block population
    # table is excluded from block aggregation rather than stopping the run.
    building_ids_numeric = pd.to_numeric(
        buildings[block_id_col], errors="coerce"
    )
    non_integer_ids = (
        building_ids_numeric.notna()
        & ((building_ids_numeric % 1) != 0)
    )
    if non_integer_ids.any():
        raise ValueError("Building block IDs must be integer-valued when provided.")

    block_id_to_pos = {
        int(block_id): position
        for position, block_id in enumerate(block_ids.to_numpy())
    }

    group_pos = np.full(len(buildings), -1, dtype=np.int64)
    has_block_id = building_ids_numeric.notna().to_numpy()
    if has_block_id.any():
        valid_building_ids = building_ids_numeric.loc[has_block_id].astype("int64")
        mapped_positions = valid_building_ids.map(block_id_to_pos)
        group_pos[has_block_id] = mapped_positions.fillna(-1).to_numpy(dtype=np.int64)

    # Only these buildings can contribute to a block aggregation.
    assigned = group_pos >= 0
    skipped_no_block = int((~has_block_id).sum())
    skipped_unmatched_block = int((has_block_id & ~assigned).sum())
    if skipped_no_block or skipped_unmatched_block:
        print(
            "Function 3: excluding buildings from block aggregation: "
            f"{skipped_no_block:,} with null/invalid block_id; "
            f"{skipped_unmatched_block:,} with block_id absent from block table. "
            "Building-level casualty estimates are still calculated for these rows."
        )
    periods = list(periods)
    severities = list(SEVERITIES)
    n_buildings = len(buildings)
    n_blocks = len(blocks)

    rate_table = deterministic_rate_table()

    pairs = list(zip(
        buildings["firuzi_type"].to_numpy(),
        buildings["damage_state"].to_numpy(),
    ))

    rate_lookup = np.full((n_buildings, len(severities)), np.nan, dtype=float)

    for i, pair in enumerate(pairs):
        rates = rate_table.get(pair)
        if rates is not None:
            rate_lookup[i, :] = rates

    valid_rate = np.isfinite(rate_lookup).all(axis=1)

    building_results = {}
    population_by_period = {}

    for period in periods:
        population = pd.to_numeric(
            buildings[f"pop_{period}"], errors="coerce"
        ).to_numpy(dtype=float)

        if not np.isfinite(population).all() or (population < 0).any():
            raise ValueError(
                f"Invalid building population values for period '{period}'."
            )

        population_by_period[period] = population

        estimates = population[:, None] * rate_lookup
        estimates[~valid_rate, :] = np.nan

        for severity_index, severity in enumerate(severities):
            building_results[f"{period}_{severity}_est"] = (
                estimates[:, severity_index]
            )

    # Population-weighted rates are used only for blocks without buildings.
    global_rates = {}

    for period in periods:
        population = population_by_period[period]
        eligible = valid_rate & (population > 0)
        denominator = float(population[eligible].sum())

        for severity_index, severity in enumerate(severities):
            numerator = float(
                np.sum(population[eligible] * rate_lookup[eligible, severity_index])
            )
            global_rates[(period, severity)] = (
                numerator / denominator if denominator > 0 else np.nan
            )

    # Preserve exact expected casualty counts at building level.
    # These unrounded values are also used for block-level aggregation.
    raw_building_results = {
        key: values.copy()
        for key, values in building_results.items()
    }

    building_counts = np.bincount(group_pos[assigned], minlength=n_blocks)
    valid_counts = np.bincount(
        group_pos[assigned],
        weights=valid_rate[assigned].astype(float),
        minlength=n_blocks,
    )

    block_status = np.full(n_blocks, "ok", dtype=object)
    block_status[(building_counts > 0) & (valid_counts == 0)] = (
        "missing_building_damage_or_rate"
    )
    block_status[
        (valid_counts > 0) & (valid_counts < building_counts)
    ] = "partial_missing_building_damage_or_rate"

    block_results = {}
    no_building = building_counts == 0
    no_valid_rate = (building_counts > 0) & (valid_counts == 0)
    block_status[no_building] = "weighted_mean_fallback"

    for period in periods:
        for severity_index, severity in enumerate(severities):
            raw_estimates = np.nan_to_num(
                raw_building_results[f"{period}_{severity}_est"],
                nan=0.0,
            )

            totals = np.bincount(
                group_pos[assigned],
                weights=raw_estimates[assigned],
                minlength=n_blocks,
            )

            _, _, res_col, non_col, out_col = PERIODS[period]
            population_columns = [res_col, non_col, out_col]

            block_population = (
                blocks[population_columns]
                .apply(pd.to_numeric, errors="coerce")
            )

            if block_population.isna().any().any():
                raise ValueError(
                    f"Block population contains null/non-numeric values "
                    f"for period '{period}'."
                )

            if (block_population < 0).any().any():
                raise ValueError(
                    f"Block population contains negative values for period '{period}'."
                )

            block_population_total = (
                block_population.sum(axis=1).to_numpy(dtype=float)
            )

            fallback_rate = global_rates[(period, severity)]

            if np.isfinite(fallback_rate):
                totals[no_building] = (
                    block_population_total[no_building] * fallback_rate
                )
            else:
                totals[no_building] = np.nan

            # Do not turn blocks with no valid building rates into false zeroes.
            totals[no_valid_rate] = np.nan

            block_results[f"{period}_{severity}_est"] = totals

    # Round block-level estimates once and enforce each block's population cap.
    for period in periods:
        _, _, res_col, non_col, out_col = PERIODS[period]
        block_population = blocks[[res_col, non_col, out_col]].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1).to_numpy(dtype=float)

        keys = [
            f"{period}_{severity}_est"
            for severity in severities
        ]
        values = np.column_stack([block_results[key] for key in keys])
        rounded = round_severity_counts(values, block_population)

        for severity_index, key in enumerate(keys):
            block_results[key] = rounded[:, severity_index]

    return building_results, block_results, valid_rate, block_status
