from __future__ import annotations

import pandas as pd

import models.casualties as model
from models.database_building_population import load_function3_population


def _prepare_function2_result(function2_result):
    """Normalize and retain only the Function 2 fields needed by Function 3."""
    result = function2_result.copy()

    aliases = {
        "id_building": "building_id",
        "damage_ratio": "dmg_ratio",
        "damage_status": "dmg_state",
    }
    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in result.columns and new not in result.columns
    }
    result = result.rename(columns=rename_map)

    required = [
        "building_id", "type_code", "year", "floors", "dmg_ratio", "dmg_state"
    ]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(
            "Function 2 output is missing required Function 3 fields: "
            f"{missing}. Available fields: {result.columns.tolist()}"
        )

    damage = result[required].copy()

    if damage["building_id"].isna().any():
        raise ValueError("Function 2 contains missing building IDs.")

    numeric_ids = pd.to_numeric(damage["building_id"], errors="coerce")
    if numeric_ids.isna().any() or (numeric_ids % 1 != 0).any():
        raise ValueError("Function 2 building IDs must be numeric integers.")

    damage["id_building"] = numeric_ids.astype("int64")
    if damage["id_building"].duplicated().any():
        duplicates = damage.loc[
            damage["id_building"].duplicated(), "id_building"
        ].head(10).tolist()
        raise ValueError(f"Function 2 contains duplicate building IDs: {duplicates}")

    for column in ("type_code", "year", "floors", "dmg_ratio"):
        damage[column] = pd.to_numeric(damage[column], errors="coerce")

    return damage


def run_function3(function2_result, config, earthquake_time):
    """
    Run deterministic casualty estimation for the population scenario selected
    by the earthquake occurrence time. Exactly one temporal scenario is used.
    """
    print("\n" + "=" * 70)
    print("FUNCTION 3 - CASUALTY ESTIMATION")
    print("=" * 70)

    selected_period = model.classify_population_period(earthquake_time)
    print(f"Earthquake occurrence time: {earthquake_time}")
    print(f"Selected population scenario: {selected_period.upper()}")

    # 1. Normalize Function 2 output.
    damage = _prepare_function2_result(function2_result)

    damage_field, damage_state = model.detect_damage_state(
        damage, requested="dmg_state"
    )
    damage["damage_state"] = damage_state

    print(f"Function 2 buildings: {len(damage):,}")
    print(f"Damage State field:   {damage_field}")
    print("\\nDamage state counts:")
    print(damage["damage_state"].value_counts(dropna=False).to_string())

    # 2. Load only the selected population scenario from PostgreSQL.
    population, blocks = load_function3_population(
        config, selected_period=selected_period
    )

    if population["population_record_missing"].any():
        missing_count = int(population["population_record_missing"].sum())
        raise RuntimeError(
            f"{missing_count:,} buildings have missing population allocation "
            f"for selected scenario '{selected_period}'."
        )

    population_ids = pd.to_numeric(population["id_building"], errors="coerce")
    if population_ids.isna().any() or (population_ids % 1 != 0).any():
        raise ValueError("Database population input contains invalid building IDs.")

    population["id_building"] = population_ids.astype("int64")
    if population["id_building"].duplicated().any():
        raise ValueError("Duplicate id_building values in population input.")

    selected_population = pd.to_numeric(
        population["selected_population"], errors="coerce"
    )
    if selected_population.isna().any():
        raise ValueError("Selected building population contains missing values.")
    if (selected_population < 0).any():
        raise ValueError("Selected building population contains negative values.")

    population["selected_population"] = selected_population.astype(float)

    # 3. Join Function 2 damage data with selected DB population.
    buildings = damage.merge(
        population[[
            "id_building", "block_id", "selected_population"
        ]],
        on="id_building",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = buildings["_merge"].ne("both")
    if unmatched.any():
        examples = buildings.loc[unmatched, "id_building"].head(10).tolist()
        raise RuntimeError(
            f"{int(unmatched.sum()):,} Function 2 buildings did not match "
            f"the population input. Example IDs: {examples}"
        )

    buildings.drop(columns="_merge", inplace=True)
    print(f"Matched buildings: {len(buildings):,}")

    # Map selected population into the model's corresponding period fields.
    res_col, non_col, *_ = model.PERIODS[selected_period]
    buildings[res_col] = buildings["selected_population"]
    buildings[non_col] = 0.0
    buildings = model.add_population_periods(
        buildings, selected_period=selected_period
    )

    # Map selected block population to the corresponding model columns.
    _, _, block_res_col, block_non_col, block_out_col = (
        model.PERIODS[selected_period]
    )
    blocks = blocks.copy()
    blocks[block_res_col] = pd.to_numeric(
        blocks["res_selected"], errors="coerce"
    )
    blocks[block_non_col] = pd.to_numeric(
        blocks["non_selected"], errors="coerce"
    )
    blocks[block_out_col] = pd.to_numeric(
        blocks["out_selected"], errors="coerce"
    )

    # 4. Derive Firuzi type and identify unsupported rate combinations.
    buildings["firuzi_type"] = model.derive_firuzi_type_vectorized(buildings)

    known_pairs = set(model.CASUALTY_RATES)
    pairs = list(zip(buildings["firuzi_type"], buildings["damage_state"]))
    valid_pair = pd.Series(
        [pair in known_pairs for pair in pairs],
        index=buildings.index,
    )

    buildings["casualty_status"] = "missing_casualty_rate"
    buildings.loc[
        buildings["damage_state"].isna(), "casualty_status"
    ] = "missing_damage_state"
    buildings.loc[
        buildings["damage_state"].notna()
        & buildings["firuzi_type"].isna(),
        "casualty_status",
    ] = "missing_firuzi_type"
    buildings.loc[valid_pair, "casualty_status"] = "ok"

    print("\\nFiruzi type counts:")
    print(buildings["firuzi_type"].value_counts(dropna=False).to_string())
    print("\\nBuilding casualty status counts:")
    print(buildings["casualty_status"].value_counts(dropna=False).to_string())

    # 5. Estimate casualties for this selected period only.
    building_results, block_results, valid_rate, block_status = (
        model.estimate_casualties_vectorized(
            buildings,
            blocks,
            block_id_col="block_id",
            selected_period=selected_period,
        )
    )

    selected_keys = [
        f"{selected_period}_{severity}_est"
        for severity in model.SEVERITIES
    ]

    for key, values in building_results.items():
        buildings[key] = values
    for key, values in block_results.items():
        blocks[key] = values

    # 6. Validate selected block estimates against selected population.
    block_population = (
        blocks[[block_res_col, block_non_col, block_out_col]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .sum(axis=1)
    )

    casualty_matrix = blocks[selected_keys].apply(
        pd.to_numeric, errors="coerce"
    )
    missing_estimate = casualty_matrix.isna().any(axis=1)
    casualty_total = casualty_matrix.fillna(0).sum(axis=1)
    over_population = casualty_total > block_population + 1e-6

    validation = pd.DataFrame([{
        "earthquake_time": str(earthquake_time),
        "population_period": selected_period,
        "blocks": len(blocks),
        "blocks_over_population": int(over_population.sum()),
        "blocks_with_missing_estimates": int(missing_estimate.sum()),
        "population_total": float(block_population.sum()),
        "NHS1_total": float(blocks[selected_keys[0]].sum(skipna=True)),
        "NHS2_total": float(blocks[selected_keys[1]].sum(skipna=True)),
        "NHS3_total": float(blocks[selected_keys[2]].sum(skipna=True)),
    }])

    print("\\nPopulation-cap validation:")
    print(validation.to_string(index=False))

    if (validation["blocks_over_population"] > 0).any():
        raise RuntimeError(
            "Population-cap validation failed. Inspect selected block "
            "population fields and casualty aggregation."
        )

    # 7. Return only one set of casualty fields, tagged with selected scenario.
    building_output = pd.DataFrame({
        "id_building": buildings["id_building"],
        "block_id": buildings["block_id"],
        "earthquake_time": str(earthquake_time),
        "population_period": selected_period,
        "damage_ratio": buildings["dmg_ratio"],
        "damage_state": buildings["damage_state"],
        "firuzi_type": buildings["firuzi_type"],
        "casualty_status": buildings["casualty_status"],
    })

    for severity in model.SEVERITIES:
        building_output[f"{severity}_est"] = buildings[
            f"{selected_period}_{severity}_est"
        ]
    building_output["medical_demand_est"] = building_output["NHS2_est"]

    block_output = pd.DataFrame({
        "block_id": blocks["block_id"],
        "earthquake_time": str(earthquake_time),
        "population_period": selected_period,
        "casualty_status": block_status,
    })
    for severity in model.SEVERITIES:
        block_output[f"{severity}_est"] = blocks[
            f"{selected_period}_{severity}_est"
        ]
    block_output["medical_demand_est"] = block_output["NHS2_est"]

    return building_output, block_output, validation
