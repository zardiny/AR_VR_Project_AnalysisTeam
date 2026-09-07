from models.darzi import calculate_darzi_pga


# ============================================================
# Input Validation Test
# ============================================================

print("=" * 70)
print("DARZI INPUT VALIDATION TEST")
print("=" * 70)


# ============================================================
# Helper function
# ============================================================

def test_invalid_input(name, **kwargs):

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    try:

        result = calculate_darzi_pga(**kwargs)

        print("Result : NO ERROR")
        print(f"Returned value : {result}")

        return False

    except (ValueError, TypeError) as e:

        print("Result : ERROR DETECTED")
        print(f"Error type : {type(e).__name__}")
        print(f"Message    : {e}")

        return True


# ============================================================
# Valid reference input
# ============================================================

valid_input = {
    "magnitude": 6.0,
    "hypocentral_distance_km": 30.0,
    "site_class": 2,
    "mechanism": "reverse",
}


# ============================================================
# 1. Invalid magnitude: negative
# ============================================================

passed = test_invalid_input(
    "Test 1: Negative magnitude",
    **{
        **valid_input,
        "magnitude": -1.0,
    }
)

assert passed, "Negative magnitude was not rejected."


# ============================================================
# 2. Invalid distance: negative
# ============================================================

passed = test_invalid_input(
    "Test 2: Negative distance",
    **{
        **valid_input,
        "hypocentral_distance_km": -10.0,
    }
)

assert passed, "Negative distance was not rejected."


# ============================================================
# 3. Invalid site class
# ============================================================

passed = test_invalid_input(
    "Test 3: Invalid site class",
    **{
        **valid_input,
        "site_class": 4,
    }
)

assert passed, "Invalid site class was not rejected."


# ============================================================
# 4. Invalid mechanism
# ============================================================

passed = test_invalid_input(
    "Test 4: Invalid mechanism",
    **{
        **valid_input,
        "mechanism": "unknown",
    }
)

assert passed, "Invalid mechanism was not rejected."


# ============================================================
# 5. Non-numeric magnitude
# ============================================================

passed = test_invalid_input(
    "Test 5: Non-numeric magnitude",
    **{
        **valid_input,
        "magnitude": "six",
    }
)

assert passed, "Non-numeric magnitude was not rejected."


# ============================================================
# 6. Non-numeric distance
# ============================================================

passed = test_invalid_input(
    "Test 6: Non-numeric distance",
    **{
        **valid_input,
        "hypocentral_distance_km": "thirty",
    }
)

assert passed, "Non-numeric distance was not rejected."


# ============================================================
# 7. None magnitude
# ============================================================

passed = test_invalid_input(
    "Test 7: None magnitude",
    **{
        **valid_input,
        "magnitude": None,
    }
)

assert passed, "None magnitude was not rejected."


# ============================================================
# 8. None distance
# ============================================================

passed = test_invalid_input(
    "Test 8: None distance",
    **{
        **valid_input,
        "hypocentral_distance_km": None,
    }
)

assert passed, "None distance was not rejected."


# ============================================================
# 9. Invalid site class type
# ============================================================

passed = test_invalid_input(
    "Test 9: Non-numeric site class",
    **{
        **valid_input,
        "site_class": "II",
    }
)

assert passed, "Invalid site class type was not rejected."


# ============================================================
# 10. Invalid mechanism type
# ============================================================

passed = test_invalid_input(
    "Test 10: None mechanism",
    **{
        **valid_input,
        "mechanism": None,
    }
)

assert passed, "None mechanism was not rejected."


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 70)
print("INPUT VALIDATION TEST: PASS")
print("=" * 70)