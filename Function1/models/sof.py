import numpy as np


def encode_sof(mechanism):
    """
    Convert faulting mechanism to Darzi SoF dummy variables.

    Darzi:
        Reverse/Thrust -> RV=1, SS=0
        Strike-slip    -> RV=0, SS=1
        Normal         -> RV=0, SS=0
        Unknown/Other  -> RV=0, SS=0

    Parameters
    ----------
    mechanism : str

    Returns
    -------
    tuple
        (RV, SS)
    """

    if mechanism is None:
        return 0, 0

    mechanism = mechanism.strip().lower()

    reverse_values = {
        "reverse",
        "thrust",
        "reverse_fault",
        "thrust_fault",
        "r"
    }

    strike_slip_values = {
        "strike-slip",
        "strike slip",
        "strike_slip",
        "ss"
    }

    normal_values = {
        "normal",
        "normal_fault",
        "n"
    }

    if mechanism in reverse_values:
        return 1, 0

    elif mechanism in strike_slip_values:
        return 0, 1

    elif mechanism in normal_values:
        return 0, 0

    else:
        raise ValueError(
            f"Unknown faulting mechanism: {mechanism}"
        )