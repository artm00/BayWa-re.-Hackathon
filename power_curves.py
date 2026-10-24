"""Power curve of the target turbines: wind speed (m/s) in, power (kW) out.

T1, T2 and T3 have identical power curve files, so there is one curve for all three.

    powercurve(7.5)                              # power in kW at 7.5 m/s
    energy_kwh(pred["T1"])                       # energy of a series of 10-minute wind speeds
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
INTERVAL_HOURS = 10 / 60


def load_power_curve(turbine, data_dir=DATA_DIR):
    """Read <turbine>_power_curve.xlsx -> table of wind_speed (m/s) and power, sorted by wind speed.

    The files state no unit for power; 4500 = 4.5 MW rated, so kW.
    """
    table = pd.read_excel(Path(data_dir) / f"{turbine}_power_curve.xlsx")
    return table[["wind_speed", "power"]].sort_values("wind_speed").reset_index(drop=True)


POWER_CURVE = load_power_curve("T1")


def powercurve(wind_speed):
    """Power (kW) for each wind speed, on the straight line between the two neighbouring table points.

    0 below the first table point and above the last one (cut-out); NaN wind speed gives NaN power.
    Accepts a scalar, list, array or Series (a Series keeps its index).
    """
    p = np.interp(wind_speed, POWER_CURVE["wind_speed"], POWER_CURVE["power"], left=0.0, right=0.0)
    if isinstance(wind_speed, pd.Series):
        return pd.Series(p, index=wind_speed.index, name=wind_speed.name)
    return p


def energy_kwh(wind_speed):
    """Energy of a series of 10-minute wind speeds in kWh, e.g. energy_kwh(pred["T1"]).

    NaN intervals are skipped, so compare energies only over the same time points.
    """
    return float(np.nansum(powercurve(wind_speed)) * INTERVAL_HOURS)
