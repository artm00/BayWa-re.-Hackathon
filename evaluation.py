"""Model evaluation and selection.

    scores = pd.concat({name: evaluate(m, ds_train, ds_test, curves) for name, m in models.items()},
                       names=["model", "turbine"])
    best_name = select_best(scores)
"""

import numpy as np
import pandas as pd

from get_dataset import target_turbines
from power_curves import energy_kwh


def evaluate(model, ds_train: pd.DataFrame, ds_test: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Score a model: learn on ds_train, predict and score ds_test.

    The target turbines are hidden (set to NaN) before predicting, so the model sees the test period
    exactly like an outage. Errors are computed only where the real wind speed exists.
    model: function model(ds_train, ds) -> predicted target wind speeds, e.g. models.neighbour_mean.
    curves: {turbine: power curve function}, i.e. get_dataset.POWER_CURVES.

    Returns one row per target turbine with columns
    wind_rmse, wind_mae (m/s), power_rmse, power_mae (kW), energy_dev_pct, coverage.
    """
    targets = target_turbines(ds_test)
    hidden = ds_test.copy()
    hidden.loc[:, hidden.columns.get_level_values("turbine").isin(targets)] = np.nan
    pred = model(ds_train, hidden)
    truth = ds_test.xs("wind_speed", axis=1, level=1)

    rows = {}
    for t in targets:
        ok = truth[t].notna() & pred[t].notna()
        y, p = truth[t][ok], pred[t][ok]
        wind_err = p - y
        power_err = curves[t](p) - curves[t](y)
        energy_true, energy_pred = energy_kwh(curves[t], y), energy_kwh(curves[t], p)
        rows[t] = {
            "wind_rmse": np.sqrt((wind_err**2).mean()),
            "wind_mae": wind_err.abs().mean(),
            "power_rmse": np.sqrt((power_err**2).mean()),
            "power_mae": power_err.abs().mean(),
            # required metric "deviation of summed energy yield", over the same time points
            "energy_dev_pct": 100 * (energy_pred - energy_true) / energy_true,
            "coverage": pred[t].notna().mean(),
        }
    return pd.DataFrame(rows).T


def select_best(scores: pd.DataFrame, metric="wind_rmse") -> str:
    """Name of the model with the lowest mean |metric| over the target turbines.

    scores: comparison table from evaluate(), index (model, turbine).
    """
    # TODO: decide whether a different model per target turbine is allowed.
    return scores[metric].abs().groupby(level=0).mean().idxmin()
