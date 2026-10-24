"""Model evaluation and selection.

Not implemented yet - only the interface and TODOs.
"""

import pandas as pd


def evaluate(model, ds_test: pd.DataFrame, curves: dict) -> pd.DataFrame:
    """Score a fitted model on ds_test.

    Returns one row per target turbine with columns
    wind_rmse, wind_mae, power_rmse, power_mae, energy_dev_pct, coverage.
    """
    # TODO: hide the answers - copy ds_test and set all target (T*) columns to NaN before calling
    #       model.predict, so the model sees the test period exactly like an outage.
    # TODO: compare with the real T wind speeds only where the real value exists
    #       (the T turbines have their own random NaNs in the test period).
    # TODO: power = curves[t].power(...) for prediction and truth -> power RMSE / MAE.
    # TODO: energy deviation = (predicted kWh - real kWh) / real kWh over the same time points
    #       (required metric: "deviation of summed energy yield").
    # TODO: coverage = share of time points where the model returned a value.
    # TODO: return a DataFrame (index = turbine) so models compare in one table:
    #       pd.concat({name: evaluate(m, ds_test, curves) for name, m in models.items()})
    raise NotImplementedError


def select_best(scores: pd.DataFrame) -> str:
    """Name of the best model in a comparison table from evaluate()."""
    # TODO: decide the selection criterion - one main metric (e.g. mean wind RMSE, or energy
    #       deviation since the insurance claim is about energy), the others as tie-breakers.
    # TODO: decide whether a different model per target turbine is allowed.
    raise NotImplementedError
