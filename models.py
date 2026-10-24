"""Prediction models: estimate the target turbines' wind speed (T1-T3) from reference turbines (H*, O*).

Every model is one function: learn on ds_train, predict the target wind speeds for ds.

    pred = neighbour_mean(ds_train, ds_test)   # DataFrame: index = ds_test.index, columns = ["T1", "T2", "T3"]

Rules for a new model function model(ds_train, ds) -> DataFrame:
    TODO: only use turbines from OUTSIDE the target park as input (README rule):
          get_dataset.reference_turbines(ds_train) -> H1..H5, O1, O2. Never T* for another T.
    TODO: read only reference columns of ds - target columns are NaN in ds_result and are hidden
          by evaluate() in ds_test.
    TODO: with get_data(use_non_complete=True) ds_train contains NaNs - drop rows per model,
          only where the columns this model uses are missing.
    TODO: fallback when a chosen reference is missing at a time point (e.g. next-best reference
          or park mean), so every outage time point gets a value.
    TODO: O1/O2 air pressure is a constant placeholder and wind_speed_stddev is missing (README).
"""

import numpy as np
import pandas as pd

from get_dataset import reference_turbines, target_turbines


def neighbour_average(ds):
    """Mean wind speed of all reference turbines at every time point (Excel AVERAGE).

    A reference missing at a time point is left out of the mean, so the average exists as long as
    at least one reference has a value.
    """
    return ds.xs("wind_speed", axis=1, level=1)[reference_turbines(ds)].mean(axis=1)


def neighbour_mean_params(ds_train):
    """Fit one straight line per target turbine: T = intercept + slope * neighbour_average.

    Returns one row per target turbine with slope, intercept, r2, corr and n (rows used).
    """
    wind = ds_train.xs("wind_speed", axis=1, level=1)
    x_all = neighbour_average(ds_train)
    rows = {}
    for t in target_turbines(ds_train):
        ok = x_all.notna() & wind[t].notna()
        x, y = x_all[ok], wind[t][ok]
        slope, intercept = np.polyfit(x, y, 1)
        corr = np.corrcoef(x, y)[0, 1]
        rows[t] = {"slope": slope, "intercept": intercept, "r2": corr**2, "corr": corr, "n": int(ok.sum())}
    return pd.DataFrame(rows).T.astype({"n": int})


def neighbour_mean(ds_train, ds):
    """Average wind speed of all reference turbines, one straight line per target turbine.

    Same method as "4_Wind park T - all turbines calculation.xlsx":
        neighbours = mean wind speed of H1..H5, O1, O2           (Excel AVERAGE)
        T = intercept + slope * neighbours, clipped at 0 m/s     (Excel SLOPE / INTERCEPT / MAX(0, ...))

    The lines are learned on ds_train (see neighbour_mean_params) and applied to ds.
    """
    params = neighbour_mean_params(ds_train)
    x = neighbour_average(ds)
    return pd.DataFrame(
        {t: (p.intercept + p.slope * x).clip(lower=0) for t, p in params.iterrows()},
        index=ds.index,
    )


# TODO: further candidate models, simplest first (the challenge prioritises explainability):
#   - best single reference per target (highest correlation), linear fit T = a * ref + b
#   - multiple linear regression on all references / mean of the H park only
#   - separate fits per wind direction sector
