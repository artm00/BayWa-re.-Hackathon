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

# wind direction sectors come from park H (column "sector" of H1, shared by H1-H5): the target park's
# own direction is missing during the outage and hidden by evaluate(), so it cannot be used
SECTOR_TURBINE = "H1"
MIN_SECTOR_ROWS = 150   # fewer training rows in a sector -> no own line, the global line is used there


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


def neighbour_mean_sector_params(ds_train):
    """neighbour_mean_params fitted separately for each of the 12 wind direction sectors (30° each).

    Returns one row per (sector, target turbine) with slope, intercept, r2, corr and n. Sectors with
    fewer than MIN_SECTOR_ROWS training rows are left out.
    """
    sector = ds_train[SECTOR_TURBINE, "sector"]
    params = {}
    for s in range(12):
        in_sector = ds_train[sector.eq(s).fillna(False)]   # rows without a sector belong to none
        if len(in_sector) >= MIN_SECTOR_ROWS:
            params[s] = neighbour_mean_params(in_sector)
    return pd.concat(params, names=["sector", "target"])


def neighbour_mean_sector(ds_train, ds):
    """Like neighbour_mean, but with one straight line per wind direction sector and target turbine.

    For every time point: look up the sector (park H's wind direction) and use that sector's slope
    and intercept: T = intercept[sector] + slope[sector] * neighbours, clipped at 0 m/s.
    Where the sector is unknown or had too few training rows, the global neighbour_mean line is used.
    """
    global_params = neighbour_mean_params(ds_train)
    sector_params = neighbour_mean_sector_params(ds_train)
    x = neighbour_average(ds)
    sector = ds[SECTOR_TURBINE, "sector"].astype(float)   # NaN where unknown

    pred = {}
    for t, g in global_params.iterrows():
        per_sector = sector_params.xs(t, level="target")   # index: sector number
        slope = sector.map(per_sector["slope"]).fillna(g.slope)
        intercept = sector.map(per_sector["intercept"]).fillna(g.intercept)
        pred[t] = (intercept + slope * x).clip(lower=0)
    return pd.DataFrame(pred, index=ds.index)


# TODO: further candidate models, simplest first (the challenge prioritises explainability):
#   - best single reference per target (highest correlation), linear fit T = a * ref + b
#   - multiple linear regression on all references / mean of the H park only
