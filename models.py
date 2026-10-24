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
RIDGE_ALPHA = 1.0       # strength of the Ridge penalty on large weights (same default as sklearn's Ridge)


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


def fit_per_sector(params_function, ds_train):
    """Run params_function separately on the training rows of each of the 12 wind direction sectors.

    Returns its rows for every sector, indexed by (sector, target turbine). Sectors with fewer than
    MIN_SECTOR_ROWS training rows are left out.
    """
    sector = ds_train[SECTOR_TURBINE, "sector"]
    params = {}
    for s in range(12):
        in_sector = ds_train[sector.eq(s).fillna(False)]   # rows without a sector belong to none
        if len(in_sector) >= MIN_SECTOR_ROWS:
            params[s] = params_function(in_sector)
    return pd.concat(params, names=["sector", "target"])


def neighbour_mean_sector_params(ds_train):
    """neighbour_mean_params fitted separately for each of the 12 wind direction sectors (30° each).

    Returns one row per (sector, target turbine) with slope, intercept, r2, corr and n.
    """
    return fit_per_sector(neighbour_mean_params, ds_train)


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


def ridge_fit(X, y, alpha=RIDGE_ALPHA):
    """Ridge regression y = intercept + X @ weights: least squares plus a penalty alpha * sum(weights²).

    Same result as sklearn.linear_model.Ridge(alpha): the intercept is not penalised, so X and y are
    centred first and the weights solve (Xc' Xc + alpha * I) w = Xc' yc.
    """
    X_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - X_mean, y - y_mean
    weights = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return weights, y_mean - X_mean @ weights


def ridge_all_references_params(ds_train):
    """Fit one Ridge regression per target turbine on the wind speed of every reference turbine:

        T = intercept + w_H1 * H1 + w_H2 * H2 + ... + w_O2 * O2

    Returns one row per target turbine: one weight per reference, intercept, r2 (share of the target's
    variation the formula explains on the training rows) and n (rows used: all references and target valid).
    """
    wind = ds_train.xs("wind_speed", axis=1, level=1)
    refs = reference_turbines(ds_train)
    rows = {}
    for t in target_turbines(ds_train):
        ok = wind[refs].notna().all(axis=1) & wind[t].notna()
        X, y = wind.loc[ok, refs].to_numpy(), wind.loc[ok, t].to_numpy()
        weights, intercept = ridge_fit(X, y)
        fitted = intercept + X @ weights
        r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows[t] = {**dict(zip(refs, weights)), "intercept": intercept, "r2": r2, "n": int(ok.sum())}
    return pd.DataFrame(rows).T.astype({"n": int})


def reference_inputs(ds_train, ds):
    """Reference wind speeds of ds (time x reference) with every gap filled, so each time point gets a
    prediction: a straight line in time between the values around a gap, otherwise the training median.
    """
    refs = reference_turbines(ds)
    train_median = ds_train.xs("wind_speed", axis=1, level=1)[refs].median()
    return ds.xs("wind_speed", axis=1, level=1)[refs].interpolate(limit_direction="both").fillna(train_median)


def ridge_all_references(ds_train, ds):
    """Every reference turbine with its own weight, one Ridge regression per target turbine.

        T = intercept + w_H1 * H1 + ... + w_O2 * O2, clipped at 0 m/s

    The weights are learned on ds_train (see ridge_all_references_params) and applied to ds, after
    filling missing reference values (see reference_inputs).
    """
    params = ridge_all_references_params(ds_train)
    X = reference_inputs(ds_train, ds)
    return pd.DataFrame(
        {t: (p["intercept"] + X @ p[X.columns]).clip(lower=0) for t, p in params.iterrows()},
        index=ds.index,
    )


def ridge_all_references_sector_params(ds_train):
    """ridge_all_references_params fitted separately for each of the 12 wind direction sectors (30° each).

    Returns one row per (sector, target turbine) with the weights, intercept, r2 and n.
    """
    return fit_per_sector(ridge_all_references_params, ds_train)


def ridge_all_references_sector(ds_train, ds):
    """Like ridge_all_references, but with its own weights per wind direction sector and target turbine.

    For every time point: look up the sector (park H's wind direction) and use that sector's weights and
    intercept. Where the sector is unknown or had too few training rows, the global weights are used.
    """
    global_params = ridge_all_references_params(ds_train)
    sector_params = ridge_all_references_sector_params(ds_train)
    X = reference_inputs(ds_train, ds)
    sector = ds[SECTOR_TURBINE, "sector"].astype(float)   # NaN where unknown

    pred = {}
    for t, g in global_params.iterrows():
        per_sector = sector_params.xs(t, level="target")   # index: sector number
        # weights and intercept of each time point's sector (time x [references..., intercept])
        coef = pd.DataFrame({c: sector.map(per_sector[c]).fillna(g[c]) for c in [*X.columns, "intercept"]})
        pred[t] = (coef["intercept"] + (coef[X.columns] * X).sum(axis=1)).clip(lower=0)
    return pd.DataFrame(pred, index=ds.index)


# TODO: further candidate models, simplest first (the challenge prioritises explainability):
#   - best single reference per target (highest correlation), linear fit T = a * ref + b
#   - mean of the H park only
