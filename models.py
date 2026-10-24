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

def best_reference_params(ds_train):
    """
    For each target turbine:
    1. Compute the correlation with every reference turbine.
    2. Select the reference with the highest correlation.
    3. Fit one linear regression:

           target = intercept + slope * best_reference

    Returns one row per target turbine.
    """
    wind = ds_train.xs("wind_speed", axis=1, level=1)

    targets = target_turbines(ds_train)
    refs = reference_turbines(ds_train)

    rows = {}

    for target in targets:

        # Correlation between this target and every reference turbine
        correlations = wind[refs].corrwith(wind[target])

        # Best reference = highest correlation
        best_ref = correlations.idxmax()

        # Use only rows where both target and reference are available
        ok = wind[best_ref].notna() & wind[target].notna()

        x = wind.loc[ok, best_ref]
        y = wind.loc[ok, target]

        # Linear regression: target = intercept + slope * reference
        slope, intercept = np.polyfit(x, y, 1)

        corr = correlations[best_ref]

        rows[target] = {
            "reference": best_ref,
            "correlation": corr,
            "slope": slope,
            "intercept": intercept,
            "r2": corr**2,
            "n": int(ok.sum()),
        }

    return pd.DataFrame(rows).T

def best_reference_linear(ds_train, ds):
    """
    Predict each target turbine using only its best-correlated
    reference turbine from the training data.

    The linear relationship is learned from ds_train and then
    applied to ds.
    """
    params = best_reference_params(ds_train)

    wind = ds.xs("wind_speed", axis=1, level=1)

    predictions = {}

    for target, p in params.iterrows():

        ref = p["reference"]

        predictions[target] = (
            p["intercept"]
            + p["slope"] * wind[ref]
        ).clip(lower=0)

    return pd.DataFrame(
        predictions,
        index=ds.index,
    )
from sklearn.linear_model import LinearRegression, Ridge

def top3_multiple_linear(ds_train, ds):
    """
    For each target turbine:
    1. Rank reference turbines by correlation.
    2. Select the top 3 references.
    3. Fit one multiple linear regression using those 3 references.
    4. Predict the target wind speed on ds.
    """

    wind_train = ds_train.xs("wind_speed", axis=1, level=1)
    wind_ds = ds.xs("wind_speed", axis=1, level=1)

    targets = target_turbines(ds_train)
    refs = reference_turbines(ds_train)

    predictions = pd.DataFrame(index=ds.index)

    for target in targets:

        # Rank references by correlation with the target
        correlations = (
            wind_train[refs]
            .corrwith(wind_train[target])
            .sort_values(ascending=False)
        )

        best_refs = list(correlations.index[:3])

        # Training data
        X_train = wind_train[best_refs]
        y_train = wind_train[target]

        valid_train = X_train.notna().all(axis=1) & y_train.notna()

        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        # Fit multiple linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Data to predict
        X_pred = wind_ds[best_refs].copy()

        # Simple fallback for missing reference values
        X_pred = X_pred.interpolate(limit_direction="both")
        X_pred = X_pred.fillna(X_train.median())

        pred = model.predict(X_pred)

        predictions[target] = np.clip(pred, 0, None)

    return predictions
def ridge_all_references(ds_train, ds):
    """
    Predict each target turbine using all reference turbines
    with Ridge regression.
    """

    wind_train = ds_train.xs("wind_speed", axis=1, level=1)
    wind_ds = ds.xs("wind_speed", axis=1, level=1)

    targets = target_turbines(ds_train)
    refs = reference_turbines(ds_train)

    predictions = pd.DataFrame(index=ds.index)

    for target in targets:

        X_train = wind_train[refs]
        y_train = wind_train[target]

        valid_train = X_train.notna().all(axis=1) & y_train.notna()

        X_train = X_train.loc[valid_train]
        y_train = y_train.loc[valid_train]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        X_pred = wind_ds[refs].copy()

        # Fallback for missing reference values
        X_pred = X_pred.interpolate(limit_direction="both")
        X_pred = X_pred.fillna(X_train.median())

        pred = model.predict(X_pred)

        predictions[target] = np.clip(pred, 0, None)

    return predictions
# TODO: further candidate models:
#   - multiple linear regression on all references
#   - separate fits per wind direction sector
