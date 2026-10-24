"""Prediction models: estimate the target turbines' wind speed (T1-T3) from reference turbines (H*, O*).

Not implemented yet - only the interface and TODOs.

    model = SomeModel().fit(ds_train)
    pred = model.predict(ds_test)   # DataFrame: index = ds_test.index, columns = ["T1", "T2", "T3"]
"""

import pandas as pd


class Model:
    """Base class for all models: learn on a train set, predict target wind speeds for any set."""

    name = "base"

    def fit(self, ds_train: pd.DataFrame) -> "Model":
        # TODO: learn the relationship target <- references from ds_train, return self.
        # TODO: only use turbines from OUTSIDE the target park as input (README rule):
        #       get_dataset.reference_turbines(ds_train) -> H1..H5, O1, O2. Never T* for another T.
        # TODO: with get_data(use_non_complete=True) ds_train contains NaNs - drop rows per model,
        #       only where the columns this model uses are missing.
        # TODO: O1/O2 air pressure is a constant placeholder and wind_speed_stddev is missing (README).
        raise NotImplementedError

    def predict(self, ds: pd.DataFrame) -> pd.DataFrame:
        # TODO: read only reference columns - target columns are NaN in ds_result and are hidden
        #       by evaluate() in ds_test.
        # TODO: fallback when a chosen reference is missing at a time point (e.g. next-best reference
        #       or park mean), so every outage time point gets a value.
        raise NotImplementedError


# TODO: candidate models, simplest first (the challenge prioritises explainability):
#   - best single reference per target (highest correlation), linear fit T = a * ref + b
#   - several references combined (mean of the H park / multiple linear regression)
#   - separate fits per wind direction sector
