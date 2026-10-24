"""Load the wind turbine data and split it into train, test and result sets.

    ds_train, ds_test, ds_result = get_data()

Each dataset is one DataFrame with a 10-minute timestamp index (Berlin local time, no tz) and
two-level columns (turbine, variable), plus a "complete" flag:

    ds_train["O1"]                               # all O1 variables
    ds_train["O1", "wind_speed"]                 # one Series
    ds_train.xs("wind_speed", axis=1, level=1)   # wind speed of all turbines side by side
    ds_train[ds_train["complete"]]               # rows where every turbine has a wind speed
"""

from pathlib import Path

import pandas as pd

FREQ = "10min"
TARGET_PREFIX = "T"  # README: target turbines start with T, all others are references


def load_turbines(data_dir="data"):
    """Read every <turbine>_data.xlsx into one wide DataFrame on a gap-free 10-minute grid.

    Timestamps missing from the files (e.g. Aug 3-4) are added as NaN rows.
    """
    frames = {}
    for path in sorted(Path(data_dir).glob("*_data.xlsx")):
        if path.name.startswith("~$"):  # Excel lock file of an open workbook
            continue
        df = pd.read_excel(path)
        name = df["turbine_name"].iloc[0]
        # Excel stores dates as float days; round away the float noise (e.g. 00:09:59.999)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("s")
        frames[name] = df.drop(columns=["turbine_name", "test_set"], errors="ignore").set_index("timestamp")

    data = pd.concat(frames, axis=1)
    data.columns.names = ["turbine", "variable"]

    grid = pd.date_range(data.index.min(), data.index.max(), freq=FREQ, name="timestamp")
    off_grid = data.index.difference(grid)
    if len(off_grid):
        raise ValueError(f"timestamps not on the {FREQ} grid: {list(off_grid[:5])}")
    data = data.reindex(grid)

    data["complete"] = data.xs("wind_speed", axis=1, level=1).notna().all(axis=1)
    return data


def target_turbines(data):
    """Turbines whose wind speed we predict (T1, T2, ...)."""
    return [t for t in data.columns.unique("turbine") if t.startswith(TARGET_PREFIX)]


def reference_turbines(data):
    """Turbines outside the target park (H*, O*) - the only ones a model may learn from."""
    targets = target_turbines(data)
    return [t for t in data.columns.unique("turbine") if t not in targets and t != "complete"]


def detect_outage(data):
    """First and last timestamp of the longest run where every target turbine lacks wind speed.

    Rows where the references are missing too are data gaps (e.g. Aug 3-4), not an outage.
    """
    wind = data.xs("wind_speed", axis=1, level=1)
    outage = wind[target_turbines(data)].isna().all(axis=1) & wind[reference_turbines(data)].notna().any(axis=1)
    run_id = (outage != outage.shift()).cumsum()[outage]
    if run_id.empty:
        raise ValueError("no outage found: no time point where all target turbines are missing")
    period = run_id.index[run_id == run_id.value_counts().idxmax()]
    return period[0], period[-1]


def get_data(data_dir="data", test_start="2026-09-04", test_end=None, result_period=None, use_non_complete=False):
    """Load all turbines and split them chronologically into train, test and result sets.

    test_start, test_end: own validation period, both inclusive. test_end=None runs to the end of
        the data; a date like "2026-09-09" includes that whole day (README "shown test data": Sep 4-9).
    result_period: (start, end) of the outage to predict for the submission. None detects it
        automatically (README "hidden test data": Sep 1-3).
    use_non_complete: False keeps only train rows where every turbine has a wind speed; True keeps
        all train rows with their NaNs. Test and result are never filtered.

    Returns ds_train, ds_test, ds_result. Train is everything outside test and result.
    """
    data = load_turbines(data_dir)

    if result_period is None:
        result_period = detect_outage(data)
    ds_result = data.loc[result_period[0]:result_period[1]].copy()
    ds_test = data.loc[test_start:test_end].copy()

    overlap = ds_test.index.intersection(ds_result.index)
    if len(overlap):
        raise ValueError(f"test period overlaps the result period ({overlap[0]} - {overlap[-1]})")

    ds_train = data.drop(ds_test.index.union(ds_result.index))
    if not use_non_complete:
        ds_train = ds_train[ds_train["complete"]]
    return ds_train.copy(), ds_test, ds_result

