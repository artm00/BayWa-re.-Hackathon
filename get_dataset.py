"""Load the wind turbine data and split it into train, test and result sets.

    ds_train, ds_test, ds_result = get_data()

Each dataset is one DataFrame with a 10-minute timestamp index (Berlin local time, no tz) and
two-level columns (turbine, variable), plus a "complete" flag:

    ds_train["O1"]                               # all O1 variables
    ds_train["O1", "wind_speed"]                 # one Series
    ds_train.xs("wind_speed", axis=1, level=1)   # wind speed of all turbines side by side
    ds_train[ds_train["complete"]]               # rows where every turbine has a wind speed
    ds_train["O1", "sector"]                     # 30° wind direction sector of O1's park (O)
"""

from pathlib import Path

import numpy as np
import pandas as pd

FREQ = "10min"
TARGET_PREFIX = "T"  # README: target turbines start with T, all others are references
FROZEN_RUN = 6       # the same wind_direction for this many consecutive timestamps = stuck sensor


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


def valid_directions(direction):
    """wind_direction (timestamp x turbine) with every invalid value set to NaN.

    Invalid: missing, outside [0, 360], or frozen - the same value for FROZEN_RUN or more consecutive
    timestamps (all values of such a run are invalid).
    """
    direction = direction.where((direction >= 0) & (direction <= 360))

    def frozen(values):
        run_id = (values != values.shift()).cumsum()   # new run whenever the value changes (NaN always does)
        return values.groupby(run_id).transform("size").ge(FROZEN_RUN) & values.notna()

    return direction.mask(direction.apply(frozen))


def add_direction_sectors(data):
    """Add park_dir, n_dir_valid and sector to every turbine; the park is the letter of the turbine name.

    park_dir: circular mean of the park's valid wind directions, arctan2(mean sin, mean cos) in [0, 360).
        Never a plain mean of degrees: the mean of 350° and 10° is 0°, not 180°.
    n_dir_valid: number of turbines in the park with a valid direction.
    sector: 12 sectors of 30°, sector 0 centred on north (345°-15°).
    All three are NaN where no turbine of the park has a valid direction.
    """
    turbines = [t for t in data.columns.unique("turbine") if t != "complete"]
    direction = valid_directions(data.xs("wind_direction", axis=1, level=1)[turbines])
    rad = np.radians(direction)

    new, sectors = {}, {}
    for park in sorted({t[0] for t in turbines}):
        members = [t for t in turbines if t[0] == park]
        n_valid = direction[members].notna().sum(axis=1)
        park_dir = np.degrees(np.arctan2(np.sin(rad[members]).mean(axis=1), np.cos(rad[members]).mean(axis=1))) % 360
        park_dir = park_dir.mask(park_dir >= 360, 0.0).where(n_valid > 0)   # float rounding can give exactly 360
        sector = ((park_dir + 15) % 360 // 30).astype("Int64")
        for t in members:
            new[t, "park_dir"] = park_dir
            new[t, "n_dir_valid"] = n_valid.where(n_valid > 0).astype("Int64")
            new[t, "sector"] = sector
        sectors[park] = sector
        print(f"park {park}: {park_dir.isna().sum()} timestamps without a valid wind direction")

    print("samples per park (rows) and sector (columns):")
    print(pd.DataFrame({park: s.value_counts() for park, s in sectors.items()}).T.reindex(columns=range(12)).fillna(0).astype(int))

    data = pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)
    data.columns.names = ["turbine", "variable"]
    return data[turbines + ["complete"]]   # keep each turbine's columns together


def get_data(data_dir="data", test_start="2026-09-04", test_end=None, result_period=None, use_non_complete=False):
    """Load all turbines and split them chronologically into train, test and result sets.

    test_start, test_end: own validation period, both inclusive. test_end=None runs to the end of
        the data; a date like "2026-09-09" includes that whole day (README "shown test data": Sep 4-9).
    result_period: (start, end) of the outage to predict for the submission. None detects it
        automatically (README "hidden test data": Sep 1-3).
    use_non_complete: False keeps only train rows where every turbine has a wind speed; True keeps
        all train rows with their NaNs. Test and result are never filtered.

    Returns ds_train, ds_test, ds_result. Train is everything outside test and result. Every turbine
    also gets park_dir, n_dir_valid and sector (see add_direction_sectors), added before the split.
    """
    data = add_direction_sectors(load_turbines(data_dir))

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

