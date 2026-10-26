"""Self-contained traffic predictor: base LightGBM model + region x hour calibration.

Supported: sites seen in the training years (<= 2019); hours 7-18 (other hours get factor 1.0).
The calibration factors are tied to the base model: if you retrain the base model, refit the factors.
Only load .pkl files you produced yourself (pickle can execute code).
"""
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"          # artifacts live in models/ next to this file
MODEL_FILE, META_FILE, CALIB_FILE = "traffic_lgbm.txt", "traffic_meta.pkl", "traffic_calib.pkl"

# must match the training code
BASE = ["region_id", "hour", "dow", "month", "is_hol", "road_type",
        "latitude", "longitude", "link_length_km"]
FEATS = BASE + ["site_prior", "site_last", "site_gap"]
KEYS = ["region_id", "hour"]
HOL = pd.to_datetime(list(holidays.UK(years=range(2000, 2031)).keys()))


class UnknownSiteError(ValueError):
    """Raised when a site id was not present in the training years."""


def _features(rows, S):
    """Model features for prediction rows (inference-only version of the training feature builder)."""
    rows = rows.reset_index(drop=True)
    X = rows[BASE].copy()
    j = rows[["count_point_id"]].join(S["site"], on="count_point_id")
    X["site_prior"] = (j["sum"] / j["count"].where(j["count"] > 0)).values
    lv = S["sy"].assign(lvl=lambda s: s["sum"] / s["count"], last_year=lambda s: s["year"])
    lv = lv.sort_values("year")[["count_point_id", "year", "lvl", "last_year"]]
    r = rows[["count_point_id", "year"]].reset_index()
    r["year"] = r["year"].astype(lv["year"].dtype)                       # merge_asof needs identical dtypes
    r["count_point_id"] = r["count_point_id"].astype(lv["count_point_id"].dtype)
    r = r.sort_values("year")
    m = (pd.merge_asof(r, lv, on="year", by="count_point_id", allow_exact_matches=False)
           .set_index("index").loc[rows.index])              # strictly earlier years only
    X["site_last"] = m["lvl"].values
    X["site_gap"] = (rows["year"] - m["last_year"]).values
    X["region_id"] = pd.Categorical(X["region_id"], categories=S["region_cats"])
    X["road_type"] = pd.Categorical(X["road_type"], categories=S["road_cats"])
    return X[FEATS]


class TrafficPredictor:
    def __init__(self, folder=MODELS_DIR):
        folder = Path(folder)
        self.booster = lgb.Booster(model_file=str(folder / MODEL_FILE))
        with open(folder / META_FILE, "rb") as f:
            self.S = pickle.load(f)["S"]
        with open(folder / CALIB_FILE, "rb") as f:
            self.factors = pickle.load(f)["factors"]
        self._sites = self.S["site"].index

    def is_known_site(self, site_id):
        return site_id in self._sites

    def predict_many(self, df, on_unknown="raise"):
        """df columns: count_point_id, date, hour.
        Returns a DataFrame (same index as df) with columns base, factor, calibrated.
        on_unknown: "raise" -> UnknownSiteError; "nan" -> NaN rows for unknown sites."""
        r = df[["count_point_id", "date", "hour"]].copy()
        known = r.count_point_id.isin(self._sites).to_numpy()
        if not known.all() and on_unknown == "raise":
            bad = r.count_point_id[~known].unique()[:10].tolist()
            raise UnknownSiteError(f"Unknown sites (not in training years): {bad}")

        out = pd.DataFrame({"base": np.nan, "factor": np.nan, "calibrated": np.nan}, index=df.index)
        if known.any():
            k = r[known].copy()
            k["date"] = pd.to_datetime(k["date"])
            k["year"], k["dow"], k["month"] = k.date.dt.year, k.date.dt.dayofweek, k.date.dt.month
            k["is_hol"] = k.date.dt.normalize().isin(HOL).astype(int)
            k = k.join(self.S["attrs"], on="count_point_id")
            base = self.booster.predict(_features(k, self.S))
            fac = k.join(self.factors, on=KEYS)["f"].fillna(1.0).to_numpy()
            out.loc[known, "base"] = base
            out.loc[known, "factor"] = fac
            out.loc[known, "calibrated"] = base * fac
        return out

    def predict_one(self, site_id, date, hour, calibrated=True):
        """Single prediction as a float. Raises UnknownSiteError for unseen sites."""
        res = self.predict_many(pd.DataFrame({"count_point_id": [site_id], "date": [date], "hour": [hour]}))
        return float(res["calibrated" if calibrated else "base"].iloc[0])


@lru_cache(maxsize=1)
def get_predictor():
    """Shared instance: model files are read once per process."""
    return TrafficPredictor()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
    else:
        site = int(sys.argv[1]) if sys.argv[1].isdigit() else sys.argv[1]
        res = get_predictor().predict_many(
            pd.DataFrame({"count_point_id": [site], "date": [sys.argv[2]], "hour": [int(sys.argv[3])]}))
        print(res.round(3).to_string(index=False))