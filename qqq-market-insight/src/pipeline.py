"""End-to-end QQQ technical-indicator analysis pipeline.

Workflow: download -> parse and clean -> feature engineering -> EDA ->
chronological split -> training -> evaluation -> held-out permutation
importance -> saved tables and figures.  The notebook imports these functions
instead of duplicating the analytical code.

Primary functions follow the supplied example: download_data, parse_raw, rsi,
add_features, metric_row, build_models, save_eda, and run. Supporting helpers
handle configuration, validation, provenance, probabilities, and file output.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from argparse import ArgumentParser
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qqq-market-insight-mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qqq_nasdaq_raw.json"
RAW_METADATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "qqq_nasdaq_raw.metadata.json"
)
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "qqq_features.csv"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"
TABLES_PATH = PROJECT_ROOT / "outputs" / "tables"
NASDAQ_API_ENDPOINT = "https://api.nasdaq.com/api/quote/{symbol}/historical"
NASDAQ_SOURCE_REFERENCE = (
    "https://www.nasdaq.com/market-activity/etf/qqq/historical"
)

REQUIRED_CONFIG_KEYS = {
    "symbol",
    "asset_class",
    "start_date",
    "end_date",
    "test_fraction",
    "random_state",
    "permutation_repeats",
}
MINIMUM_RAW_ROWS = 500
DOWNLOAD_LIMIT = 5000
REQUIRED_OHLCV_FIELDS = {"date", "open", "high", "low", "close", "volume"}

FEATURE_GROUPS = {
    "Return history": ["return_1d", "return_5d", "gap_return"],
    "Trend": [
        "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
        "sma_cross_5_20", "ema_ratio_12", "macd", "macd_signal",
    ],
    "Momentum": ["rsi_14", "stoch_k_14", "roc_10"],
    "Volatility": [
        "range_pct", "atr_14_pct", "volatility_10",
        "volatility_20", "bb_width_20",
    ],
    "Volume": ["volume_change", "volume_ratio_20", "obv_change_5"],
}
FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]
CV_SPLITS = 5
CV_GAP = 1


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load project parameters, allowing optional full-line // comments."""
    with path.open(encoding="utf-8") as config_file:
        config = json.loads("\n".join(
            line for line in config_file if not line.lstrip().startswith("//")
        ))

    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    start_date = date.fromisoformat(config["start_date"])
    end_date = date.fromisoformat(config["end_date"])
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    if not 0 < config["test_fraction"] < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if config["permutation_repeats"] < 1:
        raise ValueError("permutation_repeats must be at least 1")

    return config


def save_raw_response(
    response_body: bytes,
    *,
    request_url: str,
    retrieved_at_utc: str,
    raw_path: Path = RAW_DATA_PATH,
    metadata_path: Path = RAW_METADATA_PATH,
    force: bool = False,
    metadata_fields: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save exact response bytes plus separate provenance metadata.

    Existing raw artifacts are protected by default. A caller must explicitly
    pass ``force=True`` to replace either file. Validation may inspect the
    response, but the saved bytes are never reformatted or otherwise modified.
    """
    existing = [path for path in (raw_path, metadata_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Raw artifact already exists: {names}. "
            "Use force=True only for an intentional refresh."
        )
    if not response_body:
        raise ValueError("Cannot save an empty Nasdaq response")
    if not request_url.startswith("https://api.nasdaq.com/"):
        raise ValueError("request_url must be a Nasdaq API URL")
    if not retrieved_at_utc.endswith("Z"):
        raise ValueError("retrieved_at_utc must be an ISO-8601 UTC value ending in Z")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response_body)

    metadata = {
        "request_url": request_url,
        "retrieved_at_utc": retrieved_at_utc,
        "raw_path": str(raw_path),
        "response_bytes": len(response_body),
        "sha256": hashlib.sha256(response_body).hexdigest(),
        "source_reference_url": NASDAQ_SOURCE_REFERENCE,
    }
    if metadata_fields:
        metadata.update(metadata_fields)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return raw_path, metadata_path


def _validate_nasdaq_payload(payload: Any) -> dict[str, Any]:
    """Validate the acquisition contract and return a compact summary."""
    if not isinstance(payload, dict):
        raise ValueError("Nasdaq response must be a JSON object")

    status = payload.get("status")
    if not isinstance(status, dict) or status.get("rCode") != 200:
        message = status.get("bCodeMessage") if isinstance(status, dict) else None
        raise ValueError(f"Nasdaq API reported an error: {message or status!r}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Nasdaq response is missing the data object")
    trades_table = data.get("tradesTable")
    if not isinstance(trades_table, dict):
        raise ValueError("Nasdaq response is missing data.tradesTable")
    rows = trades_table.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Nasdaq response is missing data.tradesTable.rows")
    if len(rows) < MINIMUM_RAW_ROWS:
        raise ValueError(
            f"Nasdaq returned {len(rows)} rows; at least {MINIMUM_RAW_ROWS} are required"
        )

    missing_fields = [
        index
        for index, row in enumerate(rows)
        if not isinstance(row, dict) or not REQUIRED_OHLCV_FIELDS <= row.keys()
    ]
    if missing_fields:
        preview = missing_fields[:5]
        raise ValueError(f"Nasdaq rows missing required OHLCV fields: {preview}")

    dates = [row["date"] for row in rows]
    if any(not value for value in dates):
        raise ValueError("Nasdaq response contains an empty date")
    unique_dates = set(dates)
    if len(unique_dates) < MINIMUM_RAW_ROWS:
        raise ValueError(
            f"Nasdaq returned only {len(unique_dates)} unique dates; "
            f"at least {MINIMUM_RAW_ROWS} are required"
        )
    parsed_dates = [datetime.strptime(value, "%m/%d/%Y").date() for value in dates]
    total_records_raw = data.get("totalRecords")
    try:
        total_records = int(str(total_records_raw).replace(",", ""))
    except (TypeError, ValueError) as error:
        raise ValueError("Nasdaq response has an invalid totalRecords value") from error
    if total_records > len(rows):
        raise ValueError(
            f"Nasdaq response is truncated: {total_records} total records but "
            f"only {len(rows)} rows returned; increase the request limit"
        )

    return {
        "row_count": len(rows),
        "unique_date_count": len(unique_dates),
        "duplicate_date_count": len(dates) - len(unique_dates),
        "first_date": min(parsed_dates).isoformat(),
        "last_date": max(parsed_dates).isoformat(),
        "total_records": total_records,
    }


# 1. Data acquisition
def download_data(force: bool = False) -> None:
    """Download and preserve the validated Nasdaq QQQ historical response.

    Input: symbol, asset class, and date range from config.json.
    Output: unchanged JSON response and a separate provenance metadata file.
    Validate HTTP status, JSON schema, coverage, and at least 500 unique dates.
    Existing raw artifacts require an explicit force=True to replace them.
    """
    existing = [
        path for path in (RAW_DATA_PATH, RAW_METADATA_PATH) if path.exists()
    ]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Raw artifact already exists: {names}. "
            "Run with --force only for an intentional refresh."
        )

    config = load_config()
    params = urllib.parse.urlencode(
        {
            "assetclass": config["asset_class"],
            "fromdate": config["start_date"],
            "todate": config["end_date"],
            "limit": DOWNLOAD_LIMIT,
        }
    )
    endpoint = NASDAQ_API_ENDPOINT.format(symbol=config["symbol"])
    url = f"{endpoint}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Nasdaq request failed with HTTP status {response.status}"
                )
            response_body = response.read()
            final_url = response.geturl()
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Nasdaq request failed with HTTP {error.code}: {error.reason}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Nasdaq API: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("Nasdaq request timed out after 30 seconds") from error

    try:
        payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Nasdaq returned a response that is not valid JSON") from error
    summary = _validate_nasdaq_payload(payload)

    retrieved_at_utc = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    save_raw_response(
        response_body,
        request_url=final_url,
        retrieved_at_utc=retrieved_at_utc,
        raw_path=RAW_DATA_PATH,
        metadata_path=RAW_METADATA_PATH,
        force=force,
        metadata_fields={
            "symbol": config["symbol"],
            "asset_class": config["asset_class"],
            "configured_start_date": config["start_date"],
            "configured_end_date": config["end_date"],
            **summary,
        },
    )

    print(
        "Downloaded "
        f"{summary['row_count']} {config['symbol']} daily rows "
        f"({summary['unique_date_count']} unique dates) "
        f"from {summary['first_date']} to {summary['last_date']}."
    )
    print(f"Raw response: {RAW_DATA_PATH}")
    print(f"Metadata: {RAW_METADATA_PATH}")


# 2. Parse and clean
def _numeric(series: pd.Series) -> pd.Series:
    """Parse Nasdaq numeric strings such as ``$123.45`` and ``1,000``."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

def parse_raw(raw_path: Path | None = None) -> pd.DataFrame:
    """Convert Nasdaq JSON into an ascending, unique-date OHLCV DataFrame.

    Return Date (datetime), Open/High/Low/Close (USD), and Volume (shares).
    Drop invalid rows and exact duplicates with counts in frame.attrs['quality'].
    Conflicting same-date quotes require investigation, so fail instead of
    choosing one silently. The raw file is read only.
    """
    raw_path = Path(raw_path) if raw_path is not None else RAW_DATA_PATH
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Nasdaq response not found at {raw_path}. Run with --download first."
        )
    try:
        payload = json.loads(raw_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Raw file is not valid JSON: {raw_path}") from error
    _validate_nasdaq_payload(payload)

    rows = payload["data"]["tradesTable"]["rows"]
    frame = pd.DataFrame(rows).rename(
        columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        }
    )
    frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
    raw_rows = len(frame)
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = _numeric(frame[column])

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    nonfinite = pd.Series(
        np.isinf(frame[numeric_columns].to_numpy(dtype=float)).any(axis=1),
        index=frame.index,
    )
    missing_rows = int(frame.isna().any(axis=1).sum())
    nonpositive_rows = int(
        ((frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
         | (frame["Volume"] < 0)).sum()
    )
    invalid_high_rows = int(
        (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)).sum()
    )
    invalid_low_rows = int(
        (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)).sum()
    )
    invalid = (
        frame.isna().any(axis=1)
        | nonfinite
        | (frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (frame["Volume"] < 0)
        | (frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1))
        | (frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1))
    )
    invalid_rows = int(invalid.sum())
    valid_frame = frame.loc[~invalid]
    distinct_frame = valid_frame.drop_duplicates()
    conflicting_dates = distinct_frame.loc[
        distinct_frame["Date"].duplicated(keep=False), "Date"
    ]
    if not conflicting_dates.empty:
        preview = conflicting_dates.dt.strftime("%Y-%m-%d").unique()[:5].tolist()
        raise ValueError(f"Conflicting OHLCV quotes for the same date: {preview}")
    duplicate_dates = len(valid_frame) - len(distinct_frame)
    frame = (
        distinct_frame
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if len(frame) < MINIMUM_RAW_ROWS:
        raise ValueError(
            f"Only {len(frame)} valid OHLCV rows remain after cleaning; "
            f"at least {MINIMUM_RAW_ROWS} are required"
        )
    if not frame["Date"].is_monotonic_increasing:
        raise AssertionError("Cleaned dates are not in chronological order")

    frame.attrs["quality"] = {
        "raw_rows": raw_rows,
        "duplicate_date_rows_removed": duplicate_dates,
        "invalid_rows_removed": invalid_rows,
        "rows_with_nonfinite_values_removed": int(nonfinite.sum()),
        "rows_with_missing_values_removed": missing_rows,
        "rows_with_nonpositive_price_or_negative_volume_removed": nonpositive_rows,
        "rows_with_invalid_high_removed": invalid_high_rows,
        "rows_with_invalid_low_removed": invalid_low_rows,
        "clean_ohlcv_rows": len(frame),
        "cleaning_policy": (
            "Remove invalid OHLCV and exact duplicates; reject conflicting dates. "
            "Reason counts may overlap; raw_rows = invalid_rows_removed + "
            "duplicate_date_rows_removed + clean_ohlcv_rows."
        ),
    }
    return frame


# 3. Relative Strength Index
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Return RSI in [0, 100] with the supplied example's recursive EWM seed.

    Smooth positive and negative close changes using alpha=1/window. The first
    window changes warm up the indicator; all gains yield 100, all losses 0,
    and a flat price window yields 50. This is not an SMA-seeded Wilder RSI.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("RSI window must be a positive integer")
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    result = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    return result.mask((loss == 0) & (gain == 0), 50.0)


# 4. Feature engineering and target
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV, 21 indicators, next_return, and integer target_up.

    Input must have unique ascending dates. Every feature uses data through t;
    only next_return and target_up use Close[t+1]. Positive next-day returns
    map to 1; zero/negative returns map to 0. Removal counts are in attrs.
    """
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if (
        df.empty or df["Date"].isna().any()
        or not df["Date"].is_monotonic_increasing or df["Date"].duplicated().any()
    ):
        raise ValueError("OHLCV dates must be nonempty, unique, and in chronological order")
    if not np.isfinite(df[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV must contain finite numeric values; call parse_raw first")
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["return_1d"] = close.pct_change(fill_method=None)
    data["return_5d"] = close.pct_change(5, fill_method=None)
    data["gap_return"] = data["Open"] / close.shift(1) - 1
    for window in (5, 10, 20):
        data[f"sma_ratio_{window}"] = close / close.rolling(window).mean() - 1
    data["sma_cross_5_20"] = (
        close.rolling(5).mean() / close.rolling(20).mean() - 1
    )
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema_12 - ema_26
    data["ema_ratio_12"] = close / ema_12 - 1
    data["macd"] = macd_raw / close
    data["macd_signal"] = macd_raw.ewm(span=9, adjust=False).mean() / close
    data["rsi_14"] = rsi(close)
    rolling_low = low.rolling(14).min()
    rolling_high = high.rolling(14).max()
    data["stoch_k_14"] = (
        100 * (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    )
    data["roc_10"] = close.pct_change(10, fill_method=None)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    data["range_pct"] = (high - low) / close
    data["atr_14_pct"] = true_range.rolling(14).mean() / close
    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volatility_20"] = data["return_1d"].rolling(20).std()
    middle_band = close.rolling(20).mean()
    band_sd = close.rolling(20).std()
    data["bb_width_20"] = 4 * band_sd / middle_band

    data["volume_change"] = volume.pct_change(fill_method=None)
    data["volume_ratio_20"] = volume / volume.rolling(20).mean()
    signed_volume = np.sign(close.diff()).fillna(0) * volume
    data["obv_change_5"] = (
        signed_volume.cumsum().diff(5) / volume.rolling(5).mean()
    )

    data["next_return"] = close.shift(-1) / close - 1
    data["target_up"] = (data["next_return"] > 0).astype("Int64")
    data.loc[data["next_return"].isna(), "target_up"] = pd.NA
    data = data.replace([np.inf, -np.inf], np.nan)
    # Twenty previous returns are needed for volatility_20. The final raw
    # observation has no known next-day outcome; neither is an invalid quote.
    positions = np.arange(len(data))
    feature_invalid = data[FEATURES].isna().any(axis=1)
    terminal = data["next_return"].isna()
    warmup = feature_invalid & (positions < 20) & ~terminal
    other_invalid = feature_invalid & ~warmup & ~terminal
    model_data = data.dropna(subset=FEATURES + ["next_return", "target_up"]).copy()
    model_data["target_up"] = model_data["target_up"].astype(int)
    model_data = model_data.reset_index(drop=True)
    if not model_data["Date"].is_monotonic_increasing:
        raise AssertionError("Feature rows are not in chronological order")
    model_data.attrs["quality"] = {
        **df.attrs.get("quality", {}),
        "feature_warmup_rows_removed": int(warmup.sum()),
        "terminal_target_rows_removed": int(terminal.sum()),
        "other_feature_rows_removed": int(other_invalid.sum()),
        "feature_rows": len(model_data),
    }
    return model_data


# 5. Classification metrics
def metric_row(
    name: str, y_true: pd.Series, pred: np.ndarray, prob: np.ndarray
) -> dict[str, float | str]:
    """Return one model's accuracy, balanced accuracy, precision, recall, F1, AUC.

    pred contains class labels; prob contains probabilities for target_up=1.
    AUC is NaN when the evaluation set contains only one observed class.
    """
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true, prob)
            if pd.Series(y_true).nunique() == 2 else np.nan
        ),
    }


def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features))
    return probabilities[:, classes.index(1)]


# 6. Model training (same two-value return contract as the supplied example)
def build_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return (fitted_models, metrics) for the original three fixed models.

    Fit Majority baseline, Logistic regression, and Random forest on X_train
    only. Imputation/scaling are learned within the logistic Pipeline. Score
    X_test via metric_row; run() handles saved predictions and artifacts.
    Hyperparameters are fixed; no search or automatic model selection is used.
    """
    config = config or load_config()
    if X_train.empty or X_test.empty:
        raise ValueError("Both training and test feature matrices must be nonempty")
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training targets must contain both binary classes 0 and 1")
    if {"next_return", "target_up"} & (set(X_train.columns) | set(X_test.columns)):
        raise ValueError("Target and next_return must never appear in model features")
    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config["random_state"],
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=700,
            max_depth=5,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=config["random_state"],
            n_jobs=-1,
        ),
    }
    fitted: dict[str, Any] = {}
    metric_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = _positive_probability(model, X_test)
        fitted[name] = model
        metric_rows.append(metric_row(name, y_test, prediction, probability))
    return fitted, pd.DataFrame(metric_rows)


def _save_figure(path: Path, caption: str | None = None) -> None:
    if caption:
        plt.figtext(0.5, -0.01, caption, ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


# 7. Exploratory data analysis
def save_eda(df: pd.DataFrame) -> None:
    """Save line, histogram, box, scatter, heatmap, and bar charts plus statistics.

    Input is a feature-engineered DataFrame. run() supplies the development
    period only, so exploratory choices cannot use held-out outcomes.
    """
    data = df
    scope = (
        f"Input period: {data['Date'].min():%Y-%m-%d} to "
        f"{data['Date'].max():%Y-%m-%d} | n = {len(data):,} trading days"
    )
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    colors = {
        "navy": "#16324F", "blue": "#2F80ED", "orange": "#F2994A",
        "teal": "#219EBC", "light_blue": "#A9D6E5",
    }

    plt.figure(figsize=(11, 5))
    plt.plot(data["Date"], data["Close"], color=colors["navy"], lw=1.5, label="Close")
    plt.plot(
        data["Date"], data["Close"].rolling(50).mean(),
        color=colors["orange"], label="50-day moving average",
    )
    plt.title("QQQ closing price and 50-day moving average")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    ticks = [data["Date"].min(), *pd.date_range(
        data["Date"].min(), data["Date"].max(), freq="YS"
    ), data["Date"].max()]
    plt.xticks(ticks, [tick.strftime("%Y-%m-%d") for tick in ticks])
    _save_figure(FIGURES_PATH / "01_price_trend.png", scope)

    plt.figure(figsize=(8, 5))
    sns.histplot(data["return_1d"] * 100, bins=55, kde=True, color=colors["blue"])
    plt.axvline(0, color="black", lw=1)
    plt.title("Distribution of QQQ daily returns")
    plt.xlabel("Daily return (%)")
    plt.ylabel("Trading days")
    _save_figure(FIGURES_PATH / "02_return_distribution.png", scope)

    month_data = data.assign(
        month=data["Date"].dt.strftime("%b"), month_number=data["Date"].dt.month,
        return_pct=data["return_1d"] * 100,
    )
    order = month_data.groupby("month")["month_number"].first().sort_values().index
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=month_data, x="month", y="return_pct", order=order,
        color=colors["light_blue"], showfliers=True, fliersize=3,
    )
    plt.axhline(0, color="black", lw=0.8)
    plt.title("Daily return dispersion by calendar month")
    plt.xlabel("Calendar month")
    plt.ylabel("Daily return (%)")
    _save_figure(FIGURES_PATH / "03_monthly_boxplot.png", scope)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        data["rsi_14"], data["next_return"] * 100,
        color=colors["blue"], alpha=0.45, s=18,
    )
    plt.axhline(0, color="black", lw=0.8)
    plt.title("RSI and next-day return")
    plt.xlabel("RSI (14 trading days)")
    plt.ylabel("Next-day return (%)")
    _save_figure(FIGURES_PATH / "04_rsi_scatter.png", scope)

    plt.figure(figsize=(13, 10))
    sns.heatmap(
        data[FEATURES].corr(), cmap="vlag", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.75},
    )
    plt.title("Technical-indicator correlation heatmap")
    _save_figure(FIGURES_PATH / "05_correlation_heatmap.png", scope)

    counts = (
        data["target_up"].map({0: "Down or flat", 1: "Up"})
        .value_counts().reindex(["Down or flat", "Up"], fill_value=0)
    )
    plt.figure(figsize=(6, 4.5))
    axis = counts.plot(kind="bar", color=[colors["teal"], colors["orange"]])
    axis.bar_label(axis.containers[0])
    plt.title("Next-day direction class balance")
    plt.xlabel("Target class")
    plt.ylabel("Trading days")
    plt.xticks(rotation=0)
    _save_figure(FIGURES_PATH / "06_class_balance.png", scope)

    descriptive = data[["Open", "High", "Low", "Close", "Volume", *FEATURES]].describe().T
    descriptive.to_csv(TABLES_PATH / "descriptive_statistics.csv")
    yearly = data.set_index("Date").resample("YE").agg(
        average_return=("return_1d", "mean"),
        volatility=("return_1d", "std"),
        up_rate=("target_up", "mean"),
    )
    yearly.index = yearly.index.year
    yearly.to_csv(TABLES_PATH / "yearly_summary.csv")


def _validate_cached_raw(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Verify that the saved raw response matches its metadata and config."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw Nasdaq response not found at {RAW_DATA_PATH}. "
            "Run --stage download first."
        )
    if not RAW_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Raw provenance metadata is missing; run --stage download to create "
            "a matched raw response and metadata file"
        )

    metadata = json.loads(RAW_METADATA_PATH.read_text(encoding="utf-8"))
    for config_key, metadata_key in (
        ("symbol", "symbol"),
        ("asset_class", "asset_class"),
        ("start_date", "configured_start_date"),
        ("end_date", "configured_end_date"),
    ):
        if config[config_key] != metadata.get(metadata_key):
            raise ValueError(
                f"Cached raw {config_key} does not match config.json; restore the "
                "matching configuration or intentionally refresh with "
                "--stage download --force"
            )

    raw_sha256 = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    if raw_sha256 != metadata.get("sha256"):
        raise ValueError("Raw response checksum does not match its provenance metadata")
    return metadata, raw_sha256


def _save_processed_features(data: pd.DataFrame) -> None:
    """Save the reproducible model-ready derivative, never the raw response."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )


# 8. Workflow orchestration
def run(download: bool = False, force: bool = False) -> dict[str, Any]:
    """Run acquisition through saved tables/figures and return an audit summary.

    Reuse the preserved raw response unless download=True or raw data is absent.
    Reserve the final configured fraction for testing, and omit one boundary
    row so all training labels are known before the first held-out feature day.
    Models use their original fixed parameters. Random Forest CV and held-out
    importance are descriptive; neither step selects a model or tunes it.
    """
    if force and not download:
        raise ValueError("force=True requires download=True")
    config = load_config()

    # 1. Download (or reuse the preserved response when acquisition is omitted).
    if download or not RAW_DATA_PATH.exists():
        download_data(force=force)
    _, raw_sha256 = _validate_cached_raw(config)

    # 2. Parse and clean.
    clean_ohlcv = parse_raw()
    quality = dict(clean_ohlcv.attrs.get("quality", {}))
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    # Save the cleaning evidence even if later modeling fails.
    (TABLES_PATH / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    if quality["invalid_rows_removed"]:
        raise ValueError(
            "Invalid quote rows were removed; inspect data_quality.json before "
            "modeling. Resolve missing sessions so next_return means the next trading day."
        )

    # 3. Feature engineering.
    data = add_features(clean_ohlcv)
    quality.update(data.attrs["quality"])
    label_dates = pd.Series(
        clean_ohlcv["Date"].shift(-1).to_numpy(), index=clean_ohlcv["Date"]
    ).reindex(data["Date"]).reset_index(drop=True)
    split_index = int(len(data) * (1 - config["test_fraction"]))
    # 5 expanding CV folds need six nonempty blocks and one boundary gap.
    if split_index - 1 < 12 or len(data) - split_index < 2:
        raise ValueError("Insufficient train/test rows for a holdout and five time-series CV folds")
    _save_processed_features(data)

    # 4. Exploratory data analysis.
    save_eda(data.iloc[:split_index - 1])

    # 5. Chronological train/test split.
    train = data.iloc[:split_index - 1].copy()
    test = data.iloc[split_index:].copy()
    if train["Date"].max() >= test["Date"].min():
        raise AssertionError("Chronological split overlaps or is out of order")
    if label_dates.iloc[split_index - 2] >= test["Date"].min():
        raise AssertionError("A training label reaches into the held-out period")
    x_train, y_train = train[FEATURES], train["target_up"]
    x_test, y_test = test[FEATURES], test["target_up"]

    # 6-7. Fit the original three fixed models and evaluate the holdout.
    fitted, metrics = build_models(
        x_train, y_train, x_test, y_test, config=config
    )
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLES_PATH / "model_metrics.csv", index=False)
    predictions = pd.DataFrame({
        "Date": test["Date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "target_up": y_test.to_numpy(),
    })
    for name, model in fitted.items():
        slug = name.lower().replace(" ", "_")
        predictions[f"{slug}_prediction"] = model.predict(x_test)
        predictions[f"{slug}_probability_up"] = _positive_probability(model, x_test)
    predictions.to_csv(TABLES_PATH / "test_predictions.csv", index=False)

    forest = fitted["Random forest"]
    forest_prediction = forest.predict(x_test)
    matrix = confusion_matrix(y_test, forest_prediction, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual down_or_flat", "Actual up"],
        columns=["Predicted down_or_flat", "Predicted up"],
    ).to_csv(TABLES_PATH / "confusion_matrix.csv")
    (TABLES_PATH / "classification_report.json").write_text(
        json.dumps(
            classification_report(
                y_test, forest_prediction, output_dict=True, zero_division=0
            ),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    time_series_cv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=CV_GAP)
    cv_result = cross_validate(
        clone(forest), x_train, y_train, cv=time_series_cv,
        scoring=["accuracy", "balanced_accuracy", "f1", "roc_auc"],
        n_jobs=-1, error_score="raise",
    )
    cv_folds = pd.DataFrame({
        key.removeprefix("test_"): value for key, value in cv_result.items()
        if key.startswith("test_")
    })
    cv_folds.insert(0, "fold", range(1, len(cv_folds) + 1))
    # Store temporal boundaries so the expanding-window design can be audited.
    boundaries = []
    for train_indices, validation_indices in time_series_cv.split(x_train):
        label_end = label_dates.iloc[train_indices[-1]]
        validation_start = train.iloc[validation_indices[0]]["Date"]
        if label_end >= validation_start:
            raise AssertionError("A CV training label reaches into its validation period")
        boundaries.append({
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "train_end_date": train.iloc[train_indices[-1]]["Date"].date().isoformat(),
            "train_label_end_date": label_end.date().isoformat(),
            "validation_start_date": validation_start.date().isoformat(),
            "gap_rows": int(validation_indices[0] - train_indices[-1] - 1),
        })
    cv_folds = pd.concat([cv_folds, pd.DataFrame(boundaries)], axis=1)
    cv_folds.to_csv(TABLES_PATH / "timeseries_cv_folds.csv", index=False)
    cv_summary = pd.DataFrame([
        {"metric": score, "mean": cv_folds[score].mean(),
         "standard_deviation": cv_folds[score].std(ddof=1)}
        for score in ("accuracy", "balanced_accuracy", "f1", "roc_auc")
    ])
    cv_summary.to_csv(TABLES_PATH / "timeseries_cv_summary.csv", index=False)

    # 8. Permutation importance on the held-out test period only.
    permutation = permutation_importance(
        forest, x_test, y_test,
        n_repeats=config["permutation_repeats"],
        random_state=config["random_state"],
        scoring="balanced_accuracy", n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance_mean", ascending=False)
    family_lookup = {
        feature: family for family, features in FEATURE_GROUPS.items() for feature in features
    }
    importance["family"] = importance["feature"].map(family_lookup)
    importance.to_csv(TABLES_PATH / "permutation_importance.csv", index=False)
    family_importance = (
        importance.groupby("family", as_index=False)
        .agg(
            importance_mean=("importance_mean", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    family_importance.to_csv(TABLES_PATH / "group_importance.csv", index=False)

    # 9. Save evaluation and explanation figures plus the quality audit.
    plt.figure(figsize=(9, 6))
    top = importance.head(12).sort_values("importance_mean")
    plt.barh(
        top["feature"], top["importance_mean"], xerr=top["importance_std"],
        color="#2F80ED", alpha=0.85,
    )
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Decrease in held-out balanced accuracy")
    plt.ylabel("Feature")
    plt.title("Held-out permutation importance: Random forest")
    _save_figure(
        FIGURES_PATH / "07_feature_importance.png",
        "Error bars: +/- 1 permutation standard deviation (not a confidence interval)",
    )

    plt.figure(figsize=(8, 4.8))
    ordered_family = family_importance.sort_values("importance_mean")
    plt.barh(ordered_family["family"], ordered_family["importance_mean"], color="#219EBC")
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Sum of held-out feature permutation importance")
    plt.ylabel("Feature family")
    plt.title("Indicator family importance: Random forest")
    _save_figure(FIGURES_PATH / "08_group_importance.png")

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Down or flat", "Up"], yticklabels=["Down or flat", "Up"],
    )
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Random forest confusion matrix")
    _save_figure(FIGURES_PATH / "09_confusion_matrix.png")

    plt.figure(figsize=(7, 5))
    axis = plt.gca()
    for name, model in fitted.items():
        if name != "Majority baseline":
            RocCurveDisplay.from_estimator(model, x_test, y_test, name=name, ax=axis)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.title("Out-of-sample ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    _save_figure(FIGURES_PATH / "10_roc_curve.png")

    quality.update(
        {
            "feature_rows": len(data),
            "warmup_or_terminal_rows_removed": (
                quality["feature_warmup_rows_removed"]
                + quality["terminal_target_rows_removed"]
            ),
            "feature_missing_values": int(data[FEATURES].isna().sum().sum()),
            "processed_start_date": data["Date"].min().date().isoformat(),
            "processed_end_date": data["Date"].max().date().isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
            "holdout_gap_rows": 1,
            "holdout_gap_date": data.iloc[split_index - 1]["Date"].date().isoformat(),
            "train_end_date": train["Date"].max().date().isoformat(),
            "train_label_end_date": label_dates.iloc[split_index - 2].date().isoformat(),
            "test_start_date": test["Date"].min().date().isoformat(),
            "positive_class_rate": float(data["target_up"].mean()),
            "split_method": "chronological final holdout",
            "test_fraction_configured": config["test_fraction"],
            "permutation_importance_scope": "held-out test set only",
            "eda_scope": "training period only (holdout and boundary gap excluded)",
            "eda_rows": len(train),
            "cv_model": "Random forest",
            "cv_splits": CV_SPLITS,
            "cv_gap_rows": CV_GAP,
            "modeling_mode": "original three fixed models; no hyperparameter search",
            "permutation_importance_model": "Random forest",
            "permutation_importance_scoring": "balanced_accuracy",
            "holdout_status": "previously inspected; exploratory comparison, not fresh confirmation",
            "raw_sha256": raw_sha256,
            "config": config,
            "family_importance_method": "sum of individual feature means, not joint permutation",
        }
    )
    (TABLES_PATH / "data_quality.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "quality": quality,
        "metrics": metrics.to_dict("records"),
        "top_features": importance.head(5).to_dict("records"),
    }
    print(json.dumps(summary, indent=2))
    return summary


def run_stage(
    stage: str,
    *,
    download: bool = False,
    force: bool = False,
) -> dict[str, Any] | None:
    """Run one learning stage from this file and stop at its output boundary.

    ``parse`` is read-only. ``features`` starts again from the preserved raw
    JSON because separate command-line runs cannot share an in-memory
    DataFrame. ``all`` keeps the original end-to-end behavior.
    """
    valid_stages = {"download", "parse", "features", "all"}
    if stage not in valid_stages:
        raise ValueError(f"Unknown stage {stage!r}; choose one of {sorted(valid_stages)}")
    if download and stage != "all":
        raise ValueError("download=True is valid only for stage='all'")
    if force and not (stage == "download" or (stage == "all" and download)):
        raise ValueError(
            "force=True requires stage='download' or stage='all' with download=True"
        )

    # Download only: write the unchanged Nasdaq response and its metadata, then stop.
    if stage == "download":
        download_data(force=force)
        return None

    # Full workflow: preserve the original pipeline behavior and command flags.
    if stage == "all":
        return run(download=download, force=force)

    config = load_config()
    _, raw_sha256 = _validate_cached_raw(config)

    # Parse only: clean and audit in memory. This stage does not write a CSV.
    clean_ohlcv = parse_raw()
    quality = dict(clean_ohlcv.attrs.get("quality", {}))
    parse_summary: dict[str, Any] = {
        "stage": "parse",
        "rows": len(clean_ohlcv),
        "start_date": clean_ohlcv["Date"].min().date().isoformat(),
        "end_date": clean_ohlcv["Date"].max().date().isoformat(),
        "dtypes": {column: str(dtype) for column, dtype in clean_ohlcv.dtypes.items()},
        "quality": quality,
        "raw_sha256": raw_sha256,
    }
    if stage == "parse":
        print(json.dumps(parse_summary, indent=2))
        return parse_summary

    # Feature stage: reject broken daily rows before indicators can bridge a gap.
    if quality.get("invalid_rows_removed", 0):
        raise ValueError(
            "Invalid quote rows were removed; inspect the parse-stage quality output "
            "before creating features"
        )
    feature_data = add_features(clean_ohlcv)
    if feature_data.empty:
        raise ValueError(
            "Feature engineering produced no model-ready rows; inspect zero/constant "
            "price or volume series and the feature-removal audit"
        )
    _save_processed_features(feature_data)
    feature_quality = dict(feature_data.attrs.get("quality", {}))
    feature_summary = {
        "stage": "features",
        "rows": len(feature_data),
        "columns": len(feature_data.columns),
        "technical_indicator_count": len(FEATURES),
        "start_date": feature_data["Date"].min().date().isoformat(),
        "end_date": feature_data["Date"].max().date().isoformat(),
        "processed_path": str(PROCESSED_DATA_PATH),
        "quality": feature_quality,
        "raw_sha256": raw_sha256,
    }
    print(json.dumps(feature_summary, indent=2))
    return feature_summary


def main() -> None:
    """Command-line entry point for one stage or the complete pipeline."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("download", "parse", "features", "all"),
        default="all",
        help="run one learning stage and stop (default: all)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download Nasdaq data before --stage all (legacy end-to-end option)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow an intentional replacement of existing raw artifacts",
    )
    args = parser.parse_args()
    if args.download and args.stage != "all":
        parser.error("--download is valid only with --stage all; use --stage download")
    if args.force and not (
        args.stage == "download" or (args.stage == "all" and args.download)
    ):
        parser.error(
            "--force requires --stage download, or --stage all together with --download"
        )
    run_stage(args.stage, download=args.download, force=args.force)


if __name__ == "__main__":
    main()
