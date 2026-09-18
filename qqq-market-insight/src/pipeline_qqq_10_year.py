import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from argparse import ArgumentParser
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# 1. ตั้งค่า Paths (อ้างอิงจากโฟลเดอร์ปัจจุบันของ File)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
RAW_DATA_OHLCV_10_YEAR_PATH = (
    PROJECT_ROOT / "data" / "raw" / "qqq_nasdaq_ohlcv_10_year_raw.json"
)
RAW_METADATA_OHLCV_10_YEAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "qqq_nasdaq_ohlcv_10_year_raw.metadata.json"
)

# 2. ตั้งค่า Constants สำหรับ API
NASDAQ_API_ENDPOINT = "https://api.nasdaq.com/api/quote/{symbol}/historical"
NASDAQ_SOURCE_REFERENCE = (
    "https://www.nasdaq.com/market-activity/etf/qqq/historical"
)

MINIMUM_RAW_ROWS = 500
DOWNLOAD_LIMIT = 5000
DOWNLOAD_TIMEOUT_SECONDS = 30
REQUIRED_OHLCV_FIELDS = {"date", "open", "high", "low", "close", "volume"}
REQUIRED_CONFIG_KEYS = {
    "symbol",
    "asset_class",
    "start_date",
    "end_date",
    "test_fraction",
    "random_state",
    "permutation_repeats",
}

# 3. ฟังก์ชันอ่าน Config
def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load project parameters, allowing optional full-line // comments."""
    with path.open(encoding="utf-8") as config_file:
        # json.loads() receives text. json.load() would require an open file object.
        config = json.loads(
            "\n".join(
                line
                for line in config_file
                if not line.lstrip().startswith("//")
            )
        )

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


# 4. ฟังก์ชันบันทึกไฟล์ (พร้อมทำ Metadata ควบคุม Version)
def save_raw_response(
    response_body: bytes,
    *,
    request_url: str,
    retrieved_at_utc: str,
    raw_path: Path = RAW_DATA_OHLCV_10_YEAR_PATH,
    metadata_path: Path = RAW_METADATA_OHLCV_10_YEAR_PATH,
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
            "Use --force only for an intentional refresh."
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


# 5. ฟังก์ชันตรวจสอบความถูกต้องของข้อมูล (Validation)
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
    try:
        parsed_dates = [
            datetime.strptime(value, "%m/%d/%Y").date() for value in dates
        ]
    except (TypeError, ValueError) as error:
        raise ValueError("Nasdaq response contains an invalid date value") from error

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


# 6. Data acquisition ฟังก์ชันหลักสำหรับดึงข้อมูล
def download_data(force: bool = False) -> None:
    """Download and preserve the validated Nasdaq QQQ historical response.

    Input: symbol, asset class, and date range from config.json.
    Output: unchanged JSON response and a separate provenance metadata file.
    Validate HTTP status, JSON schema, coverage, and at least 500 unique dates.
    Existing raw artifacts require an explicit force=True to replace them.
    """
    existing = [
        path
        for path in (
            RAW_DATA_OHLCV_10_YEAR_PATH,
            RAW_METADATA_OHLCV_10_YEAR_PATH,
        )
        if path.exists()
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
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
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
        raise RuntimeError(
            f"Nasdaq request timed out after {DOWNLOAD_TIMEOUT_SECONDS} seconds"
        ) from error

    try:
        payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Nasdaq returned a response that is not valid JSON") from error
    summary = _validate_nasdaq_payload(payload)

    requested_start = date.fromisoformat(config["start_date"])
    requested_end = date.fromisoformat(config["end_date"])
    response_start = date.fromisoformat(summary["first_date"])
    response_end = date.fromisoformat(summary["last_date"])
    if response_start < requested_start or response_end > requested_end:
        raise ValueError(
            "Nasdaq response contains dates outside the configured date range"
        )

    retrieved_at_utc = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    save_raw_response(
        response_body,
        request_url=final_url,
        retrieved_at_utc=retrieved_at_utc,
        raw_path=RAW_DATA_OHLCV_10_YEAR_PATH,
        metadata_path=RAW_METADATA_OHLCV_10_YEAR_PATH,
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
    print(f"Raw response: {RAW_DATA_OHLCV_10_YEAR_PATH}")
    print(f"Metadata: {RAW_METADATA_OHLCV_10_YEAR_PATH}")


# ==========================================
# คำสั่งรัน
# ==========================================
def main() -> None:
    """CLI สำหรับดาวน์โหลดข้อมูล โดยไม่ทำงานอัตโนมัติเมื่อ import module."""
    parser = ArgumentParser(
        description="Download and validate QQQ daily OHLCV data from Nasdaq"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing raw JSON and metadata intentionally",
    )
    args = parser.parse_args()
    download_data(force=args.force)


if __name__ == "__main__":
    main()
