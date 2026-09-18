"""Focused tests for the standalone Nasdaq downloader in pipeline_qqq.py."""

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.pipeline_qqq as downloader


class PipelineQqqDownloaderTest(unittest.TestCase):
    """Validate the focused downloader without making real network requests."""

    def test_five_year_raw_paths_are_separate_from_the_original_dataset(self):
        self.assertEqual(
            downloader.RAW_DATA_OHLCV_5_YEAR_PATH.name,
            "qqq_nasdaq_ohlcv_5_year_raw.json",
        )
        self.assertEqual(
            downloader.RAW_METADATA_OHLCV_5_YEAR_PATH.name,
            "qqq_nasdaq_ohlcv_5_year_raw.metadata.json",
        )
        self.assertNotEqual(
            downloader.RAW_DATA_OHLCV_5_YEAR_PATH.name,
            "qqq_nasdaq_raw.json",
        )

    @staticmethod
    def payload_bytes(row_count: int = 500) -> bytes:
        start = date(2024, 1, 1)
        rows = [
            {
                "date": (start + timedelta(days=offset)).strftime("%m/%d/%Y"),
                "open": "1.00",
                "high": "2.00",
                "low": "0.50",
                "close": "1.50",
                "volume": "1,000",
            }
            for offset in range(row_count)
        ]
        return json.dumps(
            {
                "data": {
                    "totalRecords": row_count,
                    "tradesTable": {"rows": rows},
                },
                "status": {"rCode": 200},
            }
        ).encode()

    def test_config_with_comments_loads(self):
        config = downloader.load_config()
        self.assertEqual(config["symbol"], "QQQ")
        self.assertLessEqual(config["start_date"], config["end_date"])

    def test_download_saves_exact_response_metadata_and_expected_request(self):
        body = self.payload_bytes()

        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return body

            def geturl(self):
                return "https://api.nasdaq.com/api/quote/QQQ/historical?final=1"

        test_config = {
            "symbol": "QQQ",
            "asset_class": "etf",
            "start_date": "2024-01-01",
            "end_date": "2025-05-14",
            "test_fraction": 0.20,
            "random_state": 42,
            "permutation_repeats": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.json"
            metadata_path = root / "raw.metadata.json"
            with (
                patch.object(downloader, "RAW_DATA_OHLCV_5_YEAR_PATH", raw_path),
                patch.object(
                    downloader,
                    "RAW_METADATA_OHLCV_5_YEAR_PATH",
                    metadata_path,
                ),
                patch.object(downloader, "load_config", return_value=test_config),
                patch(
                    "urllib.request.urlopen", return_value=FakeResponse()
                ) as urlopen,
                patch("builtins.print"),
            ):
                downloader.download_data()

            self.assertEqual(raw_path.read_bytes(), body)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["row_count"], 500)
            self.assertEqual(metadata["unique_date_count"], 500)
            self.assertEqual(metadata["sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(
                metadata["request_url"],
                "https://api.nasdaq.com/api/quote/QQQ/historical?final=1",
            )

            request = urlopen.call_args.args[0]
            self.assertIn("assetclass=etf", request.full_url)
            self.assertIn("fromdate=2024-01-01", request.full_url)
            self.assertIn("todate=2025-05-14", request.full_url)
            self.assertIn("limit=5000", request.full_url)
            self.assertEqual(request.get_header("User-agent"), "Mozilla/5.0")
            self.assertEqual(request.get_header("Accept"), "application/json")
            self.assertEqual(
                urlopen.call_args.kwargs["timeout"],
                downloader.DOWNLOAD_TIMEOUT_SECONDS,
            )

    def test_existing_raw_stops_before_network_unless_force_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.json"
            metadata_path = root / "raw.metadata.json"
            raw_path.write_bytes(b"existing")
            with (
                patch.object(downloader, "RAW_DATA_OHLCV_5_YEAR_PATH", raw_path),
                patch.object(
                    downloader,
                    "RAW_METADATA_OHLCV_5_YEAR_PATH",
                    metadata_path,
                ),
                patch("urllib.request.urlopen") as urlopen,
            ):
                with self.assertRaisesRegex(FileExistsError, "--force"):
                    downloader.download_data()
            urlopen.assert_not_called()
            self.assertEqual(raw_path.read_bytes(), b"existing")

    def test_network_failure_is_clear_and_does_not_create_raw_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.json"
            metadata_path = root / "raw.metadata.json"
            with (
                patch.object(downloader, "RAW_DATA_OHLCV_5_YEAR_PATH", raw_path),
                patch.object(
                    downloader,
                    "RAW_METADATA_OHLCV_5_YEAR_PATH",
                    metadata_path,
                ),
                patch(
                    "urllib.request.urlopen",
                    side_effect=downloader.urllib.error.URLError("offline"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not reach Nasdaq API"):
                    downloader.download_data()
            self.assertFalse(raw_path.exists())
            self.assertFalse(metadata_path.exists())

    def test_cli_passes_force_without_downloading_during_import(self):
        with (
            patch.object(sys, "argv", ["pipeline_qqq.py", "--force"]),
            patch.object(downloader, "download_data") as download,
        ):
            downloader.main()
        download.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
