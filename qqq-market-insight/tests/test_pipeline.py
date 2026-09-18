"""Tests for the QQQ feature engineering and modeling pipeline."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.pipeline as pipeline
from src.pipeline import (
    FEATURES,
    _validate_nasdaq_payload,
    add_features,
    load_config,
    save_raw_response,
)


class PipelineImportTest(unittest.TestCase):
    """Basic project-setup smoke test."""

    def test_pipeline_module_imports(self):
        """The pipeline module must remain importable while it is developed."""
        import src.pipeline  # noqa: F401

    def test_project_config_is_valid(self):
        config = load_config()
        self.assertEqual(config["symbol"], "QQQ")
        self.assertGreater(config["test_fraction"], 0)
        self.assertLess(config["test_fraction"], 1)

    def test_config_accepts_full_line_comments_without_changing_string_values(self):
        expected = {
            "symbol": "QQQ",
            "asset_class": "etf",
            "start_date": "2021-01-01",
            "end_date": "2026-09-16",
            "test_fraction": 0.20,
            "random_state": 42,
            "permutation_repeats": 30,
            "source_reference": "https://www.nasdaq.com/market-activity/etf/qqq/historical",
        }
        document = json.dumps(expected, indent=2).replace(
            "{\n", "{\n  // ค่าตั้งต้นของโครงการ\n", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text("// project configuration\n\n" + document, encoding="utf-8")
            self.assertEqual(load_config(config_path), expected)

    def test_raw_response_is_exact_and_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.json"
            metadata_path = root / "raw.metadata.json"
            body = b'{"data":{"example":true}}\n'
            request_url = "https://api.nasdaq.com/api/quote/QQQ/historical"

            save_raw_response(
                body,
                request_url=request_url,
                retrieved_at_utc="2026-09-17T00:00:00Z",
                raw_path=raw_path,
                metadata_path=metadata_path,
            )
            self.assertEqual(raw_path.read_bytes(), body)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["request_url"], request_url)
            self.assertEqual(metadata["response_bytes"], len(body))

            with self.assertRaises(FileExistsError):
                save_raw_response(
                    body,
                    request_url=request_url,
                    retrieved_at_utc="2026-09-17T00:00:00Z",
                    raw_path=raw_path,
                    metadata_path=metadata_path,
                )

    def test_validates_at_least_500_unique_ohlcv_rows(self):
        rows = []
        start = date(2024, 1, 1)
        for offset in range(500):
            day = start + timedelta(days=offset)
            rows.append(
                {
                    "date": day.strftime("%m/%d/%Y"),
                    "open": "1.00",
                    "high": "2.00",
                    "low": "0.50",
                    "close": "1.50",
                    "volume": "1,000",
                }
            )
        payload = {
            "data": {"totalRecords": 500, "tradesTable": {"rows": rows}},
            "status": {"rCode": 200},
        }
        summary = _validate_nasdaq_payload(payload)
        self.assertEqual(summary["row_count"], 500)
        self.assertEqual(summary["unique_date_count"], 500)

    def test_download_preserves_mock_response_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "qqq_nasdaq_raw.json"
            metadata_path = root / "qqq_nasdaq_raw.metadata.json"
            rows = []
            start = date(2024, 1, 1)
            for offset in range(500):
                day = start + timedelta(days=offset)
                rows.append(
                    {
                        "date": day.strftime("%m/%d/%Y"),
                        "open": "1.00",
                        "high": "2.00",
                        "low": "0.50",
                        "close": "1.50",
                        "volume": "1,000",
                    }
                )
            body = json.dumps(
                {
                    "data": {
                        "totalRecords": 500,
                        "tradesTable": {"rows": rows},
                    },
                    "status": {"rCode": 200},
                }
            ).encode()

            class FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    return body

                def geturl(self):
                    return "https://api.nasdaq.com/api/quote/QQQ/historical?test=1"

            with (
                patch.object(pipeline, "RAW_DATA_PATH", raw_path),
                patch.object(pipeline, "RAW_METADATA_PATH", metadata_path),
                patch("urllib.request.urlopen", return_value=FakeResponse()) as opener,
            ):
                pipeline.download_data()

            self.assertEqual(raw_path.read_bytes(), body)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["row_count"], 500)
            request = opener.call_args.args[0]
            self.assertIn("assetclass=etf", request.full_url)
            self.assertIn("limit=5000", request.full_url)

    def test_features_are_chronological_and_target_uses_next_close(self):
        frame = self.feature_frame()
        featured = add_features(frame)
        self.assertTrue(featured["Date"].is_monotonic_increasing)
        self.assertEqual(int(featured[FEATURES].isna().sum().sum()), 0)
        expected = frame.set_index("Date")["Close"].shift(-1) / frame.set_index("Date")["Close"] - 1
        expected = expected.loc[featured["Date"]].to_numpy()
        np.testing.assert_allclose(featured["next_return"], expected)
        np.testing.assert_array_equal(featured["target_up"], (expected > 0).astype(int))
        self.assertTrue((expected > 0).any())
        self.assertTrue((expected < 0).any())
        self.assertTrue((expected == 0).any())
        self.assertNotIn(frame["Date"].iloc[-1], set(featured["Date"]))

    @staticmethod
    def feature_frame(periods=80):
        close = pd.Series(100 + np.cumsum(np.resize([1.5, -0.8, 0, 0.3, -0.4], periods)))
        return pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
                "Open": close - 0.2,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000 + np.arange(periods) * 1_000,
            }
        )

    @staticmethod
    def raw_payload():
        rows = [
            {
                "date": day.strftime("%m/%d/%Y"),
                "open": "$1.00", "high": "$2.00", "low": "$0.50",
                "close": "$1.50", "volume": "1,000",
            }
            for day in pd.date_range("2023-01-02", periods=502, freq="B")
        ]
        return {"data": {"totalRecords": len(rows), "tradesTable": {"rows": rows}},
                "status": {"rCode": 200}}

    def parse_payload(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            return pipeline.parse_raw(raw_path)

    def test_parse_removes_identical_duplicates_but_rejects_conflicts(self):
        payload = self.raw_payload()
        rows = payload["data"]["tradesTable"]["rows"]
        rows.append(dict(rows[0]))
        rows.reverse()
        payload["data"]["totalRecords"] = len(rows)
        parsed = self.parse_payload(payload)
        self.assertEqual(len(parsed), 502)
        self.assertTrue(parsed["Date"].is_monotonic_increasing)
        self.assertEqual(parsed.attrs["quality"]["duplicate_date_rows_removed"], 1)
        rows[0]["close"] = "$1.75"
        with self.assertRaisesRegex(ValueError, "(?i)conflicting"):
            self.parse_payload(payload)

    def test_parse_audits_nonfinite_ohlcv(self):
        payload = self.raw_payload()
        row = payload["data"]["tradesTable"]["rows"][100]
        row.update(open="inf", high="inf")
        parsed = self.parse_payload(payload)
        self.assertEqual(len(parsed), 501)
        self.assertEqual(parsed.attrs["quality"]["rows_with_nonfinite_values_removed"], 1)
        self.assertTrue(np.isfinite(parsed[["Open", "High", "Low", "Close", "Volume"]]).all().all())

    def test_features_reject_unsorted_and_duplicate_dates(self):
        frame = self.feature_frame()
        with patch.object(pd.Series, "pct_change", side_effect=AssertionError("Transforms ran before date validation")):
            with self.assertRaisesRegex(ValueError, "(?i)chronolog|sort|order"):
                add_features(frame.iloc[::-1])
            frame.loc[30, "Date"] = frame.loc[29, "Date"]
            with self.assertRaisesRegex(ValueError, "(?i)duplicat|unique"):
                add_features(frame)

    def test_future_observations_cannot_change_past_features(self):
        original = self.feature_frame()
        changed = original.copy()
        changed.loc[50:, ["Open", "High", "Low", "Close"]] *= 1.25
        changed.loc[50:, "Volume"] *= 2
        before = add_features(original).set_index("Date")
        after = add_features(changed).set_index("Date")
        past_dates = before.index[before.index < original.loc[50, "Date"]]
        pd.testing.assert_frame_equal(before.loc[past_dates, FEATURES], after.loc[past_dates, FEATURES])

    def test_feature_row_removals_are_reconciled_by_reason(self):
        frame = self.feature_frame()
        frame.loc[40, "Volume"] = 0
        featured = add_features(frame)
        quality = featured.attrs["quality"]
        self.assertEqual(quality["feature_warmup_rows_removed"], 20)
        self.assertEqual(quality["terminal_target_rows_removed"], 1)
        self.assertEqual(quality["other_feature_rows_removed"], 1)
        removed = sum(quality[key] for key in (
            "feature_warmup_rows_removed", "terminal_target_rows_removed", "other_feature_rows_removed"
        ))
        self.assertEqual(len(frame), len(featured) + removed)

    def test_rsi_handles_rising_falling_and_flat_prices(self):
        for prices, expected in (
            (pd.Series(np.arange(40, dtype=float)), 100.0),
            (pd.Series(np.arange(40, 0, -1, dtype=float)), 0.0),
            (pd.Series(np.full(40, 100.0)), 50.0),
        ):
            with self.subTest(expected=expected):
                result = pipeline.rsi(prices)
                self.assertTrue(result.iloc[:14].isna().all())
                np.testing.assert_allclose(result.iloc[14:], expected)

    def test_build_models_matches_example_contract_and_fits_scaler_on_train(self):
        x_train, y_train, x_test, y_test = self.model_frames()
        fitted, metrics = pipeline.build_models(x_train, y_train, x_test, y_test)
        expected_names = {"Majority baseline", "Logistic regression", "Random forest"}
        self.assertEqual(set(fitted), expected_names)
        self.assertEqual(set(metrics["model"]), expected_names)
        self.assertTrue({"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"} <= set(metrics))
        baseline = fitted["Majority baseline"]
        self.assertIsInstance(baseline, DummyClassifier)
        self.assertEqual(baseline.strategy, "most_frequent")

        logistic = fitted["Logistic regression"]
        self.assertIsInstance(logistic, Pipeline)
        self.assertIsInstance(logistic.named_steps["imputer"], SimpleImputer)
        self.assertEqual(logistic.named_steps["imputer"].strategy, "median")
        scaler = logistic.named_steps["scale"]
        self.assertIsInstance(scaler, StandardScaler)
        np.testing.assert_allclose(scaler.mean_, x_train.mean().to_numpy())
        self.assertEqual(int(scaler.n_samples_seen_), len(x_train))
        estimator = logistic.named_steps["model"]
        self.assertIsInstance(estimator, LogisticRegression)
        self.assertEqual(estimator.C, 1.0)
        self.assertEqual(estimator.class_weight, "balanced")
        self.assertEqual(estimator.max_iter, 2000)
        self.assertEqual(estimator.random_state, load_config()["random_state"])

        forest = fitted["Random forest"]
        self.assertIsInstance(forest, RandomForestClassifier)
        self.assertEqual(forest.n_estimators, 700)
        self.assertEqual(forest.max_depth, 5)
        self.assertEqual(forest.min_samples_leaf, 12)
        self.assertEqual(forest.class_weight, "balanced_subsample")
        self.assertEqual(forest.random_state, load_config()["random_state"])
        self.assertEqual(forest.n_jobs, -1)

    def test_fitted_models_do_not_depend_on_heldout_labels(self):
        x_train, y_train, x_test, y_test = self.model_frames()
        original, _ = pipeline.build_models(x_train, y_train, x_test, y_test)
        changed, _ = pipeline.build_models(x_train, y_train, x_test, 1 - y_test)
        for name in original:
            with self.subTest(model=name):
                np.testing.assert_array_equal(original[name].predict(x_test), changed[name].predict(x_test))
                np.testing.assert_allclose(
                    original[name].predict_proba(x_test), changed[name].predict_proba(x_test)
                )

    def test_parse_stage_is_read_only_and_does_not_download(self):
        clean = self.feature_frame()
        clean.attrs["quality"] = {"invalid_rows_removed": 0}
        with (
            patch.object(pipeline, "_validate_cached_raw", return_value=({}, "abc123")),
            patch.object(pipeline, "parse_raw", return_value=clean) as parse,
            patch.object(pipeline, "download_data") as download,
            patch.object(pipeline, "add_features") as add,
            patch.object(pipeline, "_save_processed_features") as save,
            patch("builtins.print"),
        ):
            summary = pipeline.run_stage("parse")

        parse.assert_called_once_with()
        download.assert_not_called()
        add.assert_not_called()
        save.assert_not_called()
        self.assertEqual(summary["stage"], "parse")
        self.assertEqual(summary["rows"], len(clean))

    def test_feature_stage_saves_model_ready_csv_without_running_analysis(self):
        clean = self.feature_frame()
        clean.attrs["quality"] = {"invalid_rows_removed": 0}
        with tempfile.TemporaryDirectory() as directory:
            processed_path = Path(directory) / "processed" / "qqq_features.csv"
            with (
                patch.object(pipeline, "PROCESSED_DATA_PATH", processed_path),
                patch.object(pipeline, "_validate_cached_raw", return_value=({}, "abc123")),
                patch.object(pipeline, "parse_raw", return_value=clean),
                patch.object(pipeline, "download_data") as download,
                patch.object(pipeline, "save_eda") as save_eda,
                patch.object(pipeline, "build_models") as build_models,
                patch("builtins.print"),
            ):
                summary = pipeline.run_stage("features")

            saved = pd.read_csv(processed_path)
            download.assert_not_called()
            save_eda.assert_not_called()
            build_models.assert_not_called()
            self.assertEqual(summary["stage"], "features")
            self.assertEqual(len(saved), summary["rows"])
            self.assertTrue({"Open", "High", "Low", "Close", "Volume", "next_return", "target_up"} <= set(saved))
            self.assertTrue(set(FEATURES) <= set(saved))
            self.assertNotIn("Unnamed: 0", saved.columns)

    def test_stage_flags_protect_raw_and_reject_ambiguous_combinations(self):
        with self.assertRaisesRegex(ValueError, "force=True"):
            pipeline.run_stage("parse", force=True)
        with self.assertRaisesRegex(ValueError, "download=True"):
            pipeline.run_stage("features", download=True)
        with patch.object(pipeline, "download_data") as download:
            pipeline.run_stage("download", force=True)
        download.assert_called_once_with(force=True)

    @staticmethod
    def model_frames():
        x_train = pd.DataFrame({"a": np.arange(60, dtype=float), "b": np.sin(np.arange(60))})
        y_train = pd.Series(np.resize([0, 1], 60))
        x_test = pd.DataFrame({"a": np.arange(1000, 1020, dtype=float), "b": np.cos(np.arange(20))})
        y_test = pd.Series(np.resize([0, 1], 20))
        return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    unittest.main()
