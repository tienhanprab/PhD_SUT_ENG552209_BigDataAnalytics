from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from solar_apriori import analysis


def _write_source_csvs(raw_directory: Path) -> None:
    generation_rows = {
        1: [
            ["15-05-2020 06:00", 1, "P1-A", 10.0, 1.0, 1.0, 101.0],
            ["15-05-2020 06:00", 1, "P1-B", 20.0, 3.0, 2.0, 202.0],
            ["15-05-2020 06:15", 1, "P1-A", 5.0, 0.5, 1.5, 101.5],
        ],
        2: [
            ["2020-05-15 06:00:00", 2, "P2-A", 50.0, 5.0, 3.0, 303.0],
            ["2020-05-15 06:00:00", 2, "P2-B", 70.0, 7.0, 4.0, 404.0],
        ],
    }
    weather_rows = {
        1: [
            ["15-05-2020 06:00", 1, "P1-W", 25.0, 30.0, 0.50],
            ["15-05-2020 06:15", 1, "P1-W", 26.0, 31.0, 0.60],
        ],
        2: [
            ["2020-05-15 06:00:00", 2, "P2-W", 35.0, 40.0, 0.75],
        ],
    }

    raw_directory.mkdir()
    for plant_number in analysis.PLANTS:
        generation = pd.DataFrame(
            generation_rows[plant_number], columns=analysis.GENERATION_COLUMNS
        )
        weather = pd.DataFrame(
            weather_rows[plant_number], columns=analysis.WEATHER_COLUMNS
        )
        generation.to_csv(
            raw_directory / f"Plant_{plant_number}_Generation_Data.csv", index=False
        )
        weather.to_csv(
            raw_directory / f"Plant_{plant_number}_Weather_Sensor_Data.csv", index=False
        )


def _binning_frame(plant_values: dict[int, list[float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plant_number, values in plant_values.items():
        for value in values:
            rows.append(
                {
                    "PLANT_NUMBER": plant_number,
                    "IRRADIATION": value,
                    "AMBIENT_TEMPERATURE": value,
                    "MODULE_TEMPERATURE": value,
                    "AC_POWER_MEAN": value,
                }
            )
    return pd.DataFrame(rows)


def test_parse_mixed_timestamp_accepts_both_formats_and_rejects_unknown() -> None:
    values = pd.Series(["15-05-2020 06:00", "2020-05-16 07:15:00"])

    parsed = analysis.parse_mixed_timestamp(values)

    assert parsed.tolist() == [
        pd.Timestamp("2020-05-15 06:00:00"),
        pd.Timestamp("2020-05-16 07:15:00"),
    ]
    with pytest.raises(ValueError, match="not-a-timestamp"):
        analysis.parse_mixed_timestamp(pd.Series(["not-a-timestamp"]))


def test_prepare_plant_time_data_aggregates_before_exact_weather_join(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    _write_source_csvs(raw_directory)

    prepared = analysis.prepare_plant_time_data(raw_directory)
    plant_time = prepared.plant_time

    assert len(plant_time) == 3
    assert not plant_time.duplicated(["PLANT_ID", "TIMESTAMP"]).any()

    plant_1_at_0600 = plant_time.loc[
        plant_time["PLANT_NUMBER"].eq(1)
        & plant_time["TIMESTAMP"].eq(pd.Timestamp("2020-05-15 06:00:00"))
    ].iloc[0]
    assert plant_1_at_0600["INVERTER_COUNT"] == 2
    assert plant_1_at_0600["EXPECTED_INVERTERS"] == 2
    assert plant_1_at_0600["INVERTER_COVERAGE"] == pytest.approx(1.0)
    assert plant_1_at_0600["DC_POWER_SUM"] == pytest.approx(30.0)
    assert plant_1_at_0600["DC_POWER_MEAN"] == pytest.approx(15.0)
    assert plant_1_at_0600["AC_POWER_SUM"] == pytest.approx(4.0)
    assert plant_1_at_0600["AC_POWER_MEAN"] == pytest.approx(2.0)
    assert plant_1_at_0600["IRRADIATION"] == pytest.approx(0.50)

    incomplete = plant_time.loc[
        plant_time["PLANT_NUMBER"].eq(1)
        & plant_time["TIMESTAMP"].eq(pd.Timestamp("2020-05-15 06:15:00"))
    ].iloc[0]
    assert incomplete["INVERTER_COUNT"] == 1
    assert incomplete["INVERTER_COVERAGE"] == pytest.approx(0.5)

    plant_1_merge = prepared.merge_quality.loc[
        prepared.merge_quality["Plant"].eq(1)
    ].iloc[0]
    assert plant_1_merge["Generation timestamps"] == 2
    assert plant_1_merge["Weather timestamps"] == 2
    assert plant_1_merge["Matched timestamps"] == 2

    plant_1_coverage = prepared.inverter_coverage.loc[
        prepared.inverter_coverage["Plant"].eq(1)
    ].iloc[0]
    assert plant_1_coverage["Complete timestamps"] == 1
    assert plant_1_coverage["Incomplete timestamps"] == 1
    assert plant_1_coverage["Minimum observed inverters"] == 1


def test_select_daylight_population_applies_all_filters_and_audits_counts() -> None:
    frame = pd.DataFrame(
        {
            "ROW_ID": ["night", "low-coverage", "selected", "missing"],
            "IRRADIATION": [0.0, 0.2, 0.3, 0.4],
            "AMBIENT_TEMPERATURE": [20.0, 21.0, 22.0, float("nan")],
            "MODULE_TEMPERATURE": [20.0, 22.0, 24.0, 26.0],
            "AC_POWER_MEAN": [0.0, 1.0, 2.0, 3.0],
            "INVERTER_COVERAGE": [1.0, 0.4, 0.5, 1.0],
        }
    )

    selected, audit = analysis.select_daylight_population(
        frame, minimum_irradiation=0.1, minimum_inverter_coverage=0.5
    )

    assert selected["ROW_ID"].tolist() == ["selected"]
    assert audit["Rows"].tolist() == [4, 3, 2, 1, 1]


def test_split_dates_is_chronological_and_keeps_whole_dates_together() -> None:
    frame = pd.DataFrame(
        {
            "DATE": pd.to_datetime(
                [
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-04",
                    "2020-01-02",
                    "2020-01-01",
                ]
            ),
            "VALUE": [3, 1, 4, 2, 10],
        }
    )

    discovery, validation, summary = analysis.split_dates(
        frame, discovery_fraction=0.5
    )

    assert set(discovery["DATE"]) == {
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
    }
    assert set(validation["DATE"]) == {
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-04"),
    }
    assert discovery["DATE"].max() < validation["DATE"].min()
    assert set(discovery["DATE"]).isdisjoint(set(validation["DATE"]))
    assert summary["Distinct dates"].tolist() == [2, 2]
    assert summary["Transactions"].tolist() == [3, 2]


def test_frozen_per_plant_tertiles_create_four_item_transactions() -> None:
    discovery = _binning_frame({1: [0.0, 10.0, 20.0], 2: [100.0, 200.0, 300.0]})
    validation = _binning_frame({1: [-10.0, 10.0, 30.0], 2: [50.0, 200.0, 350.0]})

    edges = analysis.fit_tertile_edges(discovery)
    frozen_edges = edges.copy(deep=True)
    binned = analysis.apply_tertile_edges(validation, edges)
    with_transactions = analysis.add_transactions(binned)

    pd.testing.assert_frame_equal(edges, frozen_edges)
    assert len(edges) == len(analysis.PLANTS) * len(analysis.BINNING_VARIABLES)
    assert binned.loc[binned["PLANT_NUMBER"].eq(1), "ACPower_LEVEL"].tolist() == [
        "Low",
        "Medium",
        "High",
    ]
    assert binned.loc[binned["PLANT_NUMBER"].eq(2), "ACPower_LEVEL"].tolist() == [
        "Low",
        "Medium",
        "High",
    ]
    assert with_transactions["TRANSACTION"].map(len).eq(4).all()
    assert with_transactions.iloc[1]["TRANSACTION"] == frozenset(
        {
            "Irradiation_Medium",
            "AmbientTemp_Medium",
            "ModuleTemp_Medium",
            "ACPower_Medium",
        }
    )
    assert all(
        not any(item.startswith("Plant_") for item in transaction)
        for transaction in with_transactions["TRANSACTION"]
    )


def test_mine_apriori_rules_restricts_to_environmental_to_ac_power() -> None:
    high = frozenset(
        {
            "Irradiation_High",
            "AmbientTemp_High",
            "ModuleTemp_High",
            "ACPower_High",
        }
    )
    medium = frozenset(
        {
            "Irradiation_Medium",
            "AmbientTemp_Medium",
            "ModuleTemp_Medium",
            "ACPower_Medium",
        }
    )
    low = frozenset(
        {
            "Irradiation_Low",
            "AmbientTemp_Low",
            "ModuleTemp_Low",
            "ACPower_Low",
        }
    )
    transactions = pd.DataFrame(
        {"TRANSACTION": [high] * 4 + [medium] * 2 + [low] * 4}
    )
    encoded = analysis.encode_transactions(transactions)

    frequent, all_rules, target_rules = analysis.mine_apriori_rules(
        encoded,
        minimum_support=0.2,
        minimum_confidence=0.6,
        max_length=3,
    )

    assert not frequent.empty
    assert not all_rules.empty
    assert not target_rules.empty
    assert all_rules["antecedents"].map(
        lambda items: any(item.startswith("ACPower_") for item in items)
    ).any()
    assert target_rules["consequents"].map(
        lambda items: len(items) == 1
        and next(iter(items)).startswith("ACPower_")
    ).all()
    assert target_rules["antecedents"].map(lambda items: 1 <= len(items) <= 2).all()
    assert target_rules["antecedents"].map(
        lambda items: all(
            item.startswith(("Irradiation_", "AmbientTemp_", "ModuleTemp_"))
            for item in items
        )
    ).all()
    assert (
        (target_rules["antecedents"] == frozenset({"Irradiation_High"}))
        & (target_rules["consequents"] == frozenset({"ACPower_High"}))
    ).any()


def test_rule_metrics_match_direct_support_confidence_and_lift_calculation() -> None:
    frame = pd.DataFrame(
        {
            "TRANSACTION": [
                frozenset({"X", "Y"}),
                frozenset({"X", "Y"}),
                frozenset({"X", "Z"}),
                frozenset({"Y", "Z"}),
            ],
            "DATE": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-02"]
            ),
        }
    )

    metrics = analysis._rule_metrics(
        frame, frozenset({"X"}), frozenset({"Y"})
    )

    assert metrics["transactions"] == 4
    assert metrics["antecedent_count"] == 3
    assert metrics["consequent_count"] == 3
    assert metrics["joint_count"] == 2
    assert metrics["distinct_days"] == 2
    assert metrics["support"] == pytest.approx(2 / 4)
    assert metrics["confidence"] == pytest.approx(2 / 3)
    assert metrics["consequent_support"] == pytest.approx(3 / 4)
    assert metrics["lift"] == pytest.approx((2 / 3) / (3 / 4))


def test_prune_redundant_rules_drops_weak_superset_but_keeps_material_gain() -> None:
    outcome = frozenset({"ACPower_High"})
    rules = pd.DataFrame(
        [
            {
                "antecedents": frozenset({"Irradiation_High"}),
                "consequents": outcome,
                "discovery_support": 0.30,
                "discovery_confidence": 0.70,
                "discovery_lift": 1.40,
            },
            {
                "antecedents": frozenset(
                    {"Irradiation_High", "AmbientTemp_High"}
                ),
                "consequents": outcome,
                "discovery_support": 0.20,
                "discovery_confidence": 0.73,
                "discovery_lift": 1.45,
            },
            {
                "antecedents": frozenset(
                    {"Irradiation_High", "ModuleTemp_Medium"}
                ),
                "consequents": outcome,
                "discovery_support": 0.18,
                "discovery_confidence": 0.82,
                "discovery_lift": 1.60,
            },
            {
                "antecedents": frozenset({"AmbientTemp_Low"}),
                "consequents": outcome,
                "discovery_support": 0.12,
                "discovery_confidence": 0.65,
                "discovery_lift": 1.30,
            },
        ]
    )

    pruned = analysis.prune_redundant_rules(
        rules, minimum_confidence_gain=0.05, minimum_lift_gain=0.10
    )

    retained = set(pruned["antecedents"])
    assert frozenset({"Irradiation_High", "AmbientTemp_High"}) not in retained
    assert retained == {
        frozenset({"Irradiation_High"}),
        frozenset({"Irradiation_High", "ModuleTemp_Medium"}),
        frozenset({"AmbientTemp_Low"}),
    }
    assert pruned.iloc[0]["antecedents"] == frozenset(
        {"Irradiation_High", "ModuleTemp_Medium"}
    )
