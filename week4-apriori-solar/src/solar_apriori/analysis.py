"""Core data preparation, Apriori mining, evaluation, and visualization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

GENERATION_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
]
WEATHER_COLUMNS = [
    "DATE_TIME",
    "PLANT_ID",
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
]
PLANTS = (1, 2)
LEVELS = ("Low", "Medium", "High")
BINNING_VARIABLES = {
    "IRRADIATION": "Irradiation",
    "AMBIENT_TEMPERATURE": "AmbientTemp",
    "MODULE_TEMPERATURE": "ModuleTemp",
    "AC_POWER_MEAN": "ACPower",
}
TIMESTAMP_FORMATS = ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class PreparedData:
    """Container for the auditable plant-time table and quality summaries."""

    plant_time: pd.DataFrame
    source_quality: pd.DataFrame
    merge_quality: pd.DataFrame
    inverter_coverage: pd.DataFrame


def parse_mixed_timestamp(values: pd.Series) -> pd.Series:
    """Parse the two documented timestamp layouts without locale guessing."""
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for timestamp_format in TIMESTAMP_FORMATS:
        missing = parsed.isna()
        parsed.loc[missing] = pd.to_datetime(
            values.loc[missing], format=timestamp_format, errors="coerce"
        )
    if parsed.isna().any():
        examples = values.loc[parsed.isna()].astype(str).head(3).tolist()
        raise ValueError(f"Unparseable DATE_TIME values: {examples}")
    return parsed


def _read_and_validate(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    frame = pd.read_csv(path)
    if frame.columns.tolist() != list(expected_columns):
        raise ValueError(
            f"Unexpected schema in {path.name}: {frame.columns.tolist()}"
        )
    return frame


def prepare_plant_time_data(raw_directory: Path) -> PreparedData:
    """Aggregate inverter rows before exact joining to plant-level weather rows."""
    plant_tables: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    merge_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for plant_number in PLANTS:
        generation_path = raw_directory / f"Plant_{plant_number}_Generation_Data.csv"
        weather_path = raw_directory / f"Plant_{plant_number}_Weather_Sensor_Data.csv"
        generation = _read_and_validate(generation_path, GENERATION_COLUMNS)
        weather = _read_and_validate(weather_path, WEATHER_COLUMNS)

        for source_name, frame, key_columns in (
            (
                "generation",
                generation,
                ["PLANT_ID", "DATE_TIME", "SOURCE_KEY"],
            ),
            ("weather", weather, ["PLANT_ID", "DATE_TIME"]),
        ):
            source_rows.append(
                {
                    "Plant": plant_number,
                    "Source": source_name,
                    "Rows": len(frame),
                    "Columns": frame.shape[1],
                    "Null values": int(frame.isna().sum().sum()),
                    "Exact duplicate rows": int(frame.duplicated().sum()),
                    "Duplicate grain keys": int(frame.duplicated(key_columns).sum()),
                }
            )

        generation = generation.copy()
        weather = weather.copy()
        generation["TIMESTAMP"] = parse_mixed_timestamp(generation["DATE_TIME"])
        weather["TIMESTAMP"] = parse_mixed_timestamp(weather["DATE_TIME"])

        if generation["PLANT_ID"].nunique() != 1 or weather["PLANT_ID"].nunique() != 1:
            raise ValueError(f"Plant {plant_number} source contains multiple PLANT_ID values")
        if generation[["DC_POWER", "AC_POWER"]].lt(0).any().any():
            raise ValueError(f"Plant {plant_number} generation contains negative power")
        if weather["IRRADIATION"].lt(0).any():
            raise ValueError(f"Plant {plant_number} weather contains negative irradiation")

        expected_inverters = int(generation["SOURCE_KEY"].nunique())
        aggregated = (
            generation.groupby(["PLANT_ID", "TIMESTAMP"], as_index=False)
            .agg(
                INVERTER_COUNT=("SOURCE_KEY", "nunique"),
                DC_POWER_SUM=("DC_POWER", "sum"),
                DC_POWER_MEAN=("DC_POWER", "mean"),
                AC_POWER_SUM=("AC_POWER", "sum"),
                AC_POWER_MEAN=("AC_POWER", "mean"),
            )
            .assign(EXPECTED_INVERTERS=expected_inverters, PLANT_NUMBER=plant_number)
        )
        aggregated["INVERTER_COVERAGE"] = (
            aggregated["INVERTER_COUNT"] / aggregated["EXPECTED_INVERTERS"]
        )

        weather_at_time = weather.drop(columns=["DATE_TIME", "SOURCE_KEY"])
        merged = aggregated.merge(
            weather_at_time,
            on=["PLANT_ID", "TIMESTAMP"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        merge_counts = merged["_merge"].value_counts()
        merge_rows.append(
            {
                "Plant": plant_number,
                "Generation timestamps": len(aggregated),
                "Weather timestamps": len(weather_at_time),
                "Matched timestamps": int(merge_counts.get("both", 0)),
                "Generation only": int(merge_counts.get("left_only", 0)),
                "Weather only": int(merge_counts.get("right_only", 0)),
            }
        )

        matched = merged.loc[merged["_merge"].eq("both")].drop(columns="_merge")
        coverage_distribution = (
            matched["INVERTER_COUNT"].value_counts().sort_index().to_dict()
        )
        coverage_rows.append(
            {
                "Plant": plant_number,
                "Expected inverters": expected_inverters,
                "Complete timestamps": int(
                    matched["INVERTER_COUNT"].eq(expected_inverters).sum()
                ),
                "Incomplete timestamps": int(
                    matched["INVERTER_COUNT"].lt(expected_inverters).sum()
                ),
                "Minimum observed inverters": int(matched["INVERTER_COUNT"].min()),
                "Coverage distribution": str(coverage_distribution),
            }
        )
        plant_tables.append(matched)

    plant_time = pd.concat(plant_tables, ignore_index=True).sort_values(
        ["TIMESTAMP", "PLANT_NUMBER"]
    )
    plant_time["DATE"] = plant_time["TIMESTAMP"].dt.normalize()
    return PreparedData(
        plant_time=plant_time.reset_index(drop=True),
        source_quality=pd.DataFrame(source_rows),
        merge_quality=pd.DataFrame(merge_rows),
        inverter_coverage=pd.DataFrame(coverage_rows),
    )


def select_daylight_population(
    plant_time: pd.DataFrame,
    minimum_irradiation: float = 0.0,
    minimum_inverter_coverage: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select meaningful daylight intervals and report each exclusion reason."""
    required = [
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "AC_POWER_MEAN",
        "INVERTER_COVERAGE",
    ]
    complete = plant_time[required].notna().all(axis=1)
    daylight = plant_time["IRRADIATION"].gt(minimum_irradiation)
    adequate_coverage = plant_time["INVERTER_COVERAGE"].ge(
        minimum_inverter_coverage
    )
    selected = complete & daylight & adequate_coverage
    audit = pd.DataFrame(
        {
            "Population step": [
                "Exact generation-weather matches",
                "Complete analytical fields",
                f"Daylight (IRRADIATION > {minimum_irradiation:g})",
                f"Inverter coverage >= {minimum_inverter_coverage:.0%}",
                "Final plant-time transactions",
            ],
            "Rows": [
                len(plant_time),
                int(complete.sum()),
                int((complete & daylight).sum()),
                int((complete & daylight & adequate_coverage).sum()),
                int(selected.sum()),
            ],
        }
    )
    result = plant_time.loc[selected].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("Daylight filters produced no transactions")
    return result, audit


def split_dates(
    frame: pd.DataFrame, discovery_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by whole dates to avoid leaking adjacent intervals."""
    dates = np.array(sorted(frame["DATE"].drop_duplicates()))
    if len(dates) < 2:
        raise ValueError("At least two dates are required for discovery/validation")
    discovery_count = int(np.ceil(len(dates) * discovery_fraction))
    discovery_count = min(max(discovery_count, 1), len(dates) - 1)
    discovery_dates = set(dates[:discovery_count])
    discovery = frame.loc[frame["DATE"].isin(discovery_dates)].copy()
    validation = frame.loc[~frame["DATE"].isin(discovery_dates)].copy()
    summary = pd.DataFrame(
        {
            "Split": ["Discovery", "Validation"],
            "First date": [discovery["DATE"].min(), validation["DATE"].min()],
            "Last date": [discovery["DATE"].max(), validation["DATE"].max()],
            "Distinct dates": [discovery["DATE"].nunique(), validation["DATE"].nunique()],
            "Transactions": [len(discovery), len(validation)],
        }
    )
    return discovery, validation, summary


def fit_tertile_edges(discovery: pd.DataFrame) -> pd.DataFrame:
    """Fit plant-specific tertile boundaries on discovery dates only."""
    rows: list[dict[str, object]] = []
    for plant_number in sorted(discovery["PLANT_NUMBER"].unique()):
        plant = discovery.loc[discovery["PLANT_NUMBER"].eq(plant_number)]
        for source_column, item_family in BINNING_VARIABLES.items():
            boundaries = plant[source_column].quantile([1 / 3, 2 / 3]).to_numpy()
            if len(np.unique(boundaries)) != 2:
                raise ValueError(
                    f"Cannot create three distinct bins for plant {plant_number}, "
                    f"column {source_column}: {boundaries.tolist()}"
                )
            rows.append(
                {
                    "Plant": int(plant_number),
                    "Source column": source_column,
                    "Item family": item_family,
                    "Low/Medium boundary": float(boundaries[0]),
                    "Medium/High boundary": float(boundaries[1]),
                }
            )
    return pd.DataFrame(rows)


def apply_tertile_edges(frame: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen plant-specific boundaries to any compatible population."""
    result = frame.copy()
    for source_column, item_family in BINNING_VARIABLES.items():
        level_column = f"{item_family}_LEVEL"
        result[level_column] = pd.Series(index=result.index, dtype="object")
        for plant_number in sorted(result["PLANT_NUMBER"].unique()):
            row = edges.loc[
                edges["Plant"].eq(plant_number)
                & edges["Source column"].eq(source_column)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Missing unique bin edges for plant {plant_number}, {source_column}"
                )
            lower = float(row["Low/Medium boundary"].iloc[0])
            upper = float(row["Medium/High boundary"].iloc[0])
            plant_mask = result["PLANT_NUMBER"].eq(plant_number)
            result.loc[plant_mask, level_column] = pd.cut(
                result.loc[plant_mask, source_column],
                bins=[-np.inf, lower, upper, np.inf],
                labels=LEVELS,
                include_lowest=True,
            ).astype(str)
    level_columns = [f"{family}_LEVEL" for family in BINNING_VARIABLES.values()]
    if result[level_columns].isna().any().any():
        raise ValueError("Binning produced missing transaction items")
    return result


def add_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one four-item transaction per plant-time row."""
    result = frame.copy()
    item_columns = [f"{family}_LEVEL" for family in BINNING_VARIABLES.values()]
    result["TRANSACTION"] = result[item_columns].apply(
        lambda row: frozenset(
            f"{family}_{row[f'{family}_LEVEL']}"
            for family in BINNING_VARIABLES.values()
        ),
        axis=1,
    )
    lengths = result["TRANSACTION"].map(len)
    if not lengths.eq(len(BINNING_VARIABLES)).all():
        raise ValueError("Each transaction must contain exactly one item per variable")
    return result


def transaction_lists(frame: pd.DataFrame) -> list[list[str]]:
    """Return sorted lists for presentation and TransactionEncoder."""
    return [sorted(items) for items in frame["TRANSACTION"]]


def encode_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode transactions for mlxtend Apriori."""
    encoder = TransactionEncoder()
    matrix = encoder.fit(transaction_lists(frame)).transform(
        transaction_lists(frame)
    )
    encoded = pd.DataFrame(matrix, columns=encoder.columns_, index=frame.index)
    if not encoded.sum(axis=1).eq(len(BINNING_VARIABLES)).all():
        raise ValueError("Encoded transactions do not contain four active items")
    return encoded


def mine_apriori_rules(
    encoded: pd.DataFrame,
    minimum_support: float,
    minimum_confidence: float,
    max_length: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mine frequent itemsets and retain environmental -> AC-power rules."""
    frequent_itemsets = apriori(
        encoded.astype(bool),
        min_support=minimum_support,
        use_colnames=True,
        max_len=max_length,
        low_memory=False,
    )
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame(), pd.DataFrame()
    all_rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=minimum_confidence,
    )
    if all_rules.empty:
        return frequent_itemsets, all_rules, all_rules.copy()
    target_mask = (
        all_rules["consequents"].map(
            lambda values: len(values) == 1
            and next(iter(values)).startswith("ACPower_")
        )
        & all_rules["antecedents"].map(lambda values: 1 <= len(values) <= 2)
        & all_rules["antecedents"].map(
            lambda values: all(not value.startswith("ACPower_") for value in values)
        )
    )
    target_rules = all_rules.loc[target_mask].copy().reset_index(drop=True)
    return frequent_itemsets, all_rules, target_rules


def _rule_metrics(
    frame: pd.DataFrame, antecedent: frozenset[str], consequent: frozenset[str]
) -> dict[str, float | int]:
    union = antecedent | consequent
    antecedent_mask = frame["TRANSACTION"].map(antecedent.issubset)
    consequent_mask = frame["TRANSACTION"].map(consequent.issubset)
    union_mask = frame["TRANSACTION"].map(union.issubset)
    transaction_count = len(frame)
    antecedent_count = int(antecedent_mask.sum())
    consequent_count = int(consequent_mask.sum())
    joint_count = int(union_mask.sum())
    support = joint_count / transaction_count if transaction_count else np.nan
    confidence = joint_count / antecedent_count if antecedent_count else np.nan
    consequent_support = (
        consequent_count / transaction_count if transaction_count else np.nan
    )
    lift = (
        confidence / consequent_support
        if consequent_support and not np.isnan(confidence)
        else np.nan
    )
    distinct_days = int(frame.loc[union_mask, "DATE"].nunique())
    return {
        "transactions": transaction_count,
        "antecedent_count": antecedent_count,
        "consequent_count": consequent_count,
        "joint_count": joint_count,
        "distinct_days": distinct_days,
        "support": support,
        "confidence": confidence,
        "consequent_support": consequent_support,
        "lift": lift,
    }


def enrich_rule_metrics(
    rules: pd.DataFrame,
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute rule metrics from counts across time and plant segments."""
    if rules.empty:
        return rules.copy()
    records: list[dict[str, object]] = []
    for _, row in rules.iterrows():
        antecedent = frozenset(row["antecedents"])
        consequent = frozenset(row["consequents"])
        record: dict[str, object] = {
            "antecedents": antecedent,
            "consequents": consequent,
        }
        populations = {
            "discovery": discovery,
            "validation": validation,
            "overall": overall,
        }
        for plant_number in sorted(overall["PLANT_NUMBER"].unique()):
            populations[f"plant_{int(plant_number)}"] = overall.loc[
                overall["PLANT_NUMBER"].eq(plant_number)
            ]
        for prefix, population in populations.items():
            metrics = _rule_metrics(population, antecedent, consequent)
            record.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        records.append(record)
    return pd.DataFrame(records)


def screen_stable_rules(
    rules: pd.DataFrame,
    *,
    minimum_support: float,
    minimum_confidence: float,
    minimum_joint_count: int,
    minimum_discovery_days: int,
    minimum_discovery_lift: float,
    minimum_validation_lift: float,
    minimum_validation_antecedent_count: int,
    minimum_plant_lift: float,
) -> pd.DataFrame:
    """Keep rules with evidence, temporal coverage, and directional replication."""
    if rules.empty:
        return rules.copy()
    plant_lift_columns = [
        column
        for column in rules.columns
        if column.startswith("plant_") and column.endswith("_lift")
    ]
    mask = (
        rules["discovery_support"].ge(minimum_support)
        & rules["discovery_confidence"].ge(minimum_confidence)
        & rules["discovery_joint_count"].ge(minimum_joint_count)
        & rules["discovery_distinct_days"].ge(minimum_discovery_days)
        & rules["discovery_lift"].gt(minimum_discovery_lift)
        & rules["validation_lift"].gt(minimum_validation_lift)
        & rules["validation_antecedent_count"].ge(
            minimum_validation_antecedent_count
        )
    )
    if plant_lift_columns:
        mask &= rules[plant_lift_columns].gt(minimum_plant_lift).all(axis=1)
    return rules.loc[mask].copy().reset_index(drop=True)


def prune_redundant_rules(
    rules: pd.DataFrame,
    minimum_confidence_gain: float,
    minimum_lift_gain: float,
) -> pd.DataFrame:
    """Remove longer rules that add little beyond a retained subset rule."""
    if rules.empty:
        return rules.copy()
    keep_indices: list[int] = []
    for index, row in rules.iterrows():
        antecedent = row["antecedents"]
        if len(antecedent) == 1:
            keep_indices.append(index)
            continue
        subset_rules = rules.loc[
            rules["consequents"].map(lambda value: value == row["consequents"])
            & rules["antecedents"].map(
                lambda candidate: len(candidate) == 1
                and candidate.issubset(antecedent)
            )
        ]
        is_redundant = False
        for _, subset in subset_rules.iterrows():
            confidence_gain = (
                row["discovery_confidence"] - subset["discovery_confidence"]
            )
            lift_gain = row["discovery_lift"] - subset["discovery_lift"]
            if (
                confidence_gain < minimum_confidence_gain
                and lift_gain < minimum_lift_gain
            ):
                is_redundant = True
                break
        if not is_redundant:
            keep_indices.append(index)
    result = rules.loc[keep_indices].copy()
    result = result.sort_values(
        ["discovery_lift", "discovery_confidence", "discovery_support"],
        ascending=False,
    ).reset_index(drop=True)
    return result


def tune_thresholds(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    overall: pd.DataFrame,
    support_grid: Iterable[float],
    confidence_grid: Iterable[float],
    max_length: int,
    screening: Mapping[str, float | int],
    redundancy_confidence_gain: float,
    redundancy_lift_gain: float,
) -> pd.DataFrame:
    """Evaluate a support-confidence grid with the same stability filters."""
    encoded = encode_transactions(discovery)
    records: list[dict[str, object]] = []
    for minimum_support in support_grid:
        for minimum_confidence in confidence_grid:
            frequent, all_rules, target = mine_apriori_rules(
                encoded,
                minimum_support=float(minimum_support),
                minimum_confidence=float(minimum_confidence),
                max_length=max_length,
            )
            enriched = enrich_rule_metrics(target, discovery, validation, overall)
            stable = screen_stable_rules(
                enriched,
                minimum_support=float(minimum_support),
                minimum_confidence=float(minimum_confidence),
                **screening,
            )
            nonredundant = prune_redundant_rules(
                stable,
                minimum_confidence_gain=redundancy_confidence_gain,
                minimum_lift_gain=redundancy_lift_gain,
            )
            records.append(
                {
                    "min_support": float(minimum_support),
                    "min_confidence": float(minimum_confidence),
                    "frequent_itemsets": len(frequent),
                    "all_rules": len(all_rules),
                    "target_rules": len(target),
                    "stable_rules": len(stable),
                    "nonredundant_rules": len(nonredundant),
                }
            )
    return pd.DataFrame(records)


def itemset_label(values: frozenset[str]) -> str:
    """Create a compact deterministic display label."""
    return ", ".join(sorted(values))


def make_top_rules_table(rules: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Create the assignment-ready Top-N table with exact count evidence."""
    top = (
        rules.sort_values(
            ["overall_lift", "overall_confidence", "overall_support"],
            ascending=False,
        )
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )
    top.insert(0, "Rule ID", [f"R{index}" for index in range(1, len(top) + 1)])
    top["Antecedent"] = top["antecedents"].map(itemset_label)
    top["Consequent"] = top["consequents"].map(itemset_label)
    columns = {
        "overall_antecedent_count": "n(X)",
        "overall_joint_count": "n(X∪Y)",
        "overall_distinct_days": "Distinct days",
        "overall_support": "Support",
        "overall_confidence": "Confidence",
        "overall_consequent_support": "Consequent baseline",
        "overall_lift": "Lift",
        "validation_support": "Validation support",
        "validation_confidence": "Validation confidence",
        "validation_lift": "Validation lift",
        "plant_1_lift": "Plant 1 lift",
        "plant_2_lift": "Plant 2 lift",
    }
    result = top[
        ["Rule ID", "Antecedent", "Consequent", *columns.keys()]
    ].rename(columns=columns)
    return result


def plot_tuning_heatmap(
    tuning: pd.DataFrame,
    selected_support: float,
    selected_confidence: float,
    output_path: Path | None = None,
):
    """Plot the count of stable, nonredundant rules for every threshold pair."""
    pivot = tuning.pivot(
        index="min_confidence", columns="min_support", values="nonredundant_rules"
    ).sort_index(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="g",
        cmap=sns.light_palette("#2F6B8A", as_cmap=True),
        cbar_kws={"label": "Stable nonredundant rules"},
        linewidths=0.8,
        linecolor="white",
        ax=axis,
    )
    axis.set_title("Threshold tuning: stable environmental → AC-power rules")
    axis.set_xlabel("Minimum support")
    axis.set_ylabel("Minimum confidence")
    if (
        selected_support in pivot.columns
        and selected_confidence in pivot.index
    ):
        column_index = list(pivot.columns).index(selected_support)
        row_index = list(pivot.index).index(selected_confidence)
        axis.add_patch(
            plt.Rectangle(
                (column_index, row_index),
                1,
                1,
                fill=False,
                edgecolor="#C6922B",
                linewidth=3,
            )
        )
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=200)
    return figure, axis


def plot_rule_network(
    rules: pd.DataFrame,
    output_path: Path | None = None,
    random_state: int = 42,
):
    """Plot a directed item-rule-item network that preserves conjunctions."""
    graph = nx.DiGraph()
    item_families = {
        "Irradiation": "#2F6B8A",
        "AmbientTemp": "#C6922B",
        "ModuleTemp": "#D9793D",
        "ACPower": "#6F7D43",
    }
    rule_metrics: dict[str, tuple[float, float]] = {}
    for _, row in rules.iterrows():
        rule_id = str(row["Rule ID"])
        support = float(row["Support"])
        confidence = float(row["Confidence"])
        lift = float(row["Lift"])
        graph.add_node(rule_id, kind="rule")
        rule_metrics[rule_id] = (support, lift)
        antecedents = [item.strip() for item in str(row["Antecedent"]).split(",")]
        consequent = str(row["Consequent"])
        for item in antecedents:
            family = item.rsplit("_", 1)[0]
            graph.add_node(item, kind="item", family=family)
            graph.add_edge(item, rule_id, confidence=confidence)
        family = consequent.rsplit("_", 1)[0]
        graph.add_node(consequent, kind="item", family=family)
        graph.add_edge(rule_id, consequent, confidence=confidence)

    item_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "item"]
    rule_nodes = [node for node, data in graph.nodes(data=True) if data["kind"] == "rule"]
    # Use a deterministic three-column layout so the direction reads naturally:
    # environmental item -> rule -> AC-power outcome. Rules are grouped by outcome.
    del random_state  # kept in the public API for backward-compatible reproducibility.
    outcome_nodes = sorted(
        [node for node in item_nodes if node.startswith("ACPower_")],
        key=lambda node: {"High": 0, "Medium": 1, "Low": 2}.get(
            node.rsplit("_", 1)[-1], 3
        ),
    )
    outcome_centers = {"High": 2.4, "Medium": 1.25, "Low": 0.1}
    positions: dict[str, np.ndarray] = {}
    for outcome in outcome_nodes:
        level = outcome.rsplit("_", 1)[-1]
        positions[outcome] = np.array([2.0, outcome_centers[level]])

    rules_by_outcome: dict[str, list[str]] = {outcome: [] for outcome in outcome_nodes}
    for rule_node in rule_nodes:
        successors = [node for node in graph.successors(rule_node) if node in outcome_nodes]
        if len(successors) != 1:
            raise ValueError(f"Rule {rule_node} must have exactly one AC-power outcome")
        rules_by_outcome[successors[0]].append(rule_node)
    for outcome, grouped_rules in rules_by_outcome.items():
        grouped_rules.sort(key=lambda value: int(value.removeprefix("R")))
        center = positions[outcome][1]
        offsets = (
            np.linspace(0.45, -0.45, len(grouped_rules))
            if len(grouped_rules) > 1
            else np.array([0.0])
        )
        for rule_node, offset in zip(grouped_rules, offsets, strict=True):
            positions[rule_node] = np.array([1.0, center + offset])

    antecedent_nodes = [node for node in item_nodes if node not in outcome_nodes]
    for item in antecedent_nodes:
        connected_rules = [node for node in graph.successors(item) if node in rule_nodes]
        mean_y = float(np.mean([positions[node][1] for node in connected_rules]))
        positions[item] = np.array([0.0, mean_y])
    # Resolve near-collisions in the antecedent column without changing the ordering.
    ordered_antecedents = sorted(antecedent_nodes, key=lambda node: positions[node][1])
    minimum_gap = 0.24
    for previous, current in zip(
        ordered_antecedents, ordered_antecedents[1:], strict=False
    ):
        if positions[current][1] - positions[previous][1] < minimum_gap:
            positions[current][1] = positions[previous][1] + minimum_gap

    figure, axis = plt.subplots(figsize=(15, 9))
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=antecedent_nodes,
        node_color=[
            item_families.get(graph.nodes[node].get("family", ""), "#B8B8B8")
            for node in antecedent_nodes
        ],
        node_size=2300,
        edgecolors="#333333",
        linewidths=1,
        ax=axis,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=outcome_nodes,
        node_color=item_families["ACPower"],
        node_size=2500,
        edgecolors="#333333",
        linewidths=1.2,
        ax=axis,
    )
    lifts = np.array([rule_metrics[node][1] for node in rule_nodes], dtype=float)
    supports = np.array([rule_metrics[node][0] for node in rule_nodes], dtype=float)
    normalization = Normalize(vmin=float(lifts.min()), vmax=float(lifts.max()))
    colormap = plt.get_cmap("YlOrBr")
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=rule_nodes,
        node_shape="s",
        node_color=colormap(normalization(lifts)),
        node_size=900 + 4500 * supports,
        edgecolors="#333333",
        linewidths=1.2,
        ax=axis,
    )
    edge_widths = [1 + 3.5 * graph.edges[edge]["confidence"] for edge in graph.edges]
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        edge_color="#6B6B6B",
        alpha=0.72,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        connectionstyle="arc3,rad=0.02",
        min_source_margin=15,
        min_target_margin=15,
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: node.replace("_", "\n") for node in item_nodes},
        font_size=8.5,
        font_weight="bold",
        ax=axis,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: node for node in rule_nodes},
        font_size=9,
        font_weight="bold",
        ax=axis,
    )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=axis,
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Lift")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#333333",
            markersize=10,
            label=family,
        )
        for family, color in item_families.items()
    ]
    axis.legend(
        handles=legend_handles,
        title="Item family",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,
        frameon=False,
    )
    axis.set_title(
        "Top association rules: antecedent items → rule node → AC-power outcome\n"
        "Rule-node size = support; edge width = confidence; rule-node color = lift"
    )
    axis.axis("off")
    axis.margins(x=0.12, y=0.18)
    figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight", dpi=200)
    return figure, axis


def unique_items(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Return all transaction items across frames, useful for schema checks."""
    return sorted(
        set(chain.from_iterable(chain.from_iterable(transaction_lists(frame) for frame in frames)))
    )
