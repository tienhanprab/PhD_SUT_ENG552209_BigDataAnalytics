from __future__ import annotations

from solar_apriori.notebook import build_notebook


def test_notebook_contains_all_assignment_requirements() -> None:
    notebook = build_notebook()
    markdown = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for section in [
        "## tl;dr",
        "## Context & Methods",
        "## Data",
        "## Requirement 1: Select a Kaggle dataset",
        "## Requirement 2: Prepare data as transaction lists",
        "## Requirement 3: Perform Association Rule Mining using Apriori",
        "## Requirement 4: Top-N rules table and visualization",
        "## Requirement 5: Interpret 3-5 key rules with context insights",
        "## Takeaways",
    ]:
        assert section in markdown

    for required_step in [
        "prepare_plant_time_data",
        "add_transactions",
        "tune_thresholds",
        "mine_apriori_rules",
        "selected_min_support",
        "selected_min_confidence",
        "make_top_rules_table",
        "plot_rule_network",
    ]:
        assert required_step in code


def test_notebook_documents_noncausal_scope_and_exact_rule_interpretations() -> None:
    notebook = build_notebook()
    markdown = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    assert "ไม่ใช่เหตุและผล" in markdown
    assert "Support = **33.60%**, Confidence = **96.11%**, Lift = **2.77**" in markdown
    assert markdown.count("### Key Rule") == 5
