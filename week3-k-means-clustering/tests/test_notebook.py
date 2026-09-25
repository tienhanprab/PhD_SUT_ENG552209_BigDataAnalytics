from __future__ import annotations

from optical_fibre_clustering.notebook import build_notebook


def test_notebook_contains_required_assignment_sections() -> None:
    notebook = build_notebook()
    markdown = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    for section in ["## tl;dr", "## Context & Methods", "## Data", "## Results", "## Takeaways"]:
        assert section in markdown

    for requirement in [
        "## Requirement 1: Choose any numeric dataset from Kaggle.",
        "## Requirement 2: Briefly describe the data (rows, columns, features).",
        "## Requirement 3: Prepare the data:",
        "#### Requirement 3.1: Handle missing values (e.g., fill or drop).",
        "#### Requirement 3.2: Convert categorical data to numeric.",
        "#### Requirement 3.3: Scale/normalize features.",
        "## Requirement 4: Perform clustering (e.g., K-Means), try different numbers of clusters.",
        "## Requirement 5: Evaluate with Silhouette Score or Davies-Bouldin Score.",
        "## Requirement 6: Visualize the clusters and summarize your findings.",
        "### Requirement 6.2: Summary of findings",
    ]:
        assert requirement in markdown

    for required_step in [
        "SimpleImputer",
        "StandardScaler",
        "KMeans",
        "silhouette_score",
        "davies_bouldin_score",
        "PCA",
    ]:
        assert required_step in code


def test_notebook_keeps_labels_out_of_clustering_features() -> None:
    notebook = build_notebook()
    code = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert 'feature_frame = clean_data[feature_columns].copy()' in code
    assert 'held_out_columns = analysis_config["held_out_columns"]' in code
    assert 'clustered_data["Class"]' in code
