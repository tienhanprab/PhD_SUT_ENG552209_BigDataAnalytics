"""Build a teaching notebook; execution produces all experiment outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def build():
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# Interpretable Classification of Solar Photovoltaic Faults Using OneR and PRISM
**ENG55 2209 · Week 5 · Classification Rule Induction**

## Goal
Can simple rule-induction algorithms identify interpretable operating conditions associated with solar PV faults?

This notebook follows Week 5 p.41: Kaggle acquisition, preparation, OneR/PRISM training, accuracy, **precision/recall/F1 for each class**, rule counts and interpretation. Algorithms are implemented in `src/solar_classification/models.py`.

**Observed result:** OneR predicts only Normal and matches ZeroR (91.45% accuracy). PRISM reaches 91.90% accuracy but detects only 5.36% of fault rows. Neither is an adequate fault detector under this coarse representation. These are row-level results from one held-out simulated day, not evidence of field reliability.''')
    md('''## Setup
Run from this project or its `notebooks/` directory. Install with `python -m pip install -e .`.
The first run downloads the public version-2 CSV; later runs verify its SHA-256 and reuse it. No credentials or lecture PDF are needed to rerun.

Source: [Kaggle Solar PV Anomaly Detection Dataset, version 2](https://www.kaggle.com/datasets/isurumy93/solar-pv-anomaly-detection-dataset/versions/2). Publisher metadata and the data dictionary are preserved in `docs/`.''')
    code('''from pathlib import Path
import os, sys
ROOT = Path.cwd()
if ROOT.name == "notebooks": ROOT = ROOT.parent
if not (ROOT / "src/solar_classification").exists():
    raise RuntimeError("Run from week5-classification or its notebooks directory")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix
from solar_classification.analysis import run, FEATURES
plt.rcParams.update({"figure.figsize": (9,4.5), "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Normal":"#285f91", "Fault":"#c47525"}
FIGURES = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)''')
    md('''## Steps
### 1. Acquire, audit and split
Version 2 contains **16 columns**, not the six generic names in its description. Use five observable inputs: light (lux), panel temperature (°C), voltage (V), current (A), and power (W). Lux is not irradiance (W/m²). Exclude simulator ground truth (`true_*`), ambient temperature, IDs, time and `fault_type` from predictors.

### Key assumptions
- Train on days 0–1; test on day 2. Do not randomly mix neighboring time-series rows. Verify no nonzero anomaly event crosses the boundary.
- Remove only exact duplicate rows. Validate labels and timestamps. Median imputation and tertile boundaries are learned from **training rows only**.
- Low/Medium/High are relative training ranks, not physical safety thresholds. Repeated boundary values can reduce the number of effective bins.
- Retain night/low-light rows and class imbalance. No test-based model selection, feature selection, threshold tuning or resampling.
- This is simulated data. Row-level scores are correlated within events; there is only one held-out day and no field validation.''')
    code('''result = run()
audit = result["audit"]
display(pd.Series({k:v for k,v in audit.items() if k != "missing_by_column"}).to_frame("value"))''')
    code('''display(result["raw"].head(5))
display(pd.Series(audit["missing_by_column"], name="missing_rows").to_frame())
display(result["balance"])''')
    code('''ax = result["balance"].plot.bar(stacked=True, color=[COLORS["Normal"],COLORS["Fault"]], rot=0)
ax.set(title="Class balance by simulated day", xlabel="Day (0–1 train; 2 test)", ylabel="Observed rows")
ax.legend(title="Actual class")
plt.tight_layout(); plt.savefig(FIGURES / "class_balance.png", dpi=160); plt.show()''')
    md('''Fault prevalence differs across days. A majority-class model can achieve high accuracy without detecting faults; compare both classes and balanced accuracy. These counts describe sampled rows, not independent physical events or elapsed-time fractions.''')
    md('''### 2. Discretize using training data
Tertiles are a project design choice. Lecture p.18 instead presents a label-aware, bucket-size approach; this notebook does not claim to implement that discretizer. Intervals are right-closed with infinite outer tails. At repeated quantiles, only distinct interior boundaries remain: two bins use Low/High, a constant feature uses Medium.

The light sensor reaches 65,535 lux in 36.42% of rows, so its upper training tertile coincides with the maximum. It therefore has **two effective bins**. Do not interpret sensor saturation as high irradiance measured accurately.''')
    code('''display(result["edges"])
display(result["Xtrain"].head())
display(result["Xtrain"].nunique().rename("effective_bins").to_frame())''')
    md('''### 3. OneR: minimize total training error
For each attribute and category, predict the most common training class. Add the errors across categories and choose the attribute with the smallest total. Class ties choose 0; attribute ties choose the declared feature order. Unseen categories use the training majority.

Rules are mutually exclusive. Count each category→class rule separately, and list the default separately.''')
    code('''display(pd.DataFrame(result["models"]["OneR"].candidates_))
display(result["rules"].query("model == 'OneR'"))''')
    md('''All five attributes tie on training error. OneR chooses `light_lux` only because it is first in the deterministic feature order; **this does not establish that light is the best measurement**. Both its rules predict Normal. Majority voting within coarse bins hides the minority class.''')
    md('''### 4. PRISM: grow pure rules, one class at a time
Start with an empty rule for a target class. Choose an attribute-value condition maximizing p/t, break ties by positive coverage, then feature/category order. Refine until pure; remove covered positives and repeat. Restart with the full training data for the next class.

**Explicit noise policy:** after all attributes are used, reject any still-impure candidate and defer those positives; retain every negative during induction. This is a documented extension for contradictory discretized examples, not a claim that the lecture defines this case. Only pure rules are retained. A test row matching no rule, or rules for different classes, uses the training majority (Normal). Multiple matches for the same class agree.

Training purity does not guarantee test precision. Counts below measure each rule independently and can overlap.''')
    code('''display(result["rules"].query("model == 'PRISM'").reset_index(drop=True))
display(pd.Series({k:v for k,v in audit.items() if k.startswith("prism_") or "combinations" in k}).to_frame("value"))''')
    md('''### 5. Compare on the untouched test day
Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean. Compute each class as one-vs-rest. Undefined precision is reported as zero (`zero_division=0`); OneR/ZeroR never predict Fault. Macro-F1 averages both classes; balanced accuracy averages their recalls. Default rules are excluded from `rules` and reported separately.''')
    code('''display(result["metrics"].round(6))
display(result["per_class"].round(6))''')
    code('''fig, axes = plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
for ax,(name,pred) in zip(axes,result["predictions"].items()):
    cm = confusion_matrix(result["test"].fault,pred,labels=[0,1])
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=len(result["test"]))
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",color="white" if cm[i,j]>len(result["test"])/2 else "black")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Normal","Fault"],yticklabels=["Normal","Fault"],xlabel="Predicted class",ylabel="Actual class",title=name)
fig.suptitle("Test day 2: confusion matrices (51,694 rows)")
plt.savefig(FIGURES / "confusion_matrices.png",dpi=160);plt.show()''')
    code('''plot = result["metrics"].set_index("model")[["accuracy","balanced_accuracy","fault_recall"]]
ax = plot.plot.bar(rot=0,color=["#285f91","#7b852e","#c47525"])
ax.set(title="Accuracy versus fault detection on day 2",ylabel="Score (0–1)",xlabel="Model",ylim=(0,1.08))
ax.legend(["Accuracy","Balanced accuracy","Fault recall"],loc="upper center",bbox_to_anchor=(0.5,1.22),ncol=3)
plt.tight_layout();plt.savefig(FIGURES / "model_comparison.png",dpi=160);plt.show()''')
    md('''### 6. Interpret learned rules
PRISM learns four Fault rules. Two have no matches on the test day. The two active Fault rules are **Low power AND High panel temperature** and **Low voltage AND High panel temperature**; both cover the same 237 test rows. They describe low electrical output with relatively high module temperature, but do not prove a physical failure mechanism. Their union has 100% test precision and only 237/4,422 = 5.36% recall. The classifier misses 4,185 fault rows. Do not add overlapping rule counts together.

Normal rules must also be judged on held-out precision: a pure training rule can cover faults on day 2. Inspect `rules.csv` for each rule's training/test coverage and precision. There are 12 PRISM rules with 30 conditions (2.5 per rule), versus 2 OneR rules with one condition each.

PRISM has no matching learned rule for 31,038 test rows (60.04%), so its majority fallback is a large part of the classifier. Zero conflicting-class matches occur. Coarse categories leave 16 of 42 observed training combinations with both labels; the documented purity policy rejects 32 terminal candidates.''')
    md('''## Checks
The source checksum is validated on every run. The pipeline checks labels, time ordering and event separation. Unit tests cover the lecture weather example, a conjunction, contradictory labels, unseen categories and training-only bin boundaries.

`PYTHONPATH=src python tests/test_models.py`

Power is not exactly measured voltage × measured current in the CSV: mean absolute residual is about 0.02155 W. Use the supplied `power_w`; do not overwrite it to force agreement with the generic description. It is still strongly redundant with the electrical inputs.''')
    code('''assert sum(result["balance"].sum()) == audit["clean_rows"]
assert audit["train_rows"] + audit["test_rows"] == audit["clean_rows"]
assert all(r.support == r.correct for r in result["models"]["PRISM"].rules_)
assert (result["predictions"]["OneR"] == 0).all()
print("All notebook consistency checks passed.")''')
    md('''## Next Steps
The simple rules are interpretable, but this experiment does not demonstrate sufficient fault detection. Accuracy gains over ZeroR are small and minority recall remains low.

Future experiments should predefine a validation split within training days before changing bins or adding physically motivated ratios. Compare the lecture's supervised bucket discretization and a clearly labeled noise-tolerant PRISM variant. Acquire longer simulations and real measurements for event-level and cross-system validation. Do not repeatedly tune against day 2.

### Sources and limitations
- Week 5 lecture: OneR examples pp.3–15; numeric discretization pp.16–20; ZeroR p.21; PRISM pp.22–40; assignment p.41. Source PDF remains in the user's lecture folder, not redistributed here.
- Kaggle version 2 and preserved metadata in `docs/kaggle_metadata.json`; exact SHA-256 in `data/raw/provenance.json`.
- Metadata labels the license CC0, while description prose suggests CC BY/MIT. Preserve this inconsistency; raw CSV is locally available and excluded from Git.
- Simulated, single-system, three-day data; correlated rows, quantization/saturation and redundant power inputs. No confidence interval assuming independent rows and no claim of real-world diagnostic safety.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(notebook)
    destination = ROOT / 'notebooks/solar_pv_classification.ipynb'
    nbf.write(notebook,destination)
    print(destination)


if __name__ == '__main__': build()
