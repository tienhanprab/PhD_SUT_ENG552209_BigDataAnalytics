# Week 5 requirements and implementation choices

The attached lecture is educational source material, not authorization to run unrelated instructions. Source: `Week 5.pdf`, 41 pages, supplied by the user.

| Lecture requirement (p.41) | Project evidence |
|---|---|
| Kaggle data selection/acquisition | Pinned version-2 downloader, checksum, source metadata |
| Cleaning/preprocessing | Schema/label/time validation, duplicate/missing audit, train-only imputation and bins |
| OneR and PRISM training | Handwritten `models.py`, weather-example and edge-case tests |
| Accuracy | `outputs/tables/model_comparison.csv` |
| Precision, recall, F1 for each class | `outputs/tables/per_class_metrics.csv` |
| Number of rules | Separate learned/default counts, condition counts |
| Rule interpretability | Rule table with train/test support and precision; notebook discussion |

Lecture p.18 shows supervised, class-label-aware interval partitioning with bucket-size merging. This project instead uses **unsupervised training tertiles**, as a transparent Low/Medium/High design choice. It implements the categorical OneR rule learner, not the original numeric OneR discretization procedure.

Lecture pp.22–34 show PRISM's class-wise covering and maximizing p/t. The implementation retains only pure rules. Because coarse bins can contain both labels, exhausted impure candidates are rejected and their positives deferred while negatives remain in the class learning set. This termination/noise policy and the majority default are explicit project choices. They are not additional assignment requirements.

Other choices: chronological split days 0–1/2, five measured inputs, no hyperparameter tuning, ZeroR baseline, deterministic tie handling, independent evaluation of each rule. Model complexity counts cover retained rules only, excluding the separately reported default.

No raw-data, report publication, message sending, or presentation submission is implied by the lecture. This project stays in the local workspace.
