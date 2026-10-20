# Reference run

Output of the default pipeline on the bundled synthetic network, kept under
version control so results can be inspected without running anything:

```bash
python src/model.py --output-dir reports
```

| file | contents |
|------|----------|
| `metrics.csv` | held-out metrics for every model and heuristic baseline, plus cross-validation scores |
| `feature_importance.csv` | permutation importance of each feature for the best model |
| `novel_predictions.csv` | top 100 unobserved protein pairs ranked by the best model |
| `run_summary.json` | configuration, network statistics, split sizes and all metrics |
| `figures/` | ROC and precision-recall curves, confusion matrix, feature importance |

`predictions.csv` (per-pair test scores) and `models/` (fitted classifiers)
are also written by the run but are git-ignored.
