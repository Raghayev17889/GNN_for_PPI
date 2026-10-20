# Graph-Based Protein-Protein Interaction (PPI) Prediction

[![CI](https://github.com/Raghayev17889/GNN_for_PPI/actions/workflows/ci.yml/badge.svg)](https://github.com/Raghayev17889/GNN_for_PPI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains a modular Python pipeline for modelling and predicting protein-protein
interactions (PPIs). By combining structural graph theory, topological network metrics and machine
learning classifiers, it shows how computational network analysis can uncover hidden functional
relationships in molecular biology.

It was built as part of a bioinformatics portfolio and connects graph analytics with biological network
interpretation, following the analytical frameworks used in systems biology research. Everything runs on
**real human interactome data**: four public maps are downloaded, checksum-verified, preprocessed with an
audit trail, analysed and modelled.

**Results in brief**

* On the **HI-III** human interactome (5,216 proteins, 22,195 binary interactions), topology alone predicts
  held-out interactions with **ROC-AUC 0.972**. Against hard negatives (non-interacting proteins two or
  three steps apart) it still reaches **0.934**.
* The most informative signal is not shared partners but **paths of length three (L3)**. In systematic
  yeast two-hybrid maps, interacting proteins share *fewer* partners than degree-matched random networks
  (0.78-0.88×) but are linked by *more* length-3 paths (1.5×). This matches the complementarity principle
  of protein binding (Kovács et al., 2019).
* Across four interactomes (binary Y2H, literature-curated, co-complex mass spectrometry), L3 beats
  common neighbours on every map. With hard negatives, common-neighbour scores fall to chance or below.
  The supervised models are the most robust method everywhere (ROC-AUC 0.90-0.97).

<p align="center">
  <img src="reports/hi-iii/figures/roc_curves.png" width="48%" alt="ROC curves on HI-III">
  <img src="reports/data_analysis/figures/closure_enrichment.png" width="48%" alt="Triangle vs square enrichment across interactomes">
</p>

## Key features

* **Real data acquisition.** A registry of public human interactomes (HI-III, HI-II-14, Lit-BM-13,
  BioPlex 2.0, BioGRID) and an NCBI gene annotation, downloaded from commit-pinned URLs with SHA-256
  verification, retries and atomic writes (`src/datasets.py`).
* **Audited preprocessing.** Parses header-less files with classic-Mac line endings, removes homodimers,
  collapses duplicate and reversed pairs, drops discontinued gene IDs and non-protein-coding loci, keeps
  the largest component, adds gene symbols, and records what every step removed (`src/preprocessing.py`).
* **Exploratory data analysis.** Network statistics, degree distributions, overlap between maps, hubs,
  and a triangle-vs-square closure test against degree-preserving null models (`src/data_analysis.py`).
* **Network construction.** Generic edge lists (TSV/CSV) and STRING `protein.links` files into `NetworkX`
  graphs (`src/graph_builder.py`).
* **Leakage-free evaluation protocol.** Hidden test and training interactions are removed from the graphs
  on which their features are computed, so no pair ever "sees" its own edge. A protected random spanning
  tree keeps every protein connected. Negatives can be sampled **uniformly** or **locally** (2-3 hops apart).
* **Topological features.** 15 symmetric features per protein pair:
  * Node level (min and max over the pair): degree, clustering coefficient, betweenness and closeness
    centrality.
  * Pairwise: common neighbours, Jaccard (neighbourhood overlap), Adamic-Adar, resource allocation,
    preferential attachment, **degree-normalised L3 paths**, and shortest-path distance.
  * Sparse-matrix implementations, compiled closeness, and automatic sampled betweenness above 3,000
    proteins.
* **Machine-learning classification.** Logistic regression (log-scaled features), random forest and
  histogram gradient boosting from `scikit-learn`, with stratified cross-validation.
* **Thorough evaluation.** ROC-AUC, average precision (PR-AUC), R-precision, precision, recall, F1, MCC,
  confusion matrices and permutation importance, benchmarked against unsupervised heuristics and across
  datasets (`src/benchmark.py`).
* **Novel interaction ranking** with gene symbols, plus a check for hub bias.
* **Tested:** 83 `pytest` tests and GitHub Actions CI, which downloads and models HI-III on every push.

## Data

| dataset | type | proteins | interactions | source |
|---|---|---:|---:|---|
| **HI-III** (default) | binary, yeast two-hybrid | 5,216 | 22,195 | CCSB systematic screen, published as HuRI (Luck et al., 2020) |
| HI-II-14 | binary, yeast two-hybrid | 3,985 | 12,883 | Rolland et al., 2014 |
| Lit-BM-13 | binary, literature-curated | 4,909 | 9,865 | Rolland et al., 2014 |
| BioPlex 2.0 | co-complex, AP-MS | 10,801 | 55,862 | Huttlin et al., 2017 |
| BioGRID | mixed | 15,284 | 183,397 | BioGRID human physical interactions |

Counts are after preprocessing. Files come from the repository of Kovács et al. (2019) as Entrez gene ID
edge lists, and gene symbols and types from an NCBI Entrez Gene snapshot. See
[`data/README.md`](data/README.md) for provenance and formats. A small **synthetic** network
(`data/raw/sample_ppi_edges.tsv`) is also bundled for offline tests and demos.

### Preprocessing (HI-III)

| step | interactions removed | interactions left | proteins left |
|---|---:|---:|---:|
| raw file | | 23,322 | 5,604 |
| self-interactions (homodimers) | 322 | 23,000 | 5,579 |
| duplicate / reversed pairs | 0 | 23,000 | 5,579 |
| identifiers missing from annotation | 16 | 22,984 | 5,572 |
| non-protein-coding genes | 672 | 22,312 | 5,417 |
| largest connected component | 117 | 22,195 | 5,216 |

The audits for every dataset are in `reports/data_analysis/preprocessing_steps.csv`.

## Data analysis

The full analysis is in
[`notebooks/01_data_preprocessing_and_eda.ipynb`](notebooks/01_data_preprocessing_and_eda.ipynb) and
[`reports/data_analysis/`](reports/data_analysis/).

* All four maps are sparse (density ≈ 0.1%), small-world (mean distance 3.9-5.3 hops), heavy-tailed and
  disassortative. The Y2H maps have the most extreme hubs: in HI-III, CYSRT1 has 505 partners while the
  median protein has 3.
* The maps overlap moderately in proteins (Jaccard 0.22-0.48) but barely in interactions (about 0.01, or
  0.15 for HI-III vs HI-II-14, which come from the same pipeline). Each map samples a small,
  technology-specific part of the interactome.
* **Triangles vs squares.** Compared with degree-preserving randomisations, Y2H maps are *depleted* in
  shared partners (0.78-0.88×) and enriched in length-3 paths (1.5×). Literature-curated and co-complex
  maps are enriched in both (6-14×). Coverage matters too: even in BioPlex only 38% of interactions have
  a common neighbour, while 83% have a length-3 path.

<p align="center">
  <img src="reports/data_analysis/figures/degree_ccdf.png" width="48%" alt="Degree distributions">
  <img src="reports/data_analysis/figures/interaction_overlap.png" width="40%" alt="Interaction overlap between maps">
</p>

## Results

### HI-III human interactome

Reproduce with `python src/model.py`, which downloads and preprocesses HI-III on first use. The test set has
2,220 held-out interactions plus 2,220 negatives. Outputs are in [`reports/hi-iii/`](reports/hi-iii/) and
the walkthrough is in
[`notebooks/02_feature_and_model_analysis.ipynb`](notebooks/02_feature_and_model_analysis.ipynb).

| Method | Type | ROC-AUC (uniform neg.) | Avg. precision (uniform) | ROC-AUC (local neg.) | Avg. precision (local) |
|---|---|---:|---:|---:|---:|
| Gradient boosting | model | **0.972** | **0.973** | **0.934** | **0.927** |
| Logistic regression | model | 0.971 | 0.973 | 0.925 | 0.916 |
| Random forest | model | 0.971 | 0.971 | 0.932 | 0.925 |
| L3 paths | heuristic | 0.958 | 0.962 | 0.872 | 0.883 |
| Preferential attachment | heuristic | 0.957 | 0.957 | 0.852 | 0.840 |
| Shortest path | heuristic | 0.882 | 0.826 | 0.441 | 0.472 |
| Adamic-Adar | heuristic | 0.751 | 0.755 | 0.503 | 0.581 |
| Common neighbours | heuristic | 0.749 | 0.745 | 0.527 | 0.594 |
| Jaccard | heuristic | 0.735 | 0.682 | 0.364 | 0.404 |

Five-fold cross-validated ROC-AUC on the training pairs is 0.969-0.970 for all three models. Permutation
importance of the best model is dominated by L3: shuffling it costs 0.37 ROC-AUC, more than 25 times any
other feature.

<p align="center">
  <img src="reports/hi-iii/figures/pr_curves.png" width="48%" alt="Precision-recall curves on HI-III">
  <img src="reports/hi-iii/figures/feature_importance.png" width="48%" alt="Permutation importance on HI-III">
</p>

**Candidate novel interactions.** `reports/hi-iii/novel_predictions.csv` lists the 100 unobserved pairs the
best model ranks highest (e.g. KRTAP10-8–KANK2, KRTAP10-3–LCE2A). Almost all of them involve one of
the top-1% hubs, mainly keratin-associated proteins with hundreds of Y2H partners. Notebook 02 therefore
also lists the best-ranked candidates that avoid the largest hubs.

### Benchmark across interactomes

`python src/benchmark.py` runs the full pipeline on four maps with both negative-sampling protocols
([`reports/benchmark/`](reports/benchmark/)).

<p align="center">
  <img src="reports/benchmark/figures/roc_auc_heatmap.png" width="85%" alt="ROC-AUC across datasets and methods">
</p>

* L3 beats common neighbours on every map, including co-complex BioPlex (0.91 vs 0.67).
* With hard negatives, common-neighbour, Jaccard and shortest-path scores fall to chance or below on every
  map. Nearby non-interacting proteins share partners and are as close as true interactors.
* Lit-BM-13 is the exception for L3: on the literature-curated map, preferential attachment is the stronger
  single signal (0.90 vs 0.83 with uniform negatives, 0.79 vs 0.66 with local ones). The trained models
  still reach 0.94.
* The supervised models are the best method in every setting, and their advantage grows with hard negatives
  (+0.06 to +0.26 ROC-AUC over L3).

## Methodology

```
datasets ──► preprocessing ──► graph_builder ──► leakage-free split ──► feature_extractor ──► model ──► evaluate
 download     audited cleaning   NetworkX graph   train/test graphs     15 features/pair      3 classifiers  metrics,
 + verify     + gene symbols                      + negatives                                  + CV           plots, ranking
                   └──► data_analysis (statistics, overlaps, closure vs null models)            benchmark (4 maps × 2 protocols)
```

1. **Split.** Hide 10% of interactions as test positives (`test_graph = G - test`). Then hide another
   10% as training positives (`train_graph = test_graph - train`). Sample the same number of
   non-interacting pairs for each split; train and test negatives are disjoint.
2. **Featurise.** Training pairs are featurised on `train_graph` and test pairs on `test_graph`. This
   simulates predicting interactions that have not been observed yet, and avoids the common leakage in
   which a pair's own edge inflates its common-neighbour or degree features.
3. **Train.** Fit each classifier with 5-fold stratified cross-validation on the training pairs.
4. **Evaluate.** Score test pairs with every model and every heuristic, plot ROC/PR curves, and compute
   permutation importance for the best model.
5. **Predict.** Featurise the top-L3 unobserved pairs on the full network and rank them with the best model.

**L3 score** (degree-normalised paths of length three):

$$\mathrm{L3}(u,v)=\sum_{a,b}\frac{A_{ua}\,A_{ab}\,A_{bv}}{\sqrt{k_a\,k_b}}$$

It is computed for many pairs at once as rows of $A D^{-1/2} A D^{-1/2} A$ with sparse matrices.

**Closure test.** For each interaction, the number of shared partners and of simple length-3 paths
($W^3_{uv} - k_u - k_v + 1$) are compared with the same quantities in degree-preserving randomisations
(double edge swaps; Maslov & Sneppen, 2002).

## Project structure

```
├── data/
│   ├── external/                    # downloaded raw interactomes (git-ignored)
│   ├── processed/                   # cleaned edge lists + audit reports (git-ignored)
│   ├── raw/sample_ppi_edges.tsv     # bundled synthetic network
│   ├── generate_synthetic_ppi.py    # duplication-divergence generator
│   └── README.md                    # provenance, formats, preprocessing
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb   # real data: cleaning audit, statistics, overlaps, closure
│   ├── 02_feature_and_model_analysis.ipynb   # HI-III: features, models, hard negatives, novel predictions
│   └── 03_synthetic_network_exploration.ipynb  # appendix: the synthetic sample
├── reports/
│   ├── data_analysis/               # tables and figures from src/data_analysis.py
│   ├── hi-iii/                      # reference pipeline run on HI-III
│   ├── benchmark/                   # cross-dataset benchmark
│   └── synthetic_sample/            # pipeline on the synthetic network
├── src/
│   ├── datasets.py                  # dataset registry, checksum-verified downloads
│   ├── preprocessing.py             # audited cleaning + gene annotation
│   ├── data_analysis.py             # network statistics, overlaps, closure tests, plots
│   ├── graph_builder.py             # loading, graph construction, leakage-free splits
│   ├── feature_extractor.py         # topological node and pair features
│   ├── model.py                     # classifiers and the end-to-end pipeline (CLI)
│   ├── evaluate.py                  # metrics, plots, evaluation CLI
│   └── benchmark.py                 # pipeline across datasets and protocols
├── tests/                           # pytest suite
├── .github/workflows/ci.yml         # tests + synthetic and HI-III pipeline runs
├── requirements.txt
├── requirements-dev.txt             # + pytest, JupyterLab
└── README.md
```

## Installation and setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Raghayev17889/GNN_for_PPI.git
   cd GNN_for_PPI
   ```

2. Create a virtual environment and install dependencies (Python 3.10+):

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   # for tests and notebooks:
   pip install -r requirements-dev.txt
   ```

Downloads need internet access to `raw.githubusercontent.com` (about 8 MB for all datasets).

## Usage

### 1. Get and preprocess the data

```bash
python src/datasets.py list                     # available datasets
python src/datasets.py download --all           # data/external/, checksum-verified
python src/preprocessing.py --all               # data/processed/<name>.tsv + <name>.report.json
```

Both steps are optional: the pipeline downloads and preprocesses its dataset on first use.

### 2. Analyse the networks

```bash
python src/data_analysis.py                     # writes reports/data_analysis/
```

### 3. Run the pipeline

```bash
python src/model.py                                     # HI-III, writes to results/
python src/model.py --negatives local                   # harder negative pairs
python src/model.py --dataset bioplex-2                 # another registry interactome
python src/model.py --input data/raw/sample_ppi_edges.tsv   # bundled synthetic network
```

| option | default | description |
|---|---|---|
| `--dataset` | `hi-iii` | registry interactome (`hi-iii`, `hi-ii-14`, `lit-bm-13`, `bioplex-2`, `biogrid`) |
| `--input` | – | any edge list (TSV/CSV) or STRING file instead of `--dataset` |
| `--min-score` | none | drop interactions below this confidence |
| `--source-col`, `--target-col`, `--weight-col` | auto | column names for custom tables |
| `--test-fraction`, `--train-fraction` | 0.1, 0.1 | fractions of interactions hidden |
| `--negatives` | `uniform` | `uniform` or `local` negative sampling |
| `--negative-ratio` | 1.0 | negatives per positive |
| `--models` | all three | subset of `logistic_regression random_forest gradient_boosting` |
| `--cv-folds` | 5 | cross-validation folds (0 disables) |
| `--betweenness-samples` | `auto` | `exact`, a sample size, or `auto` (exact up to 3,000 proteins, else 1,000 samples) |
| `--top-novel` | 100 | number of novel interactions to rank (0 skips) |
| `--output-dir` | `results/` | where artifacts are written |
| `--seed` | 42 | controls every random step |

Outputs: `metrics.csv`, `predictions.csv`, `feature_importance.csv`, `novel_predictions.csv` (with gene
symbols), `run_summary.json`, `figures/*.png` and `models/*.joblib`. To re-plot or re-score saved
predictions, run `python src/evaluate.py --predictions results/predictions.csv`.

### 4. Benchmark across datasets

```bash
python src/benchmark.py                          # 4 interactomes × 2 protocols, about 15 minutes
```

### 5. Explore the notebooks

```bash
jupyter lab notebooks/
```

### 6. Use as a library

```python
from src.graph_builder import load_graph, make_link_prediction_split
from src.feature_extractor import TopologicalFeatureExtractor
from src.model import train_models
from src.preprocessing import processed_path

G = load_graph(processed_path("hi-iii"))           # downloads + preprocesses on first use
split = make_link_prediction_split(G, negative_strategy="local", seed=0)
extractor = TopologicalFeatureExtractor(betweenness_samples=1000)
X_train = extractor.fit_transform(split.train_graph, split.train_pairs)
X_test = TopologicalFeatureExtractor(betweenness_samples=1000).fit_transform(split.test_graph, split.test_pairs)
models = train_models(X_train, split.train_labels)
probabilities = models["gradient_boosting"].predict_proba(X_test)
```

### 7. Run the tests

```bash
pytest
```

The tests use small in-memory graphs and need no downloads.

## Limitations and next steps

* Held-out edges from the same screen are an easier target than truly new interactions, and every map is
  shaped by its assay (Y2H hubs, literature study bias, complex-level AP-MS). Predictions need
  experimental validation.
* Top-ranked novel candidates favour hubs. Degree-aware calibration or degree-matched negatives would be a
  useful extension.
* Topology ignores sequence, structure and expression. Sequence embeddings (e.g. ESM) or Gene Ontology
  similarity as pair features, and a graph neural network (GNN) baseline trained on the same leakage-free
  split, are natural next steps. A further test is to train on one map and evaluate on another.
* The gene annotation is a 2016 NCBI snapshot. A current HGNC or NCBI release would recover a few more
  discontinued identifiers.

## Technical stack

* **Language:** Python 3.10+
* **Graph analysis:** NetworkX, SciPy (sparse matrices, `csgraph`)
* **Machine learning:** scikit-learn, pandas, NumPy
* **Visualisation:** Matplotlib, Seaborn
* **Quality:** pytest, GitHub Actions

## References

* Kovács, I. A. *et al.* Network-based prediction of protein interactions. *Nature Communications* 10,
  1240 (2019).
* Luck, K. *et al.* A reference map of the human binary protein interactome. *Nature* 580, 402-408 (2020).
* Rolland, T. *et al.* A proteome-scale map of the human interactome network. *Cell* 159, 1212-1226 (2014).
* Huttlin, E. L. *et al.* Architecture of the human interactome defines protein communities and disease
  networks. *Nature* 545, 505-509 (2017).
* Maslov, S. & Sneppen, K. Specificity and stability in topology of protein networks. *Science* 296,
  910-913 (2002).
* Liben-Nowell, D. & Kleinberg, J. The link-prediction problem for social networks. *JASIST* 58,
  1019-1031 (2007).
* Vázquez, A., Flammini, A., Maritan, A. & Vespignani, A. Modeling of protein interaction networks.
  *ComPlexUs* 1, 38-44 (2003).

## License

The code is open source and available under the [MIT License](LICENSE). The interaction datasets
belong to their original authors and are downloaded from their published sources, not redistributed here.
Please cite the original publications when you use them.
