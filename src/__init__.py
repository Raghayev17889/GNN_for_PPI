"""Graph-based protein-protein interaction (PPI) prediction.

Modules
-------
datasets
    Registry and checksum-verified downloads of public human interactomes.
preprocessing
    Audited cleaning of raw interaction data with gene annotation.
data_analysis
    Network statistics, overlaps between maps and closure tests.
graph_builder
    Load PPI edge lists, build NetworkX graphs and leakage-free splits.
feature_extractor
    Topological node and node-pair features for link prediction.
model
    Supervised classifiers and the end-to-end training pipeline.
evaluate
    Evaluation metrics and plots.
benchmark
    The pipeline across datasets and negative-sampling protocols.
"""

__version__ = "0.2.0"
