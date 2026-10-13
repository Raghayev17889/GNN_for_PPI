"""Graph-based protein-protein interaction (PPI) prediction.

Modules
-------
graph_builder
    Load PPI edge lists and build NetworkX graphs.
feature_extractor
    Topological node and node-pair features for link prediction.
model
    Supervised classifiers and the end-to-end training pipeline.
evaluate
    Evaluation metrics and plots.
"""

__version__ = "0.1.0"
