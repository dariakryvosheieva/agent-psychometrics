"""Experiment B: Frontier Task Difficulty Prediction.

Predicts difficulty of frontier tasks (tasks only solvable by newer models)
using various methods WITHOUT access to held-out post-frontier agents.

Setting:
- Date-based split: pre-frontier (< 20250807) vs post-frontier (>= 20250807)
- Frontier tasks: ≤10% pre-frontier pass rate, >10% post-frontier pass rate
- Evaluation: ROC-AUC after projecting predicted difficulties onto oracle IRT scale

Methods compared:
- Oracle (upper bound): Use true IRT difficulties
- Baseline IRT: Train IRT on pre-frontier agents only
- Embedding + Ridge: Task embeddings from any backbone model
- LLM Judge + Ridge: LLM-extracted semantic features
- (Optional) SAD-IRT: State-aware deep IRT from experiment_sad_irt

Usage:
    python -m experiment_b.compare_methods --output_csv chris_output/experiment_b_results.csv

See README.md for full documentation.
"""
