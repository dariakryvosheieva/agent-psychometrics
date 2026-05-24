"""Subset Extrapolation Experiment.

Benchmark designers run agents on a randomly chosen subset of S < N tasks and
want to predict each agent's overall score on the full benchmark. This module
compares the naive empirical-subset average against the IRT + features pipeline
from experiment_new_tasks/.

Entry point:
- python -m experiment_subset_extrapolation.run_all_datasets
"""
