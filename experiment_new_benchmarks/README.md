# Experiment New Benchmarks

Train on three benchmarks and evaluate on a fourth, held-out benchmark. The
default runner holds out SWE-bench Pro and GSO.

```bash
python -m experiment_new_benchmarks.run_all_datasets
```

For each held-out benchmark, the experiment:

1. trains a model + scaffold IRT on the other three benchmarks;
2. fits embedding, LLM-judge, and combined difficulty regressors from
   train-benchmark task features to train-benchmark IRT difficulties;
3. evaluates held-out responses whose LLM and scaffold were observed in the
   training benchmarks;
4. compares each method against a baseline that predicts the LLM's empirical
   training success rate, ignoring scaffold. An Oracle skyline (the canonical
   full single-benchmark IRT on the held-out benchmark) is also reported.

The LLM-as-a-judge held-out item difficulty predictions are written as
`{output_dir}/{heldout_dataset}/predictions.csv` with columns
`item_id,diff_pred,split,fold`. When running a single held-out benchmark, the
same file is also copied to `{output_dir}/predictions.csv` for downstream tools
such as adaptive task selection.

Significance is computed with paired clustered bootstrapping of
`AUC(method) - AUC(baseline)`, clustering held-out observations by task. The
resulting p-values are then Holm-Bonferroni corrected across the family of
(held-out benchmark, method) comparisons (see `HOLM_FAMILY_METHODS` in
`run_all_datasets.py`; the adjustment lives in
`experiment_new_tasks/bootstrap.py`).

