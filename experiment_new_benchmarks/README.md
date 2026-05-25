# Experiment New Benchmarks

Train on three benchmarks and evaluate on a fourth, held-out benchmark. The
default runner holds out SWE-bench Pro and GSO.

```bash
python -m experiment_new_benchmarks.run_all_datasets --sequential
```

For each held-out benchmark, the experiment:

1. trains a model + scaffold IRT on the other three benchmarks;
2. fits embedding, LLM-judge, and combined difficulty regressors from
   train-benchmark task features to train-benchmark IRT difficulties;
3. evaluates held-out responses whose LLM and scaffold were observed in the
   training benchmarks;
4. compares each method against a baseline that predicts the LLM's empirical
   training success rate, ignoring scaffold.

Significance is computed with paired clustered bootstrapping of
`AUC(method) - AUC(baseline)`, clustering held-out observations by task.

