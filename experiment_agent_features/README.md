# Agent Feature Experiments (Sections 4.2-4.3)

Predicts agent-task outcomes using **LLM-scaffold decomposition**: agent ability = LLM ability + scaffold ability (additive in the IRT model). This allows the model to generalize to new agents, new observations, and entirely new benchmarks.

## Quick Start

```bash
source .venv/bin/activate

# New Responses (Section 4.2, Table 3)
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by observation \
    --train_benchmarks verified,pro,terminal_bench,gso \
    --out_dir data/held_out_responses

# New Agents (Section 4.3, Table 4)
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by agent \
    --train_benchmarks verified

# New Benchmark (Section 4.3, Table 5)
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by benchmark \
    --method combined \
    --train_benchmarks verified,pro,terminal_bench \
    --ood_benchmark gso \
    --out_dir data/held_out_benchmark
```

## Experiments

| Experiment | Flag | Description |
|------------|------|-------------|
| New Responses | `--split_by observation` | Predict individual agent-task outcomes (random hold-out) |
| New Agents | `--split_by agent` | Predict performance of unseen agents |
| New Benchmark | `--split_by benchmark` | Predict on an entirely new benchmark |

## Feature Methods

| Method | Flag | Description |
|--------|------|-------------|
| Embedding | `--method embedding` | Task embeddings (DeepSeek-R1-Distill-Qwen-32B) |
| LLM Judge | `--method judge` | LLM-extracted semantic features (15 unified features) |
| Combined | `--method combined` | Both embedding and LLM judge features |

Embedding generation requires a GPU. Use `--embeddings_cache` with pre-computed embeddings from `embeddings/` to skip generation.

## Files

| File | Purpose |
|------|---------|
| `predict_question_difficulty.py` | Single-benchmark prediction on new tasks (imported as `base` by `predict_question_difficulty_multi_benchmark.py`) |
| `predict_question_difficulty_multi_benchmark.py` | Multi- and single-benchmark experiments with agent features |
| `run_single.sh` | SLURM script for `predict_question_difficulty.py` |
| `run_multi.sh` | SLURM script for `predict_question_difficulty_multi_benchmark.py` |
| `terminalbench_scatterplot.py` | Generates validation scatter plot (Figure 3) |
