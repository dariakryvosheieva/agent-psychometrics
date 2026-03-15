# Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks

We present a framework for predicting success or failure on individual tasks tailored to the agentic coding regime. Our approach augments **Item Response Theory (IRT)** with rich features extracted from tasks, including issue statements, repository contexts, solutions, and test cases, and introduces a novel decomposition of agent ability into **LLM and scaffold ability components**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e87e7dec-b1c4-4a77-beb0-5e5bde815f57" alt="Agent Psychometrics" width="600">
</p>

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run Experiment New Tasks on all datasets
python -m experiment_new_tasks.run_all_datasets

# Run Experiment New Responses
python experiment_agent_features.predict_question_difficulty_multi_benchmark.py \
    --split_by observation \
    --train_benchmarks verified,terminalbench,pro,gso

# Run Experiment New Agents
python experiment_agent_features.predict_question_difficulty_multi_benchmark.py \
    --split_by agent \
    --train_benchmarks verified

# Run Experiment New Benchmarks
python experiment_agent_features.predict_question_difficulty_multi_benchmark.py \
    --split_by benchmark \
    --train_benchmarks verified,terminalbench,pro \
    --ood_benchmark gso \
    --method judge

# Run Appendix H Hard Tasks (frontier task difficulty prediction)
python -m experiment_appendix_h_hard_tasks.compare_methods

# Train IRT model
python swebench_irt/train.py --dims 1 --model 1pl \
    --data_path data/swebench_verified/responses.jsonl
```

## Repository Structure

```
agent-psychometrics/
├── experiment_new_tasks/              # New Tasks experiment
├── experiment_agent_features/         # Experiments involving agent features: New Responses, New Agents, and New Benchmarks
├── experiment_appendix_h_hard_tasks/  # Frontier task difficulty prediction
│   ├── trajectory_data/               #   Downloaded trajectories (76 agents)
│   ├── trajectory_summarization_api/  #   Trajectory summarization
│   └── trajectory_upload/             #   Trajectory conversion and upload
├── llm_judge_feature_extraction/      # LLM-as-a-judge feature extraction
│   └── auditor_agent/                 #   Repository state feature extraction
├── llm_judge_features/                # LLM-as-a-judge feature CSV files
│   ├── backbone_ablation/             #   Features from GPT-5.4 and Claude 4.6 Sonnet (Appendix C.2)
│   ├── defaults/                      #   Features used in main experiments
│   └── information_ablation/          #   Features used in the feature source ablation experiment (Table 3)
├── swebench_irt/                      # IRT model training
├── py_irt/                            # IRT library (local fork)
└── data/                              # Input data + IRT models
```

## Datasets

All input data lives under `data/{dataset}/`:

| Dataset | Tasks | Agents | Response Matrix | IRT Model |
|---------|-------|--------|----------------|-----------|
| SWE-bench Verified | 500 | 134 | `data/swebench_verified/responses.jsonl` | `data/swebench_verified/irt/1d_1pl/` |
| GSO | 102 | 15 | `data/gso/responses.jsonl` | `data/gso/irt/1d_1pl/` |
| TerminalBench | 89 | 112 | `data/terminalbench/responses.jsonl` | `data/terminalbench/irt/1d_1pl/` |
| SWE-bench Pro | 730 | 14 | `data/swebench_pro/responses.jsonl` | `data/swebench_pro/irt/1d_1pl/` |

## Documentation

| Document | Purpose |
|----------|---------|
| [experiment_new_tasks/README.md](experiment_new_tasks/README.md) | Experiment New Tasks details |
| [experiment_appendix_h_hard_tasks/README.md](experiment_appendix_h_hard_tasks/README.md) | Appendix H Hard Tasks details |
| [llm_judge_feature_extraction/README.md](llm_judge_feature_extraction/README.md) | LLM judge feature extraction |
