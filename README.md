# Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks

<!-- TODO: Replace with arxiv link once available -->
<!-- [Paper](https://arxiv.org/abs/XXXX.XXXXX) -->

We present a framework for predicting success or failure on individual tasks tailored to the agentic coding regime. Our approach augments **Item Response Theory (IRT)** with rich features extracted from tasks, including issue statements, repository contexts, solutions, and test cases, and introduces a novel decomposition of agent ability into **LLM and scaffold ability components**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e87e7dec-b1c4-4a77-beb0-5e5bde815f57" alt="Agent Psychometrics" width="600">
</p>

## Quick Start

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Experiment New Tasks on all datasets
python -m experiment_new_tasks.run_all_datasets

# Run Experiment New Responses
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by observation \
    --train_benchmarks verified,terminalbench,pro,gso \
    --out_dir held_out_responses

# Run Experiment New Agents
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by agent \
    --train_benchmarks verified \
    --out_dir held_out_agents

# Run Experiment New Benchmarks
python -m experiment_agent_features.predict_question_difficulty_multi_benchmark \
    --split_by benchmark \
    --train_benchmarks verified,terminalbench,pro \
    --ood_benchmark gso \
    --out_dir data/held_out_benchmark \
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
| SWE-bench Pro | 730 | 14 | `data/swebench_pro/responses.jsonl` | `data/swebench_pro/irt/1d_1pl/` |
| GSO | 102 | 15 | `data/gso/responses.jsonl` | `data/gso/irt/1d_1pl/` |
| Terminal-Bench 2.0 | 89 | 112 | `data/terminalbench/responses.jsonl` | `data/terminalbench/irt/1d_1pl/` |

## Documentation

| Document | Purpose |
|----------|---------|
| [experiment_new_tasks/README.md](experiment_new_tasks/README.md) | Experiment New Tasks details |
| [experiment_agent_features/README.md](experiment_agent_features/README.md) | Agent feature experiments |
| [experiment_appendix_h_hard_tasks/README.md](experiment_appendix_h_hard_tasks/README.md) | Appendix H Hard Tasks details |
| [llm_judge_feature_extraction/README.md](llm_judge_feature_extraction/README.md) | LLM judge feature extraction |

## Key Files

| File | Purpose |
|------|---------|
| `experiment_new_tasks/run_all_datasets.py` | Run Experiment New Tasks |
| `experiment_agent_features/predict_question_difficulty_multi_benchmark.py` | Agent feature experiments (Tables 3-5) |
| `swebench_irt/train.py` | Train IRT models |

## Citation

<!-- TODO: Fill in once the paper is available on arxiv/Google Scholar -->
```bibtex
@inproceedings{agent-psychometrics,
    title={Agent Psychometrics: Task-Level Performance Prediction in Agentic Coding Benchmarks},
    author={},
    booktitle={ICLR},
    year={2026}
}
```
