# LLM Judge Features

Pre-computed LLM judge feature CSVs organized by use case.

## Directory Structure

```
llm_judge_features/
├── defaults/                          # v7_unified_15 — used by run_all_datasets.py
│   ├── swebench_verified/llm_judge_features.csv   (15 features)
│   ├── gso/llm_judge_features.csv
│   ├── swebench_pro/llm_judge_features.csv
│   └── terminalbench/llm_judge_features.csv
│
├── information_ablation/              # v8 features for info-level ablation
│   ├── source/                        # 28-feature v8+auditor CSVs (input to ablation)
│   │   ├── swebench_verified.csv
│   │   ├── gso.csv
│   │   ├── swebench_pro.csv
│   │   └── terminalbench.csv
│   ├── swebench_verified/             # Generated top-15 CSVs per level
│   │   ├── 1_problem_15.csv
│   │   ├── 2_problem_auditor_15.csv
│   │   ├── 3_problem_auditor_test_15.csv
│   │   └── 4_full_15.csv
│   ├── gso/
│   ├── swebench_pro/
│   └── terminalbench/
│
└── backbone_ablation/                 # Different LLM backbones
    ├── v3_gpt54_solution/             # GPT 5.4 (28 features)
    │   ├── swebench_verified/llm_judge_features.csv
    │   ├── gso/llm_judge_features.csv
    │   ├── swebench_pro/llm_judge_features.csv
    │   └── terminalbench/llm_judge_features.csv
    └── v5_sonnet_solution/            # Sonnet 4.6 (28 features)
        └── ... (same 4 datasets)
```

## Feature Sets

### Defaults (v7_unified_15)
15 features selected via greedy forward selection to maximize cross-dataset mean AUC.
Backbone: Claude Opus 4.6, solution override level.

### Information Ablation (v8)
28 features at natural information levels (no leakage). Source CSVs combine
20 Opus 4.6 judge features + 8 GPT 5.4 auditor features. Per-dataset CSVs
are top-15 selections at each cumulative info level (Problem → +Auditor → +Test → +Solution).

### Backbone Ablation
28 features (20 judge + 8 GPT 5.4 auditor) from alternative LLM backbones:
- **v3_gpt54_solution**: GPT 5.4, solution override level
- **v5_sonnet_solution**: Sonnet 4.6, solution override level
