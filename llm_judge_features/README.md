# LLM Judge Features

Pre-computed LLM judge feature CSVs organized by use case.

## Directory Structure

```
llm_judge_features/
├── defaults/                          # Used by run_all_datasets.py (Table 2)
│   ├── swebench_verified/llm_judge_features.csv   (15 features)
│   ├── gso/llm_judge_features.csv
│   ├── swebench_pro/llm_judge_features.csv
│   └── terminalbench/llm_judge_features.csv
│
├── information_ablation/              # Features for info-level ablation (Tables 3 & 10)
│   ├── source/                        # 28-feature source CSVs (input to ablation)
│   │   ├── swebench_verified.csv
│   │   ├── gso.csv
│   │   ├── swebench_pro.csv
│   │   └── terminalbench.csv
│   ├── per_level_source/              # Per-level override source CSVs
│   │   └── {dataset}/{level}.csv     # level in {problem, environment, test, solution, problem_solution}
│   ├── swebench_verified/             # Generated top-15 CSVs per ablation row
│   │   ├── 1_problem_15.csv                          # Problem statement only
│   │   ├── 2_problem_auditor_15.csv                  # Problem + repo state
│   │   ├── 3_problem_auditor_test_15.csv             # Problem + repo + tests (= Full minus solution)
│   │   ├── 4_full_15.csv                             # Full (all four sources)
│   │   ├── 5_problem_auditor_solution_15.csv         # Full minus tests
│   │   └── 6_problem_test_solution_15.csv            # Full minus repo state
│   ├── gso/
│   ├── swebench_pro/
│   └── terminalbench/
│
└── backbone_ablation/                 # Different LLM backbones (Appendix C.2)
    ├── v3_gpt54_solution/             # GPT 5.4 features (28 features)
    │   ├── swebench_verified/llm_judge_features.csv
    │   ├── gso/llm_judge_features.csv
    │   ├── swebench_pro/llm_judge_features.csv
    │   └── terminalbench/llm_judge_features.csv
    └── v5_sonnet_solution/            # Sonnet 4.6 features (28 features)
        └── ... (same 4 datasets)
```

## Feature Sets

### Defaults
15 features selected via greedy forward selection to maximize cross-dataset mean AUC.
Backbone: Claude Opus 4.6, solution override level.

### Information Ablation
28 features at natural information levels (no leakage). Source CSVs combine
20 Opus 4.6 judge features + 8 GPT 5.4 auditor features. Per-dataset CSVs
are top-15 selections for two ablations: the hierarchical ablation
(rows 1-4) that adds sources in order of increasing access, and the
leave-one-out ablation (rows 5-6, plus row 3 reused as "Full minus solution")
that drops a single source from the full pool. Both ablations use the same
28-feature underlying source CSVs.

### Backbone Ablation
28 features (20 judge + 8 GPT 5.4 auditor) from alternative LLM backbones:
- **v3_gpt54_solution**: GPT 5.4, solution override level
- **v5_sonnet_solution**: Sonnet 4.6, solution override level
