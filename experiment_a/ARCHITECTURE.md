# Experiment A Architecture

This document describes the class hierarchy and data flow for Experiment A (Prior Validation).

## Class Hierarchy & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ embeddings.npz   │    │ llm_features.csv │    │ responses.jsonl          │   │
│  │ (5120-dim)       │    │ (9-10 features)  │    │ (agent × task matrix)    │   │
│  └────────┬─────────┘    └────────┬─────────┘    └───────────┬──────────────┘   │
│           │                       │                          │                   │
└───────────┼───────────────────────┼──────────────────────────┼───────────────────┘
            ▼                       ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE SOURCES                                        │
│                    (experiment_ab_shared/feature_source.py)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  TaskFeatureSource (ABC)                                                         │
│  ├── get_features(task_ids) → np.ndarray                                        │
│  ├── task_ids, feature_dim, feature_names                                       │
│  │                                                                               │
│  │   ┌─────────────────────────┐     ┌─────────────────────────┐                │
│  ├──►│ EmbeddingFeatureSource  │     │   CSVFeatureSource      │                │
│  │   │ (loads .npz files)      │     │   (loads CSV columns)   │                │
│  │   └───────────┬─────────────┘     └───────────┬─────────────┘                │
│  │               │                               │                               │
│  │               ▼                               ▼                               │
│  │   ┌─────────────────────────────────────────────────────────┐                │
│  │   │              RegularizedFeatureSource                    │                │
│  │   │   (wraps source + alpha regularization strength)         │                │
│  │   └───────────────────────┬─────────────────────────────────┘                │
│  │                           │                                                   │
│  │                           ▼                                                   │
│  │   ┌─────────────────────────────────────────────────────────┐                │
│  └──►│              GroupedFeatureSource                        │                │
│      │   (combines multiple RegularizedFeatureSources)          │                │
│      │   e.g., [Embeddings(α=10000), LLM(α=100)]                │                │
│      └─────────────────────────────────────────────────────────┘                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PREDICTORS                                          │
│                   (experiment_ab_shared/feature_predictor.py)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DifficultyPredictorBase (ABC)                                                   │
│  ├── fit(task_ids, ground_truth_β)                                              │
│  ├── predict(task_ids) → Dict[task_id, β̂]                                       │
│  │                                                                               │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │
│  ├─►│ FeatureBasedPredictor                                                 │    │
│  │  │  source: TaskFeatureSource                                            │    │
│  │  │  method: "ridge" | "lasso"                                            │    │
│  │  │  _scaler: StandardScaler → _model: RidgeCV/LassoCV                   │    │
│  │  └──────────────────────────────────────────────────────────────────────┘    │
│  │                                                                               │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │
│  ├─►│ GroupedRidgePredictor                                                 │    │
│  │  │  source: GroupedFeatureSource                                         │    │
│  │  │  Grid search over per-source alphas                                   │    │
│  │  │  Feature scaling: StandardScaler → per-group 1/sqrt(α) scaling       │    │
│  │  └──────────────────────────────────────────────────────────────────────┘    │
│  │                                                                               │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │
│  └─►│ StackedResidualPredictor                                              │    │
│     │  base_source → RidgeCV → β̂_base                                       │    │
│     │  residual_source → RidgeCV → β̂_residual (target = β_true - β̂_base)   │    │
│     │  Final: β̂ = β̂_base + β̂_residual                                       │    │
│     └──────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CV PREDICTOR PROTOCOL                                    │
│                     (experiment_a/shared/cross_validation.py)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CVPredictor (Protocol)                                                          │
│  ├── fit(data: ExperimentData, train_task_ids)                                  │
│  ├── predict_probability(data, agent_id, task_id) → P(success)                  │
│  │                                                                               │
│  │  ┌─────────────────────────────────────────────────────────────┐             │
│  ├─►│ OraclePredictor                                              │             │
│  │  │  Uses full_abilities (θ) and full_items (β) directly        │             │
│  │  │  P = sigmoid(θ_oracle - β_true)                             │             │
│  │  └─────────────────────────────────────────────────────────────┘             │
│  │                                                                               │
│  │  ┌─────────────────────────────────────────────────────────────┐             │
│  ├─►│ DifficultyPredictorAdapter                                   │             │
│  │  │  Wraps any DifficultyPredictorBase                          │             │
│  │  │  fit(): extracts β from train_items, calls wrapped.fit()    │             │
│  │  │  predict_probability(): P = sigmoid(θ - β̂_predicted)        │             │
│  │  └─────────────────────────────────────────────────────────────┘             │
│  │                                                                               │
│  │  ┌─────────────────────────────────────────────────────────────┐             │
│  ├─►│ AgentOnlyPredictor (baseline)                                │             │
│  │  │  Returns agent's empirical success rate (ignores task)       │             │
│  │  └─────────────────────────────────────────────────────────────┘             │
│  │                                                                               │
│  │  ┌─────────────────────────────────────────────────────────────┐             │
│  └─►│ ConstantPredictor (baseline)                                 │             │
│     │  P = sigmoid(θ - β_mean)                                     │             │
│     └─────────────────────────────────────────────────────────────┘             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT DATA                                        │
│                      (experiment_ab_shared/dataset.py)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ExperimentData[ResponseT] (ABC, Generic)                                        │
│  ├── responses: Dict[agent_id, Dict[task_id, response]]                         │
│  ├── train_abilities: DataFrame (θ from train-only IRT)                         │
│  ├── train_items: DataFrame (β from train-only IRT)                             │
│  ├── full_abilities: DataFrame (θ from full IRT - oracle)                       │
│  ├── full_items: DataFrame (β from full IRT - oracle)                           │
│  ├── expand_for_auc(agent, task, prob) → (y_true, y_scores)                     │
│  │                                                                               │
│  │   ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  ├──►│ BinaryExperimentData     │    │ BinomialExperimentData   │               │
│  │   │ response = 0 | 1         │    │ response = {success, n}  │               │
│  │   │ (SWE-bench)              │    │ (TerminalBench)          │               │
│  │   └──────────────────────────┘    └──────────────────────────┘               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          CROSS-VALIDATION                                        │
│                    (experiment_a/shared/cross_validation.py)                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  k_fold_split_tasks(task_ids, k=5) → [(train_ids, test_ids), ...]               │
│                                                                                  │
│  run_cv(predictor, folds, load_fold_data):                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  For each fold:                                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │  1. Load fold data (with train-only IRT model)                  │    │    │
│  │  │  2. predictor.fit(data, train_task_ids)                         │    │    │
│  │  │  3. For each (agent, task) in test set:                         │    │    │
│  │  │     prob = predictor.predict_probability(data, agent, task)     │    │    │
│  │  │  4. Compute fold AUC                                            │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                          │    │
│  │  Return: {mean_auc, std_auc, fold_aucs}                                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE ORCHESTRATION                                   │
│                       (experiment_a/shared/pipeline.py)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  run_cross_validation(config, spec):                                             │
│                                                                                  │
│  1. Load oracle IRT ─────────────────────────────────────────────────────────►  │
│                                                                                  │
│  2. k_fold_split_tasks() ────────────────────────────────────────────────────►  │
│                                                                                  │
│  3. build_cv_predictors():                                                       │
│     ├── OraclePredictor                                                          │
│     ├── DifficultyPredictorAdapter(FeatureBasedPredictor(embeddings))           │
│     ├── DifficultyPredictorAdapter(FeatureBasedPredictor(llm_judge))            │
│     ├── DifficultyPredictorAdapter(GroupedRidgePredictor(emb + llm))            │
│     ├── DifficultyPredictorAdapter(StackedResidualPredictor(emb → llm))         │
│     ├── DifficultyPredictorAdapter(StackedResidualPredictor(llm → emb))         │
│     ├── ConstantPredictor                                                        │
│     └── AgentOnlyPredictor                                                       │
│                                                                                  │
│  4. For each predictor: run_cv() ────────────────────────────────────────────►  │
│                                                                                  │
│  5. Save results to JSON ────────────────────────────────────────────────────►  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## End-to-End Data Flow Summary

```
┌────────────────────┐
│  Raw Data Files    │
│  - embeddings.npz  │
│  - llm_features.csv│
│  - responses.jsonl │
│  - IRT models      │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Feature Sources   │  TaskFeatureSource → get_features(task_ids)
│  (Embedding, CSV,  │
│   Grouped)         │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Predictors        │  fit(task_ids, β_true) → predict(task_ids) → β̂
│  (Ridge, Grouped,  │
│   Stacked)         │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  CV Adapters       │  DifficultyPredictorAdapter wraps predictors
│  (implements       │  predict_probability() = sigmoid(θ - β̂)
│   CVPredictor)     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Cross-Validation  │  5-fold CV with train-only IRT per fold
│  run_cv()          │  Collects predictions, computes AUC
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Results           │  {predictor_name: {mean_auc, std_auc}}
│  JSON output       │
└────────────────────┘
```

## Key Files

| Component | File |
|-----------|------|
| Entry point | `train_evaluate.py` |
| Pipeline orchestration | `shared/pipeline.py` |
| CV framework | `shared/cross_validation.py` |
| Baselines/Oracle | `shared/baselines.py` |
| Feature sources | `../experiment_ab_shared/feature_source.py` |
| Predictors | `../experiment_ab_shared/feature_predictor.py` |
| Dataset classes | `../experiment_ab_shared/dataset.py` |

## Key Design Patterns

### 1. Abstract Base Classes with Composition
- `TaskFeatureSource` is extended by concrete types (Embedding, CSV)
- `GroupedFeatureSource` contains multiple `RegularizedFeatureSource` instances
- Allows flexible feature stacking without code duplication

### 2. Adapter Pattern
- `DifficultyPredictorAdapter` adapts `DifficultyPredictorBase` to implement `CVPredictor`
- Decouples predictor logic from CV framework
- Allows mixing different predictor types in same CV run

### 3. Protocol-Based Polymorphism
- `CVPredictor` is a Protocol (structural subtyping)
- Any class implementing `fit()` and `predict_probability()` works
- Enables adding new predictors without modifying existing code

### 4. Fail-Loudly Philosophy
- `get_features()` raises `ValueError` if task is missing
- No silent skipping or NaN handling
- Makes debugging easier by catching issues at source

### 5. Two-Stage IRT Training
- **Train-only IRT**: Built per-fold from train tasks only
- **Full IRT**: Built once from all tasks (cached)
- Prevents data leakage and provides oracle baseline

## IRT Formula

The core IRT model uses:

```
P(success) = sigmoid(θ - β)
```

Where:
- `θ` = agent ability (higher = more capable)
- `β` = task difficulty (higher = harder)
- `sigmoid(x) = 1 / (1 + exp(-x))`
