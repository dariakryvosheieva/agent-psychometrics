"""Shared utilities for Experiment B."""

# Config
from experiment_b.shared.config_base import DatasetConfig

# Data preparation
from experiment_b.shared.data_preparation import (
    ExperimentData,
    split_agents_by_dates,
    compute_pass_rates,
    identify_frontier_tasks_passrate,
    identify_frontier_tasks_irt,
    identify_frontier_tasks_zero_pre,
    identify_nontrivial_tasks,
    get_all_agents_from_responses,
    get_agents_with_trajectories,
    get_pre_frontier_agents,
    get_or_train_baseline_irt,
    load_and_prepare_data,
)

# Evaluation
from experiment_b.shared.evaluation import (
    load_responses_dict,
    analyze_scale_alignment,
    compute_scale_offset,
    shift_to_oracle_scale,
    compute_frontier_auc,
    compute_method_metrics,
    compute_mean_per_agent_auc,
    filter_agents_with_frontier_variance,
    evaluate_predictor,
)

# Prediction methods
from experiment_b.shared.prediction_methods import (
    FeatureIRTPredictor,
    FeatureIRTResults,
    build_feature_sources,
    collect_ridge_predictions,
    collect_grouped_ridge_predictions,
    collect_feature_irt_predictions,
    collect_sad_irt_predictions,
)

# Frontier evaluation
from experiment_b.shared.frontier_evaluation import (
    DateForecastingData,
    setup_date_forecasting,
    evaluate_all_frontier_definitions,
)

# Output formatting
from experiment_b.shared.output_formatting import (
    print_comparison_table,
    save_results_csv,
    print_date_forecast_table,
)

# Diagnostics
from experiment_b.shared.diagnostics import (
    plot_grid_search_heatmap,
    plot_training_loss_curves,
    plot_loss_components,
    print_diagnostic_summary,
    save_and_plot_diagnostics,
)

# Date forecasting (kept as separate utility module)
from experiment_b.shared.date_forecasting import (
    parse_date,
    FirstCapableDatesResult,
    compute_first_capable_dates,
    split_tasks_by_first_capable_date,
    compute_ground_truth_days,
    AbilityOverTimeResult,
    fit_ability_over_time,
    DateForecastModel,
    compute_date_forecast_metrics,
)

__all__ = [
    # config_base
    "DatasetConfig",
    # data_preparation
    "ExperimentData",
    "split_agents_by_dates",
    "compute_pass_rates",
    "identify_frontier_tasks_passrate",
    "identify_frontier_tasks_irt",
    "identify_frontier_tasks_zero_pre",
    "identify_nontrivial_tasks",
    "get_all_agents_from_responses",
    "get_agents_with_trajectories",
    "get_pre_frontier_agents",
    "get_or_train_baseline_irt",
    "load_and_prepare_data",
    # evaluation
    "load_responses_dict",
    "analyze_scale_alignment",
    "compute_scale_offset",
    "shift_to_oracle_scale",
    "compute_frontier_auc",
    "compute_method_metrics",
    "compute_mean_per_agent_auc",
    "filter_agents_with_frontier_variance",
    "evaluate_predictor",
    # prediction_methods
    "FeatureIRTPredictor",
    "FeatureIRTResults",
    "build_feature_sources",
    "collect_ridge_predictions",
    "collect_grouped_ridge_predictions",
    "collect_feature_irt_predictions",
    "collect_sad_irt_predictions",
    # frontier_evaluation
    "DateForecastingData",
    "setup_date_forecasting",
    "evaluate_all_frontier_definitions",
    # output_formatting
    "print_comparison_table",
    "save_results_csv",
    "print_date_forecast_table",
    # diagnostics
    "plot_grid_search_heatmap",
    "plot_training_loss_curves",
    "plot_loss_components",
    "print_diagnostic_summary",
    "save_and_plot_diagnostics",
    # date_forecasting
    "parse_date",
    "FirstCapableDatesResult",
    "compute_first_capable_dates",
    "split_tasks_by_first_capable_date",
    "compute_ground_truth_days",
    "AbilityOverTimeResult",
    "fit_ability_over_time",
    "DateForecastModel",
    "compute_date_forecast_metrics",
]
