"""Main evolution loop orchestration and CLI."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import EvolutionConfig
from .data_loader import DataLoader
from .feature_evaluator import FeatureEvaluator
from .feature_generator import FeatureGenerator
from .feature_refiner import FeatureRefiner
from .feature_store import (
    Feature,
    FeatureEvaluation,
    FeatureStore,
    GenerationSummary,
)
from .llm_client import LLMClient
from .selection import FeatureSelector


class EvolutionRunner:
    """Orchestrates the evolutionary feature discovery process."""

    def __init__(self, config: EvolutionConfig):
        """Initialize evolution runner.

        Args:
            config: Evolution configuration.
        """
        self.config = config
        self.store = FeatureStore(config.output_dir)
        self.data_loader = DataLoader(config.items_path)
        self.llm_client = LLMClient(
            model=config.model,
            api_delay=config.api_delay,
            max_retries=config.max_retries,
        )
        self.generator = FeatureGenerator(config, self.data_loader, self.llm_client)
        self.evaluator = FeatureEvaluator(self.llm_client)
        self.refiner = FeatureRefiner(config, self.llm_client)
        self.selector = FeatureSelector(
            top_k=config.top_k,
            redundancy_threshold=config.redundancy_threshold,
            diversity_threshold=config.diversity_threshold,
        )

        # Track evolution history
        self.history: List[GenerationSummary] = []
        self.best_correlation: float = 0.0
        self.plateau_count: int = 0

    def run_generation_zero(self, verbose: bool = True) -> tuple:
        """Run generation 0: initial feature generation and evaluation.

        Args:
            verbose: Whether to print progress.

        Returns:
            Tuple of (features, evaluations, summary).
        """
        if verbose:
            print("\n" + "=" * 60)
            print("GENERATION 0: Initial Feature Generation")
            print("=" * 60)

        # Generate initial features
        if verbose:
            print(f"\nGenerating {self.config.initial_features} initial features...")

        features = self.generator.generate_initial_features(
            n_features=self.config.initial_features,
            generation=0,
        )

        if verbose:
            print(f"Generated {len(features)} features")
            for f in features:
                print(f"  - {f.name}: {f.description[:60]}...")

        # Save features
        self.store.save_features(0, features)

        # Evaluate features
        if verbose:
            print(f"\nEvaluating on {self.config.tasks_per_eval} tasks...")

        tasks = self.data_loader.stratified_sample(
            self.config.tasks_per_eval,
            seed=0,
        )

        evaluations = self.evaluator.evaluate_features(features, tasks, verbose=verbose)

        # Save evaluations
        self.store.save_evaluations(0, evaluations)

        # Create summary
        best_eval = max(evaluations, key=lambda e: e.abs_correlation)
        summary = GenerationSummary(
            generation=0,
            n_features=len(features),
            n_surviving=len(features),
            best_correlation=best_eval.correlation,
            best_feature_id=best_eval.feature_id,
            mean_correlation=sum(e.correlation for e in evaluations) / len(evaluations),
            timestamp=datetime.now().isoformat(),
            token_usage=self.llm_client.usage.to_dict(),
        )

        self.store.save_summary(summary)
        self.history.append(summary)
        self.best_correlation = best_eval.abs_correlation

        if verbose:
            print(f"\nGeneration 0 Summary:")
            print(f"  Best correlation: {best_eval.correlation:+.3f} ({best_eval.feature_id})")
            print(f"  Mean correlation: {summary.mean_correlation:+.3f}")

        return features, evaluations, summary

    def run_generation(
        self,
        generation: int,
        prev_features: List[Feature],
        prev_evaluations: List[FeatureEvaluation],
        verbose: bool = True,
    ) -> tuple:
        """Run a single evolution generation.

        Args:
            generation: Generation number.
            prev_features: Features from previous generation.
            prev_evaluations: Evaluations from previous generation.
            verbose: Whether to print progress.

        Returns:
            Tuple of (features, evaluations, summary, should_stop).
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"GENERATION {generation}")
            print("=" * 60)

        # Select survivors
        if verbose:
            print("\nSelecting top features...")

        surviving_features, surviving_evals = self.selector.select(
            prev_features, prev_evaluations
        )

        if verbose:
            print(f"Selected {len(surviving_features)} survivors:")
            for f, e in zip(surviving_features, surviving_evals):
                print(f"  - {f.name}: r = {e.correlation:+.3f}")

        # Get failure cases for refinement
        tasks = self.data_loader.stratified_sample(
            self.config.tasks_per_eval,
            seed=generation,
        )
        task_map = {t.task_id: t for t in tasks}

        failure_cases_map: Dict[str, List[Dict]] = {}
        for feature, evaluation in zip(surviving_features, surviving_evals):
            failure_cases_map[feature.id] = self.evaluator.get_failure_cases(
                feature, evaluation, tasks, n_failures=5
            )

        # Evolve features
        if verbose:
            print("\nEvolving features...")

        evolved_features = self.refiner.evolve_population(
            surviving_features,
            surviving_evals,
            failure_cases_map,
            generation,
        )

        if verbose:
            print(f"Created {len(evolved_features)} evolved features")

        # Generate novel features (zero-order mutation)
        n_novel = max(1, self.config.initial_features // 4)
        if verbose:
            print(f"Generating {n_novel} novel features...")

        novel_features = self.generator.generate_with_context(
            n_features=n_novel,
            existing_features=surviving_features,
            generation=generation,
        )

        # Combine all features for this generation
        all_features = surviving_features + evolved_features + novel_features

        if verbose:
            print(f"\nTotal features for evaluation: {len(all_features)}")

        # Save features
        self.store.save_features(generation, all_features)

        # Evaluate all features on new random sample
        if verbose:
            print(f"Evaluating on {self.config.tasks_per_eval} tasks...")

        eval_tasks = self.data_loader.stratified_sample(
            self.config.tasks_per_eval,
            seed=generation * 1000,  # Different seed for evaluation
        )

        evaluations = self.evaluator.evaluate_features(
            all_features, eval_tasks, verbose=verbose
        )

        # Save evaluations
        self.store.save_evaluations(generation, evaluations)

        # Create summary
        best_eval = max(evaluations, key=lambda e: e.abs_correlation)
        summary = GenerationSummary(
            generation=generation,
            n_features=len(all_features),
            n_surviving=len(surviving_features),
            best_correlation=best_eval.correlation,
            best_feature_id=best_eval.feature_id,
            mean_correlation=sum(e.correlation for e in evaluations) / len(evaluations),
            timestamp=datetime.now().isoformat(),
            token_usage=self.llm_client.usage.to_dict(),
        )

        self.store.save_summary(summary)
        self.history.append(summary)

        # Check for improvement
        improvement = best_eval.abs_correlation - self.best_correlation
        if improvement > self.config.plateau_threshold:
            self.best_correlation = best_eval.abs_correlation
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        should_stop = self.plateau_count >= self.config.plateau_patience

        if verbose:
            print(f"\nGeneration {generation} Summary:")
            print(f"  Best correlation: {best_eval.correlation:+.3f} ({best_eval.feature_id})")
            print(f"  Mean correlation: {summary.mean_correlation:+.3f}")
            print(f"  Improvement: {improvement:+.4f}")
            print(f"  Plateau count: {self.plateau_count}/{self.config.plateau_patience}")
            if should_stop:
                print("  >>> Stopping: plateau reached")

        return all_features, evaluations, summary, should_stop

    def run(
        self,
        resume: bool = False,
        verbose: bool = True,
    ) -> tuple:
        """Run the full evolution loop.

        Args:
            resume: Whether to resume from checkpoint.
            verbose: Whether to print progress.

        Returns:
            Tuple of (best_features, best_evaluations).
        """
        start_generation = 0
        features = []
        evaluations = []

        # Check for resume
        if resume:
            checkpoint = self.store.load_checkpoint()
            if checkpoint:
                start_generation = checkpoint["generation"] + 1
                features = self.store.load_features(checkpoint["generation"])
                evaluations = self.store.load_evaluations(checkpoint["generation"])
                self.best_correlation = checkpoint["state"].get("best_correlation", 0.0)
                self.plateau_count = checkpoint["state"].get("plateau_count", 0)
                if verbose:
                    print(f"Resuming from generation {start_generation}")

        # Generation 0
        if start_generation == 0:
            features, evaluations, _ = self.run_generation_zero(verbose)
            start_generation = 1

            # Save checkpoint
            self.store.save_checkpoint(0, {
                "best_correlation": self.best_correlation,
                "plateau_count": self.plateau_count,
            })

        # Evolution loop
        for gen in range(start_generation, self.config.max_generations + 1):
            features, evaluations, summary, should_stop = self.run_generation(
                gen, features, evaluations, verbose
            )

            # Save checkpoint
            self.store.save_checkpoint(gen, {
                "best_correlation": self.best_correlation,
                "plateau_count": self.plateau_count,
            })

            if should_stop:
                break

        # Select final best features
        best_features, best_evals = self.selector.select(features, evaluations)

        # Save best features
        self.store.save_best_features(best_features, best_evals)

        # Save evolution log
        self.store.save_evolution_log({
            "config": {
                "model": self.config.model,
                "initial_features": self.config.initial_features,
                "top_k": self.config.top_k,
                "tasks_per_eval": self.config.tasks_per_eval,
                "max_generations": self.config.max_generations,
            },
            "generations": [s.to_dict() for s in self.history],
            "final_usage": self.llm_client.usage.to_dict(),
            "completed_at": datetime.now().isoformat(),
        })

        if verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETE")
            print("=" * 60)
            print(f"\nBest features (top {len(best_features)}):")
            for f, e in zip(best_features, best_evals):
                print(f"  - {f.name}: r = {e.correlation:+.3f}")
                print(f"    {f.description[:80]}...")

            print(f"\nTotal API usage:")
            print(f"  Calls: {self.llm_client.usage.total_calls}")
            print(f"  Input tokens: {self.llm_client.usage.total_input_tokens:,}")
            print(f"  Output tokens: {self.llm_client.usage.total_output_tokens:,}")
            print(f"  Estimated cost: ${self.llm_client.usage.estimated_cost:.2f}")

        return best_features, best_evals


def run_dry_run(config: EvolutionConfig):
    """Run a dry run to show execution plan without API calls.

    Args:
        config: Evolution configuration.
    """
    print("\n" + "=" * 60)
    print("DRY RUN - Execution Plan")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Model: {config.model}")
    print(f"  Initial features: {config.initial_features}")
    print(f"  Top K: {config.top_k}")
    print(f"  Tasks per eval: {config.tasks_per_eval}")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Output dir: {config.output_dir}")

    # Load data
    print("\nLoading data...")
    data_loader = DataLoader(config.items_path)
    df = data_loader.load()
    print(f"  Loaded {len(df)} tasks")
    print(f"  Difficulty range: [{df['b'].min():.2f}, {df['b'].max():.2f}]")

    # Estimate costs
    calls_gen0 = config.initial_features * config.tasks_per_eval + 1  # +1 for generation
    calls_per_gen = (
        config.top_k * config.tasks_per_eval  # Evaluate survivors
        + config.top_k  # Evolve survivors
        + config.initial_features // 4  # Novel features
        + (config.top_k + config.initial_features // 4) * config.tasks_per_eval  # Evaluate new
    )
    total_calls = calls_gen0 + calls_per_gen * config.max_generations

    tokens_per_call = 2000  # Rough estimate
    total_tokens = total_calls * tokens_per_call

    # Cost estimate based on model
    if "opus" in config.model.lower():
        cost_per_1m = 15 + 75  # Input + output (rough)
    else:
        cost_per_1m = 3 + 15  # Sonnet pricing

    estimated_cost = total_tokens * cost_per_1m / 1_000_000

    print(f"\nEstimated API usage:")
    print(f"  Generation 0: ~{calls_gen0} calls")
    print(f"  Per generation: ~{calls_per_gen} calls")
    print(f"  Total calls: ~{total_calls}")
    print(f"  Total tokens: ~{total_tokens:,}")
    print(f"  Estimated cost: ${estimated_cost:.2f}")

    print(f"\nExecution plan:")
    print(f"  1. Generate {config.initial_features} initial features from difficulty extremes")
    print(f"  2. Evaluate on {config.tasks_per_eval} stratified tasks")
    print(f"  3. For each generation (up to {config.max_generations}):")
    print(f"     - Select top {config.top_k} diverse features")
    print(f"     - Evolve using 5 mutation operators")
    print(f"     - Generate {config.initial_features // 4} novel features")
    print(f"     - Evaluate all on new random sample")
    print(f"  4. Stop if no improvement for {config.plateau_patience} generations")
    print(f"  5. Save best features to {config.output_dir / 'best_features.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Evolutionary feature discovery for IRT difficulty prediction'
    )

    # Model settings
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                        help='Model to use (default: claude-sonnet-4-20250514)')
    parser.add_argument('--api_delay', type=float, default=0.5,
                        help='Delay between API calls in seconds')

    # Evolution parameters
    parser.add_argument('--initial_features', type=int, default=10,
                        help='Number of initial features to generate')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top features to keep each generation')
    parser.add_argument('--tasks_per_eval', type=int, default=50,
                        help='Number of tasks per evaluation')
    parser.add_argument('--max_generations', type=int, default=5,
                        help='Maximum number of generations')

    # Paths
    parser.add_argument('--output_dir', type=str,
                        default='llm_judge/evolutionary_results',
                        help='Output directory for results')
    parser.add_argument('--items_path', type=str,
                        default='clean_data/swebench_verified_20250930_full/1d/items.csv',
                        help='Path to IRT items CSV')

    # Execution modes
    parser.add_argument('--dry_run', action='store_true',
                        help='Show execution plan without running')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Build config
    config = EvolutionConfig(
        model=args.model,
        api_delay=args.api_delay,
        initial_features=args.initial_features,
        top_k=args.top_k,
        tasks_per_eval=args.tasks_per_eval,
        max_generations=args.max_generations,
        output_dir=Path(args.output_dir),
        items_path=Path(args.items_path),
    )

    if args.dry_run:
        run_dry_run(config)
        return

    # Run evolution
    runner = EvolutionRunner(config)
    best_features, best_evals = runner.run(
        resume=args.resume,
        verbose=not args.quiet,
    )

    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
