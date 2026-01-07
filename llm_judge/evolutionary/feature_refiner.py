"""Feature evolution via multiple mutation operators (PromptBreeder-inspired)."""

import random
from typing import Dict, List, Optional, Tuple

from .config import EvolutionConfig
from .data_loader import Task
from .feature_store import Feature, FeatureEvaluation
from .llm_client import LLMClient


# Default mutation prompts (can be evolved)
DEFAULT_MUTATION_PROMPTS = {
    "direct_mutation": """Improve this feature based on failure cases. The feature should better distinguish easy from hard tasks.

Current feature: {feature_description}
Current extraction prompt: {extraction_prompt}
Current correlation: {correlation}

Failure cases (high error):
{failure_cases}

Generate an improved version that addresses these failures while maintaining what works.""",

    "eda_mutation": """Create a hybrid feature by combining the best aspects of these two features:

Feature A (r={corr_a}): {desc_a}
Feature B (r={corr_b}): {desc_b}

Create a new feature that captures complementary aspects of both.""",

    "hypermutation": """You are improving the mutation process itself.

Current mutation prompt:
{mutation_prompt}

This mutation prompt achieved correlation: {correlation}
The best features so far have correlations around: {best_correlation}

Suggest an improved mutation prompt that would generate better features.""",
}


class FeatureRefiner:
    """Evolve features using multiple mutation operators."""

    def __init__(
        self,
        config: EvolutionConfig,
        llm_client: LLMClient,
    ):
        """Initialize feature refiner.

        Args:
            config: Evolution configuration.
            llm_client: LLM client for API calls.
        """
        self.config = config
        self.llm_client = llm_client
        self.mutation_prompts = DEFAULT_MUTATION_PROMPTS.copy()

    def select_mutation_operator(self) -> str:
        """Select a mutation operator based on weights.

        Returns:
            Name of the selected mutation operator.
        """
        operators = list(self.config.mutation_weights.keys())
        weights = [self.config.mutation_weights[op] for op in operators]
        return random.choices(operators, weights=weights)[0]

    def direct_mutation(
        self,
        feature: Feature,
        evaluation: FeatureEvaluation,
        failure_cases: List[Dict],
        generation: int,
    ) -> Feature:
        """Directly mutate a feature based on failure analysis.

        Args:
            feature: Feature to mutate.
            evaluation: Current evaluation results.
            failure_cases: List of failure case dicts.
            generation: Current generation number.

        Returns:
            Mutated Feature.
        """
        failure_text = "\n".join(
            f"- {fc['task_id']}: predicted={fc['predicted_score']:.1f}, "
            f"actual_difficulty={fc['actual_difficulty']:.2f}, "
            f"error={fc['error']:.2f}"
            for fc in failure_cases[:5]
        )

        prompt = self.mutation_prompts["direct_mutation"].format(
            feature_description=feature.description,
            extraction_prompt=feature.extraction_prompt,
            correlation=f"{evaluation.correlation:+.3f}",
            failure_cases=failure_text,
        )

        prompt += """

Respond with JSON:
{
    "name": "<improved_name>",
    "description": "<improved description>",
    "scale_low": "<what 1 means>",
    "scale_high": "<what 5 means>",
    "hypothesis": "<why this should work better>",
    "extraction_prompt": "<improved prompt>"
}"""

        response = self.llm_client.call_json(prompt, temperature=0.7)

        return Feature(
            id=f"gen{generation}_dm_{feature.name[:10]}",
            name=response["name"],
            description=response["description"],
            extraction_prompt=response["extraction_prompt"],
            scale_low=response["scale_low"],
            scale_high=response["scale_high"],
            hypothesis=response["hypothesis"],
            parent_id=feature.id,
            mutation_type="direct_mutation",
            generation=generation,
        )

    def eda_mutation(
        self,
        feature_a: Feature,
        eval_a: FeatureEvaluation,
        feature_b: Feature,
        eval_b: FeatureEvaluation,
        generation: int,
    ) -> Feature:
        """Create hybrid feature from two parents (crossover).

        Args:
            feature_a: First parent feature.
            eval_a: Evaluation of first parent.
            feature_b: Second parent feature.
            eval_b: Evaluation of second parent.
            generation: Current generation number.

        Returns:
            Hybrid Feature.
        """
        prompt = self.mutation_prompts["eda_mutation"].format(
            corr_a=f"{eval_a.correlation:+.3f}",
            desc_a=f"{feature_a.name}: {feature_a.description}",
            corr_b=f"{eval_b.correlation:+.3f}",
            desc_b=f"{feature_b.name}: {feature_b.description}",
        )

        prompt += """

Create a new hybrid feature. Respond with JSON:
{
    "name": "<hybrid_name>",
    "description": "<combined description>",
    "scale_low": "<what 1 means>",
    "scale_high": "<what 5 means>",
    "hypothesis": "<why combining these aspects should work>",
    "extraction_prompt": "<prompt for hybrid feature>"
}"""

        response = self.llm_client.call_json(prompt, temperature=0.8)

        return Feature(
            id=f"gen{generation}_eda_{feature_a.name[:5]}_{feature_b.name[:5]}",
            name=response["name"],
            description=response["description"],
            extraction_prompt=response["extraction_prompt"],
            scale_low=response["scale_low"],
            scale_high=response["scale_high"],
            hypothesis=response["hypothesis"],
            parent_id=f"{feature_a.id}+{feature_b.id}",
            mutation_type="eda_mutation",
            generation=generation,
        )

    def hypermutation(
        self,
        feature: Feature,
        evaluation: FeatureEvaluation,
        best_correlation: float,
        generation: int,
    ) -> Tuple[Feature, str]:
        """Mutate the mutation prompt itself, then apply it.

        Args:
            feature: Feature to evolve.
            evaluation: Current evaluation results.
            best_correlation: Best correlation achieved so far.
            generation: Current generation number.

        Returns:
            Tuple of (new Feature, new mutation prompt).
        """
        # First, evolve the mutation prompt
        meta_prompt = self.mutation_prompts["hypermutation"].format(
            mutation_prompt=self.mutation_prompts["direct_mutation"][:500],
            correlation=f"{evaluation.correlation:+.3f}",
            best_correlation=f"{best_correlation:+.3f}",
        )

        meta_prompt += """

Respond with JSON:
{
    "improved_mutation_prompt": "<new mutation prompt>"
}"""

        meta_response = self.llm_client.call_json(meta_prompt, temperature=0.9)
        new_mutation_prompt = meta_response.get(
            "improved_mutation_prompt",
            self.mutation_prompts["direct_mutation"],
        )

        # Now apply the new mutation prompt
        apply_prompt = f"""{new_mutation_prompt}

Current feature:
- Name: {feature.name}
- Description: {feature.description}
- Correlation: {evaluation.correlation:+.3f}

Respond with JSON:
{{
    "name": "<mutated_name>",
    "description": "<mutated description>",
    "scale_low": "<what 1 means>",
    "scale_high": "<what 5 means>",
    "hypothesis": "<hypothesis>",
    "extraction_prompt": "<extraction prompt>"
}}"""

        response = self.llm_client.call_json(apply_prompt, temperature=0.7)

        new_feature = Feature(
            id=f"gen{generation}_hyper_{feature.name[:10]}",
            name=response["name"],
            description=response["description"],
            extraction_prompt=response["extraction_prompt"],
            scale_low=response["scale_low"],
            scale_high=response["scale_high"],
            hypothesis=response["hypothesis"],
            parent_id=feature.id,
            mutation_type="hypermutation",
            mutation_prompt=new_mutation_prompt[:500],
            generation=generation,
        )

        return new_feature, new_mutation_prompt

    def evolve_feature(
        self,
        feature: Feature,
        evaluation: FeatureEvaluation,
        failure_cases: List[Dict],
        all_features: List[Feature],
        all_evaluations: List[FeatureEvaluation],
        generation: int,
    ) -> Feature:
        """Evolve a single feature using a randomly selected operator.

        Args:
            feature: Feature to evolve.
            evaluation: Evaluation of this feature.
            failure_cases: Failure cases for this feature.
            all_features: All surviving features.
            all_evaluations: All evaluations.
            generation: Current generation number.

        Returns:
            Evolved Feature.
        """
        operator = self.select_mutation_operator()

        if operator == "direct_mutation":
            return self.direct_mutation(feature, evaluation, failure_cases, generation)

        elif operator == "eda_mutation":
            # Select another feature for crossover
            other_features = [f for f in all_features if f.id != feature.id]
            if other_features:
                other = random.choice(other_features)
                other_eval = next(
                    (e for e in all_evaluations if e.feature_id == other.id),
                    evaluation,
                )
                return self.eda_mutation(
                    feature, evaluation, other, other_eval, generation
                )
            else:
                return self.direct_mutation(
                    feature, evaluation, failure_cases, generation
                )

        elif operator == "hypermutation":
            best_corr = max(e.abs_correlation for e in all_evaluations)
            new_feature, new_prompt = self.hypermutation(
                feature, evaluation, best_corr, generation
            )
            # Optionally update mutation prompts
            if random.random() < 0.3:  # 30% chance to keep evolved prompt
                self.mutation_prompts["direct_mutation"] = new_prompt
            return new_feature

        elif operator == "zero_order":
            # Handled separately in evolution loop
            return self.direct_mutation(feature, evaluation, failure_cases, generation)

        else:
            return self.direct_mutation(feature, evaluation, failure_cases, generation)

    def evolve_population(
        self,
        features: List[Feature],
        evaluations: List[FeatureEvaluation],
        failure_cases_map: Dict[str, List[Dict]],
        generation: int,
    ) -> List[Feature]:
        """Evolve a population of features.

        Args:
            features: List of surviving features.
            evaluations: Evaluations of surviving features.
            failure_cases_map: Dict mapping feature_id to failure cases.
            generation: Current generation number.

        Returns:
            List of evolved Feature objects.
        """
        evolved = []

        for feature in features:
            evaluation = next(
                (e for e in evaluations if e.feature_id == feature.id),
                None,
            )
            if evaluation is None:
                continue

            failure_cases = failure_cases_map.get(feature.id, [])

            try:
                new_feature = self.evolve_feature(
                    feature,
                    evaluation,
                    failure_cases,
                    features,
                    evaluations,
                    generation,
                )
                evolved.append(new_feature)
            except Exception as e:
                print(f"Warning: Failed to evolve {feature.name}: {e}")

        return evolved
