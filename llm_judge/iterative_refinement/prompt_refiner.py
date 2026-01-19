"""LLM-based prompt refinement using residual analysis.

Uses an LLM to propose improved feature definitions based on:
1. Current feature correlations and entropies
2. High-residual tasks where predictions fail
3. Feature coefficients showing which features matter most
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from llm_judge.iterative_refinement.prompt_store import (
    FeatureDefinition,
    generate_prompt_from_schema,
)
from llm_judge.iterative_refinement.residual_analyzer import (
    ResidualAnalysis,
    format_residual_analysis_for_llm,
)


@dataclass
class RefinementProposal:
    """A proposed refinement to the feature schema."""

    # New feature definitions
    new_features: List[FeatureDefinition]

    # Changes summary
    features_added: List[str]
    features_removed: List[str]
    features_modified: List[str]

    # Reasoning from LLM
    reasoning: str

    # Archived features (removed from previous version)
    archived_features: List[FeatureDefinition]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "features_added": self.features_added,
            "features_removed": self.features_removed,
            "features_modified": self.features_modified,
            "reasoning": self.reasoning,
            "n_new_features": len(self.new_features),
        }


REFINEMENT_SYSTEM_PROMPT = """You are an expert at designing feature extraction prompts for difficulty prediction.

Your task is to refine a set of features used by an LLM judge to predict how difficult programming tasks are for AI coding agents.

The current features are used to predict IRT difficulty scores (β). Higher β = harder task.

Based on the prediction failures shown, propose improvements to the feature schema:
1. MODIFY features that have poor scale anchors or unclear definitions
2. REMOVE features that are redundant (highly correlated) or uninformative (low entropy)
3. ADD new features that would distinguish the failure cases

Guidelines:
- Keep total features between 6-12 (balance between coverage and noise)
- Each feature should have clear, distinct scale anchors
- Features should be observable from the problem statement and gold patch
- Consider what makes tasks "deceptively" hard or easy

Output a JSON object with:
1. "features": array of feature definitions
2. "changes_summary": brief description of what changed and why
3. "reasoning": explanation of why these changes should improve predictions"""


def build_refinement_prompt(
    current_features: List[FeatureDefinition],
    analysis: ResidualAnalysis,
    quick_eval_metrics: Dict[str, Any],
) -> str:
    """Build the prompt for the LLM refiner.

    Args:
        current_features: Current feature definitions
        analysis: Residual analysis with high-residual tasks
        quick_eval_metrics: Metrics from quick evaluation (entropy, correlations)

    Returns:
        Prompt string for refinement
    """
    lines = [
        "# Current Feature Schema",
        "",
        f"The current schema has {len(current_features)} features.",
        f"Correlation with ground truth difficulty: r = {quick_eval_metrics.get('pearson_r', 'N/A')}",
        "",
        "## Feature Definitions",
        "",
    ]

    for f in current_features:
        lines.append(f"### {f.name} ({f.min_value}-{f.max_value})")
        lines.append(f"*{f.description}*")
        lines.append(f"- Low ({f.min_value}): {f.scale_low}")
        lines.append(f"- High ({f.max_value}): {f.scale_high}")
        lines.append("")

    # Add entropy information
    entropies = quick_eval_metrics.get("feature_entropies", {})
    if entropies:
        lines.extend(
            [
                "## Feature Quality Metrics",
                "",
                "### Entropy (higher = more informative, uses full scale)",
                "",
            ]
        )
        for name, entropy in sorted(entropies.items(), key=lambda x: x[1]):
            flag = " ⚠️ LOW" if entropy < 1.0 else ""
            lines.append(f"- {name}: {entropy:.2f}{flag}")

    # Add redundancy information
    redundant = quick_eval_metrics.get("redundant_pairs", [])
    if redundant:
        lines.extend(
            [
                "",
                "### Redundant Feature Pairs (r > 0.9)",
                "",
            ]
        )
        for pair in redundant:
            if isinstance(pair, dict):
                lines.append(f"- {pair['f1']} ↔ {pair['f2']}: r = {pair['r']:.2f}")
            else:
                lines.append(f"- {pair[0]} ↔ {pair[1]}: r = {pair[2]:.2f}")

    # Add residual analysis
    lines.append("")
    lines.append(format_residual_analysis_for_llm(analysis, current_features))

    lines.extend(
        [
            "",
            "# Your Task",
            "",
            "Propose an improved feature schema that would better predict these failure cases.",
            "",
            "Output JSON in this format:",
            "```json",
            "{",
            '  "features": [',
            "    {",
            '      "name": "feature_name",',
            '      "description": "What this feature measures",',
            '      "scale_low": "Description of lowest value",',
            '      "scale_high": "Description of highest value",',
            '      "min_value": 0,',
            '      "max_value": 5,',
            '      "extraction_prompt": "Full prompt text for extracting this feature"',
            "    },",
            "    ...",
            "  ],",
            '  "changes_summary": "Brief description of changes",',
            '  "reasoning": "Why these changes should improve predictions"',
            "}",
            "```",
        ]
    )

    return "\n".join(lines)


def propose_refinement(
    current_features: List[FeatureDefinition],
    analysis: ResidualAnalysis,
    quick_eval_metrics: Dict[str, Any],
    model: str = "gpt-5.2",
) -> RefinementProposal:
    """Use LLM to propose refined feature definitions.

    Args:
        current_features: Current feature definitions
        analysis: Residual analysis with failure cases
        quick_eval_metrics: Metrics from quick evaluation
        model: Model to use for refinement

    Returns:
        RefinementProposal with new feature schema
    """
    client = OpenAI()

    prompt = build_refinement_prompt(current_features, analysis, quick_eval_metrics)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=4000,
    )

    # Parse response
    text = response.output_text.strip()

    # Extract JSON from response
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        json_str = text

    data = json.loads(json_str)

    # Parse features
    new_features = []
    for f in data.get("features", []):
        new_features.append(
            FeatureDefinition(
                name=f["name"],
                description=f["description"],
                scale_low=f["scale_low"],
                scale_high=f["scale_high"],
                min_value=f["min_value"],
                max_value=f["max_value"],
                extraction_prompt=f["extraction_prompt"],
            )
        )

    # Determine changes
    current_names = {f.name for f in current_features}
    new_names = {f.name for f in new_features}

    features_added = list(new_names - current_names)
    features_removed = list(current_names - new_names)
    features_modified = [
        name
        for name in current_names & new_names
        if _feature_modified(
            next(f for f in current_features if f.name == name),
            next(f for f in new_features if f.name == name),
        )
    ]

    # Archive removed features
    archived = [f for f in current_features if f.name in features_removed]

    return RefinementProposal(
        new_features=new_features,
        features_added=features_added,
        features_removed=features_removed,
        features_modified=features_modified,
        reasoning=data.get("reasoning", "") + "\n\n" + data.get("changes_summary", ""),
        archived_features=archived,
    )


def _feature_modified(old: FeatureDefinition, new: FeatureDefinition) -> bool:
    """Check if a feature was meaningfully modified."""
    return (
        old.description != new.description
        or old.scale_low != new.scale_low
        or old.scale_high != new.scale_high
        or old.min_value != new.min_value
        or old.max_value != new.max_value
    )


def apply_refinement_constraints(
    proposal: RefinementProposal,
    min_features: int = 6,
    max_features: int = 12,
) -> RefinementProposal:
    """Apply constraints to ensure valid feature schema.

    Args:
        proposal: The raw proposal from LLM
        min_features: Minimum allowed features
        max_features: Maximum allowed features

    Returns:
        Constrained proposal
    """
    features = proposal.new_features

    # Ensure we have at least min_features
    if len(features) < min_features:
        print(f"Warning: Only {len(features)} features proposed, keeping all")

    # Trim to max_features if needed
    if len(features) > max_features:
        print(f"Warning: {len(features)} features proposed, trimming to {max_features}")
        features = features[:max_features]

    return RefinementProposal(
        new_features=features,
        features_added=proposal.features_added,
        features_removed=proposal.features_removed,
        features_modified=proposal.features_modified,
        reasoning=proposal.reasoning,
        archived_features=proposal.archived_features,
    )