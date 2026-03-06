"""V3 prompt configuration - adds apparent_success feature.

This version adds a feature to capture when agents appear to succeed
(smooth trajectory, verification, confidence) but actually fail,
which may indicate hidden task complexity.
"""

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition
from experiment_b.trajectory_features.simple_extractor import TrajectoryPromptConfig as PromptConfig

FRONTIER_V3_FEATURES = [
    FeatureDefinition(
        name="solution_stability",
        min_value=1,
        max_value=5,
        description="Once the agent finds an approach, does it hold or keep breaking?",
    ),
    FeatureDefinition(
        name="conceptual_pivot_count",
        min_value=0,
        max_value=5,
        description="How many times does the agent fundamentally change their understanding of the problem?",
    ),
    FeatureDefinition(
        name="progress_linearity",
        min_value=1,
        max_value=5,
        description="Does the agent make steady linear progress or have many setbacks?",
    ),
    FeatureDefinition(
        name="apparent_success",
        min_value=1,
        max_value=5,
        description="How successful does the trajectory APPEAR to be (ignoring actual failure)?",
    ),
]

FRONTIER_V3_PROMPT = '''You are analyzing an AI coding agent's full trajectory on a software engineering task.

The agent attempted to solve a GitHub issue but FAILED. Analyze the ENTIRE trajectory carefully, paying attention to:
- How the agent's understanding evolves over time
- Whether fixes work or cause new problems
- How often the agent has to backtrack or revise their approach
- How confident/successful the agent APPEARS to be

## TRAJECTORY
Agent: {agent}
Task: {task_id}
Outcome: FAILED (agent did not resolve the task)

### Agent's Full Attempt:
{trajectory_content}

## FEATURES TO EVALUATE

Analyze the trajectory from start to finish and rate each feature.

1. **solution_stability** (1-5): Once found, does the approach hold?
   - 1: Approach keeps breaking, constant instability
   - 2: Approach works sometimes but frequently fails
   - 3: Approach mostly works but has edge case issues
   - 4: Approach is mostly stable with minor adjustments
   - 5: First working approach is stable throughout

2. **conceptual_pivot_count** (0-5): How many times does the agent fundamentally re-understand the problem?
   - 0: Agent's initial understanding is correct and doesn't change
   - 1-2: Minor refinements to understanding
   - 3-4: Significant pivots in how agent conceptualizes the problem
   - 5: Agent completely misunderstands initially, has to start over

3. **progress_linearity** (1-5): How linear is the agent's progress?
   - 1: Constant setbacks, no steady progress, lots of backtracking
   - 2: More setbacks than progress
   - 3: Mixed - some progress, some setbacks
   - 4: Mostly linear progress with minor setbacks
   - 5: Steady linear progress from start to finish

4. **apparent_success** (1-5): How successful does the trajectory APPEAR to be?
   This is about appearances, NOT actual success (all tasks failed).
   - 1: Agent clearly struggled (stuck in loops, obvious errors, confusion)
   - 2: Agent had significant problems, many failed attempts visible
   - 3: Agent made decent progress but with some clear issues
   - 4: Agent appeared to implement a working solution with minor issues
   - 5: Agent appeared to fully succeed (smooth implementation, passed their own tests, expressed confidence in completion)

   KEY: High apparent_success despite actual failure may indicate HIDDEN complexity in the task.

## RESPONSE FORMAT

Respond with ONLY a JSON object in this exact format:
{{
    "solution_stability": <1-5>,
    "conceptual_pivot_count": <0-5>,
    "progress_linearity": <1-5>,
    "apparent_success": <1-5>,
    "reasoning": "<2-3 sentences explaining your key observations about the trajectory>"
}}

IMPORTANT: All 4 features are REQUIRED. Base ratings on the ENTIRE trajectory, not just beginning or end.
'''


def get_frontier_v3_config() -> PromptConfig:
    """Get the V3 prompt configuration for frontier trajectory features."""
    return PromptConfig(
        name="frontier_trajectory_v3",
        features=FRONTIER_V3_FEATURES,
        prompt_template=FRONTIER_V3_PROMPT,
        task_id_field="task_id",
        truncation_limits={
            "trajectory_content": 80000,
        },
    )
