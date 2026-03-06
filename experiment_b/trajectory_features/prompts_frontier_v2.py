"""V2 prompt configuration for frontier trajectory feature extraction.

Reduced to 3 features based on initial correlation analysis:
- solution_stability
- progress_linearity
- conceptual_pivot_count
"""

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition
from experiment_b.trajectory_features.simple_extractor import TrajectoryPromptConfig as PromptConfig

FRONTIER_V2_FEATURES = [
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
]

FRONTIER_V2_PROMPT = '''You are analyzing an AI coding agent's full trajectory on a software engineering task.

The agent attempted to solve a GitHub issue but FAILED. Analyze the ENTIRE trajectory carefully, paying attention to:
- How the agent's understanding evolves over time
- Whether fixes work or cause new problems
- How often the agent has to backtrack or revise their approach

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

## RESPONSE FORMAT

Respond with ONLY a JSON object in this exact format:
{{
    "solution_stability": <1-5>,
    "conceptual_pivot_count": <0-5>,
    "progress_linearity": <1-5>,
    "reasoning": "<2-3 sentences explaining your key observations about the trajectory>"
}}

IMPORTANT: All 3 features are REQUIRED. Base ratings on the ENTIRE trajectory, not just beginning or end.
'''


def get_frontier_v2_config() -> PromptConfig:
    """Get the V2 prompt configuration for frontier trajectory features."""
    return PromptConfig(
        name="frontier_trajectory_v2",
        features=FRONTIER_V2_FEATURES,
        prompt_template=FRONTIER_V2_PROMPT,
        task_id_field="task_id",
        truncation_limits={
            "trajectory_content": 80000,
        },
    )
