"""V1 prompt configuration for frontier trajectory feature extraction.

This prompt extracts features from agent trajectories that may be
informative about the underlying task's difficulty.

These features focus on observable trajectory patterns based on analysis
of easy vs hard frontier tasks.
"""

from experiment_ab_shared.llm_judge.prompt_config import FeatureDefinition
from experiment_b.trajectory_features.simple_extractor import TrajectoryPromptConfig as PromptConfig

# V1 Features: Trajectory-based features derived from qualitative analysis
# Key insight: Hard tasks show more "fix-break-revise" cycles and test regressions
FRONTIER_V1_FEATURES = [
    FeatureDefinition(
        name="fix_revision_count",
        min_value=0,
        max_value=10,
        description="How many times does the agent substantially revise their fix approach?",
    ),
    FeatureDefinition(
        name="test_regression_occurrence",
        min_value=1,
        max_value=5,
        description="Do fixes cause other tests to fail, requiring the agent to backtrack?",
    ),
    FeatureDefinition(
        name="error_guidance_quality",
        min_value=1,
        max_value=5,
        description="Do error messages and test failures guide the agent toward the solution?",
    ),
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

FRONTIER_V1_PROMPT = '''You are analyzing an AI coding agent's full trajectory on a software engineering task.

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

1. **fix_revision_count** (0-10): How many times does the agent substantially revise their fix?
   - Count major approach changes, not minor tweaks
   - 0: Agent finds one approach and sticks with it
   - 3-4: Several revisions needed
   - 7+: Constant revision, agent keeps changing strategy

2. **test_regression_occurrence** (1-5): Do fixes cause OTHER tests to fail?
   - 1: No test regressions - fixes don't break existing functionality
   - 2: Minor regressions that are easily fixed
   - 3: Some regressions requiring adjustment
   - 4: Multiple regressions, agent struggles to fix without breaking things
   - 5: Severe regressions - every fix breaks something else

3. **error_guidance_quality** (1-5): Do errors guide the agent toward the solution?
   - 1: Errors are misleading or send agent in wrong direction
   - 2: Errors don't provide useful guidance
   - 3: Errors give partial guidance
   - 4: Errors clearly indicate what's wrong
   - 5: Errors directly point to the fix location and approach

4. **solution_stability** (1-5): Once found, does the approach hold?
   - 1: Approach keeps breaking, constant instability
   - 2: Approach works sometimes but frequently fails
   - 3: Approach mostly works but has edge case issues
   - 4: Approach is mostly stable with minor adjustments
   - 5: First working approach is stable throughout

5. **conceptual_pivot_count** (0-5): How many times does the agent fundamentally re-understand the problem?
   - 0: Agent's initial understanding is correct and doesn't change
   - 1-2: Minor refinements to understanding
   - 3-4: Significant pivots in how agent conceptualizes the problem
   - 5: Agent completely misunderstands initially, has to start over

6. **progress_linearity** (1-5): How linear is the agent's progress?
   - 1: Constant setbacks, no steady progress, lots of backtracking
   - 2: More setbacks than progress
   - 3: Mixed - some progress, some setbacks
   - 4: Mostly linear progress with minor setbacks
   - 5: Steady linear progress from start to finish

## RESPONSE FORMAT

Respond with ONLY a JSON object in this exact format:
{{
    "fix_revision_count": <0-10>,
    "test_regression_occurrence": <1-5>,
    "error_guidance_quality": <1-5>,
    "solution_stability": <1-5>,
    "conceptual_pivot_count": <0-5>,
    "progress_linearity": <1-5>,
    "reasoning": "<2-3 sentences explaining your key observations about the trajectory>"
}}

IMPORTANT: All 6 features are REQUIRED. Base ratings on the ENTIRE trajectory, not just beginning or end.
'''


def get_frontier_v1_config() -> PromptConfig:
    """Get the V1 prompt configuration for frontier trajectory features."""
    return PromptConfig(
        name="frontier_trajectory_v1",
        features=FRONTIER_V1_FEATURES,
        prompt_template=FRONTIER_V1_PROMPT,
        task_id_field="task_id",
        truncation_limits={
            "trajectory_content": 80000,  # ~20k tokens, need more for full trajectory
        },
    )
