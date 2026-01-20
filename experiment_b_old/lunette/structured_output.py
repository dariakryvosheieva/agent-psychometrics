"""Lunette structured output support for Experiment B trajectory grading.

This module provides:
1. TrajectoryFeatures - Pydantic model for structured output from trajectory grading
2. TrajectoryGradingPlan - Custom AnalysisPlan that includes the schema in serialization

Usage:
    from experiment_b.lunette.structured_output import TrajectoryGradingPlan, TrajectoryFeatures

    plan = TrajectoryGradingPlan(
        name="difficulty-prediction",
        prompt=TRAJECTORY_GRADING_PROMPT,
    )

    results = await client.investigate(run_id=run_id, plan=plan)

    # Results will be structured as TrajectoryFeatures
    for result in results.results:
        features = result.data  # Dict matching TrajectoryFeatures schema
"""

from typing import ClassVar, Literal, List, Optional

from pydantic import BaseModel, Field

from lunette.analysis import AnalysisPlanBase


class TrajectoryFeatures(BaseModel):
    """Pydantic model for structured output from Lunette trajectory grading.

    This model captures:
    - Primary difficulty score (0-1)
    - Agentic competencies (1-4 scale, from AgentDiagnose paper)
    - Failure mode indicators (binary, from SWE-bench Failures paper)
    - Trajectory signals (binary)
    """

    # ===== Primary Output =====
    difficulty_score: float = Field(
        ge=0.0, le=1.0,
        description="Predicted task difficulty (0.0=very easy, 1.0=very hard)"
    )

    # ===== Agentic Competencies (1-4 scale) =====
    backtracking_exploration: int = Field(
        ge=1, le=4,
        description="How well the agent explored alternatives when stuck (1=poor, 4=excellent)"
    )
    task_decomposition: int = Field(
        ge=1, le=4,
        description="How well the agent broke down the task (1=poor, 4=excellent)"
    )
    observation_reading: int = Field(
        ge=1, le=4,
        description="How well the agent read and understood observations (1=poor, 4=excellent)"
    )
    self_verification: int = Field(
        ge=1, le=4,
        description="How well the agent verified its own work (1=poor, 4=excellent)"
    )

    # ===== Failure Mode Indicators (binary) =====
    localization_failure: Literal[0, 1] = Field(
        description="Did the agent fail to find the right code location? (0=no, 1=yes)"
    )
    strategy_defect: Literal[0, 1] = Field(
        description="Did the agent use a flawed approach? (0=no, 1=yes)"
    )
    implementation_defect: Literal[0, 1] = Field(
        description="Was the implementation buggy/incorrect? (0=no, 1=yes)"
    )
    incomplete_repair: Literal[0, 1] = Field(
        description="Was the fix partial/incomplete? (0=no, 1=yes)"
    )
    verification_failure: Literal[0, 1] = Field(
        description="Did the agent fail to properly verify its fix? (0=no, 1=yes)"
    )

    # ===== Trajectory Signals (binary) =====
    agent_looping: Literal[0, 1] = Field(
        description="Did the agent get stuck in repetitive loops? (0=no, 1=yes)"
    )
    agent_gave_up_early: Literal[0, 1] = Field(
        description="Did the agent stop trying prematurely? (0=no, 1=yes)"
    )
    agent_wrong_focus: Literal[0, 1] = Field(
        description="Did the agent fixate on irrelevant code? (0=no, 1=yes)"
    )
    context_overflow: Literal[0, 1] = Field(
        description="Did the agent lose track of earlier context? (0=no, 1=yes)"
    )

    # ===== Explanation =====
    reasoning: str = Field(
        description="1-2 sentence explanation of why this difficulty score was assigned"
    )


class TrajectoryGradingPlan(AnalysisPlanBase):
    """Custom AnalysisPlan for grading trajectories with structured output.

    This plan overrides model_dump() to include the result_schema JSON schema
    in the serialized output, which gets sent to the Lunette API.

    The Lunette backend uses the schema to force the judge model to output
    structured data matching the TrajectoryFeatures Pydantic model via tool use.
    """

    kind: Literal["grading"] = "grading"
    result_schema: ClassVar[type[TrajectoryFeatures]] = TrajectoryFeatures

    def model_dump(self, **kwargs):
        """Override to include result_schema in serialized output."""
        d = super().model_dump(**kwargs)
        if self.result_schema is not None:
            d["result_schema"] = self.result_schema.model_json_schema()
        return d


# Grading prompt for Lunette trajectory analysis
TRAJECTORY_GRADING_PROMPT = """You are analyzing a SWE-bench agent trajectory to PREDICT TASK DIFFICULTY.

The trajectory shows an agent attempting to solve a software engineering task. Based on how the agent struggled (or didn't), estimate how difficult this task is.

## YOUR TASK: Analyze the trajectory and predict difficulty

Output your analysis as structured data with these fields:

### difficulty_score (0.0 to 1.0)
- 0.0 = Very easy task (agent solved it quickly with no issues)
- 0.3 = Easy task (minor struggles but clear path to solution)
- 0.5 = Medium difficulty (some exploration needed, partial failures)
- 0.7 = Hard task (significant struggles, multiple failure modes)
- 1.0 = Very hard task (agent failed completely despite substantial effort)

### Difficulty Indicators to Consider

**Harder task indicators:**
- Agent tried multiple approaches that failed
- Agent got stuck in loops or repeated similar mistakes
- Agent mislocated the relevant code
- Agent's fix was superficial or incomplete
- Agent lost track of context
- Many back-and-forth iterations without progress

**Easier task indicators:**
- Agent found the right location quickly
- Agent's first approach worked
- Clear error messages guided the agent
- Simple, localized fix was sufficient
- Agent verified solution correctly

### Agentic Competencies (1-4 scale each)
Rate how well the agent demonstrated each competency:
- **backtracking_exploration**: Did the agent explore alternatives when stuck?
- **task_decomposition**: Did the agent break down the problem effectively?
- **observation_reading**: Did the agent read and understand output correctly?
- **self_verification**: Did the agent verify its own work?

### Failure Modes (0 or 1 each)
Indicate if each failure mode was present:
- **localization_failure**: Agent failed to find the right code location
- **strategy_defect**: Agent used fundamentally wrong approach
- **implementation_defect**: Agent's implementation was buggy
- **incomplete_repair**: Agent's fix was partial or incomplete
- **verification_failure**: Agent didn't properly verify its fix

### Trajectory Signals (0 or 1 each)
Note any observed signals:
- **agent_looping**: Agent got stuck in repetitive loops
- **agent_gave_up_early**: Agent stopped trying prematurely
- **agent_wrong_focus**: Agent fixated on irrelevant code
- **context_overflow**: Agent lost track of earlier findings

### reasoning
Provide 1-2 sentences explaining the key factors that led to your difficulty score.
"""
