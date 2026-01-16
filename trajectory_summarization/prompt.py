"""Summarization prompt template for trajectory summarization."""

SUMMARIZATION_PROMPT = """You are analyzing a SWE-bench agent trajectory. Write a free-form summary focusing on what this trajectory reveals about task difficulty.

## Task Information
- Task ID: {task_id}
- Agent: {agent}
- Outcome: {outcome}

## Full Trajectory
{full_trajectory}

## Instructions
Write a concise narrative summary (target ~400 words, max 500) covering:

1. What approach did the agent take? (systematic debugging, trial-and-error, etc.)
2. What parts of the codebase did the agent explore?
3. What difficulties or blockers did the agent encounter?
4. Did the agent successfully identify the root cause?
5. What was the agent's final action or state?
6. What does this trajectory suggest about the task's difficulty?

Focus on difficulty-relevant signals: complexity of the fix, domain knowledge required, whether the problem was clear, and how much exploration was needed.

Write your summary as a single flowing paragraph or two, not a bulleted list."""


def format_summarization_prompt(
    task_id: str,
    agent: str,
    resolved: bool,
    full_trajectory: str,
) -> str:
    """Format the summarization prompt with task-specific information.

    Args:
        task_id: SWE-bench task ID (e.g., "django__django-11179")
        agent: Agent name/ID
        resolved: Whether the agent successfully solved the task
        full_trajectory: Full trajectory text (all messages concatenated)

    Returns:
        Formatted prompt string
    """
    outcome = "Resolved (Success)" if resolved else "Failed"

    return SUMMARIZATION_PROMPT.format(
        task_id=task_id,
        agent=agent,
        outcome=outcome,
        full_trajectory=full_trajectory,
    )
