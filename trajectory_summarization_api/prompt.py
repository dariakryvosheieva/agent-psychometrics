"""Summarization prompt template for API-based trajectory summarization."""

SUMMARIZATION_PROMPT = """You are summarizing an AI agent's attempt to solve a SWE-bench software engineering task. Create a concise summary that captures the agent's approach and key events.

## TASK
**Task ID:** {task_id}
**Repository:** {repo}
**Outcome:** {outcome}

**Problem Statement:**
{problem_statement}

## AGENT TRAJECTORY
{trajectory_text}

## INSTRUCTIONS
Create a BRIEF summary (~400 words, HARD CAP of 500 tokens). Be ruthlessly concise—no redundant details.

Cover these areas in flowing narrative (1-2 sentences each):

1. **Initial Approach**: Strategy for understanding and solving the task.

2. **Exploration & Localization**: Key files investigated. Did they find the relevant code?

3. **Key Observations**: Critical errors or outputs. Only include specific messages if essential.

4. **Solution Attempts**: Changes made. Include code only if it reveals the core approach.

5. **Final Outcome**: Success or failure? Key insight or failure point.

6. **Failure Analysis** (if FAILED): What specifically caused the failure? Examples:
   - Incorrect fix that didn't pass tests
   - Got stuck on a blocker (missing imports, complex dependencies)
   - Incomplete solution (found the issue but didn't implement fix)
   - Misunderstood the problem
   - Made edits but introduced new bugs

7. **Task Characteristics**: What made this task easy or difficult?

IMPORTANT:
- Write in THIRD PERSON (e.g., "The agent reproduced the bug..." not "I reproduced the bug...")
- Avoid redundancy. Do not repeat information across sections.
- Summarize repetitive actions in one phrase (e.g., "multiple failed edits").
- Only preserve code/errors absolutely essential for understanding.
- Omit boilerplate descriptions.

Summary:"""


def format_summarization_prompt(
    task_id: str,
    repo: str,
    resolved: bool,
    problem_statement: str,
    trajectory_text: str,
) -> str:
    """Format the summarization prompt with task-specific information.

    Args:
        task_id: SWE-bench task ID (e.g., "django__django-11179")
        repo: Repository name (e.g., "django/django")
        resolved: Whether the agent successfully solved the task
        problem_statement: The problem statement from SWE-bench
        trajectory_text: Full trajectory text (all messages concatenated)

    Returns:
        Formatted prompt string
    """
    outcome = "RESOLVED (Success)" if resolved else "FAILED"

    return SUMMARIZATION_PROMPT.format(
        task_id=task_id,
        repo=repo,
        outcome=outcome,
        problem_statement=problem_statement,
        trajectory_text=trajectory_text,
    )
