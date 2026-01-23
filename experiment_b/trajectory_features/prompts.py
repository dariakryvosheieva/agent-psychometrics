"""Prompts for trajectory feature extraction."""

TRAJECTORY_FEATURE_PROMPT = '''You are an expert at analyzing AI coding agent trajectories.

## Task
Analyze this trajectory of an AI agent attempting to solve a GitHub issue. Extract features that indicate the difficulty the agent experienced.

## Trajectory
Agent: {agent_name}
Task: {task_id}
Final outcome: {resolved}

### Agent Messages:
{trajectory_content}

## Features to Extract

Analyze the trajectory and rate each feature. Be precise and consistent.

1. **loop_detection** (0-5): How much did the agent repeat similar failed approaches?
   - 0: No repetition, each action was distinct
   - 1: Minor repetition (1-2 similar attempts)
   - 2: Some repetition with slight variations
   - 3: Moderate looping (3-5 similar failed attempts)
   - 4: Significant looping, agent clearly stuck
   - 5: Severe looping, agent kept trying same failed approach many times

2. **localization_quality** (0-5): How well did the agent identify the correct code location?
   - 0: Never found the right files or functions
   - 1: Found related files but wrong location
   - 2: Found correct file but wrong function/section
   - 3: Found correct general area
   - 4: Found correct location with some exploration
   - 5: Immediately identified exact location

3. **debugging_cycles** (count): Number of debug-fix cycles observed
   - Count how many times the agent: made a change, tested/observed an error, then tried a different fix
   - 0 means direct solution without iteration

4. **error_recovery** (0-5): How well did the agent recover from errors?
   - 0: Never encountered errors OR failed to recover from any
   - 1: Recovered from few errors with difficulty
   - 2: Recovered from some errors
   - 3: Reasonably good at recovering
   - 4: Good error recovery
   - 5: Excellent error recovery, learned from each mistake

5. **exploration_breadth** (count): How many distinct files or approaches did the agent explore?
   - Count distinct files opened/read AND distinct solution approaches tried

6. **focus_drift** (0-5): Did the agent stay on task or get distracted?
   - 0: Perfectly focused on the issue
   - 1: Mostly focused with minor tangents
   - 2: Some drift but mostly on task
   - 3: Moderate drift, explored unrelated areas
   - 4: Significant drift, much time spent on unrelated work
   - 5: Severe drift, agent got completely lost

7. **solution_completeness** (0-5): How complete was the attempted solution?
   - 0: No solution attempted
   - 1: Minimal attempt, clearly incomplete
   - 2: Partial solution, missing key components
   - 3: Reasonable attempt, some gaps
   - 4: Nearly complete solution
   - 5: Complete, thorough solution attempt

8. **edge_case_handling** (0-5): Did the agent consider and handle edge cases?
   - 0: No edge cases considered at all
   - 1: Missed obvious edge cases
   - 2: Addressed some edge cases but missed important ones
   - 3: Reasonable edge case coverage
   - 4: Good edge case handling, caught most issues
   - 5: Excellent, proactively identified and handled all edge cases

9. **test_verification** (0-5): Did the agent verify their solution works?
   - 0: No testing or verification attempted
   - 1: Minimal testing, didn't confirm fix works
   - 2: Some testing but incomplete verification
   - 3: Ran tests but didn't fully verify against the issue
   - 4: Good testing, verified fix addresses the issue
   - 5: Thorough testing including edge cases and regression checks

## Response Format

IMPORTANT: You MUST include ALL 9 numeric features. Do NOT omit any field.

Respond with ONLY a JSON object in this exact format:
{{
    "loop_detection": <0-5>,
    "localization_quality": <0-5>,
    "debugging_cycles": <count>,
    "error_recovery": <0-5>,
    "exploration_breadth": <count>,
    "focus_drift": <0-5>,
    "solution_completeness": <0-5>,
    "edge_case_handling": <0-5>,
    "test_verification": <0-5>,
    "reasoning": "<brief 2-3 sentence explanation of your ratings>"
}}

Every field above is REQUIRED. If a feature is not applicable (e.g., no focus drift observed), use 0.
'''


def format_trajectory_for_prompt(
    trajectory: dict,
    max_messages: int = 100,
    max_chars_per_message: int = 2000,
) -> str:
    """Format trajectory messages for inclusion in prompt.

    Args:
        trajectory: Trajectory dict with 'messages' list
        max_messages: Maximum number of messages to include
        max_chars_per_message: Maximum characters per message

    Returns:
        Formatted string of trajectory messages
    """
    messages = trajectory.get("messages", [])

    # If too many messages, sample beginning, middle, and end
    if len(messages) > max_messages:
        n_start = max_messages // 3
        n_end = max_messages // 3
        n_middle = max_messages - n_start - n_end

        middle_start = len(messages) // 2 - n_middle // 2
        middle_end = middle_start + n_middle

        sampled = (
            messages[:n_start] +
            [{"role": "system", "content": f"... [{len(messages) - max_messages} messages omitted] ..."}] +
            messages[middle_start:middle_end] +
            [{"role": "system", "content": "... [messages omitted] ..."}] +
            messages[-n_end:]
        )
        messages = sampled

    formatted_parts = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        # Truncate long messages
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message] + "\n... [truncated]"

        formatted_parts.append(f"[{role}]\n{content}")

    return "\n\n---\n\n".join(formatted_parts)
