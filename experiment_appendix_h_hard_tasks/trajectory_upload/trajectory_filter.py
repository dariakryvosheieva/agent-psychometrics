"""
Filter SWE-bench trajectories to extract only editing behavior.

Removes chat messages, thoughts, and other potentially spurious signals,
keeping only the actions that modify code and their direct observations.

Supports both:
- Old SWE-agent format: {trajectory: [{action, observation}]}
- New unified format: {messages: [{role, content}]}

Usage:
    # Filter a single trajectory and print to stdout
    python llm_judge/trajectory_filter.py --traj path/to/task.traj

    # Filter a single trajectory and save to file
    python llm_judge/trajectory_filter.py --traj path/to/task.traj --output filtered.json

    # Filter all trajectories for an agent
    python llm_judge/trajectory_filter.py --agent 20240620_sweagent_claude3.5sonnet --output_dir filtered_trajs/

    # Filter unified format trajectories
    python llm_judge/trajectory_filter.py --unified path/to/unified.json --output filtered.json

    # Filter with model name redaction disabled
    python llm_judge/trajectory_filter.py --traj path/to/task.traj --no_redact
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


# Model/agent names to redact from observations
MODEL_PATTERNS = [
    r'claude', r'gpt-?4', r'gpt-?3\.?5', r'sonnet', r'opus', r'haiku',
    r'gemini', r'llama', r'mistral', r'qwen', r'deepseek',
    r'sweagent', r'swe-agent', r'autocoderrover', r'openhands',  # Note: autocodeRRover has two r's
    r'anthropic', r'openai', r'agentless', r'moatless',
]


def redact_model_info(text: str, patterns: list[str] = MODEL_PATTERNS) -> str:
    """Redact potential model-identifying information from text."""
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)
    return result


def is_codebase_interaction(action: str, observation: str = '') -> bool:
    """
    Determine if an action/observation pair represents actual codebase interaction.

    Only keeps actual actions and their results - filters out:
    - System prompts and setup context
    - Issue descriptions
    - Planning/thinking messages
    - Agent commentary about what it knows

    Args:
        action: The command/action text (may be user or assistant role depending on format)
        observation: The result/observation text

    Returns:
        True if this represents actual codebase interaction (command executed, file viewed, etc.)
    """
    action_lower = action.lower().strip()
    obs_lower = observation.lower()
    combined = action_lower + ' ' + obs_lower

    # === EXCLUSION PATTERNS (filter these out) ===

    # System prompts and setup
    setup_patterns = [
        action_lower.startswith("setting:"),
        action_lower.startswith("you are"),
        "autonomous programmer" in action_lower,
        "working directly in the command line" in action_lower,
        action_lower.startswith("instructions:"),
        action_lower.startswith("environment:"),
    ]
    if any(setup_patterns):
        return False

    # Issue descriptions and problem context
    issue_patterns = [
        "we're currently solving the following issue" in action_lower,
        "we are currently solving the following issue" in action_lower,
        "consider the following issue" in action_lower,
        "here's the issue text" in action_lower,
        "issue description" in action_lower,
        action_lower.startswith("issue:"),
        "problem statement" in action_lower,
        "bug report" in action_lower,
        "i've uploaded a python code repository" in action_lower,
        "<pr_description>" in action_lower,
        "<issue_description>" in action_lower,
        "uploaded_files" in action_lower and "/workspace" in action_lower,
        # Additional issue description patterns
        "the reported issue is:" in action_lower,
        "your task is to solve" in action_lower,
        "you're tasks is to solve" in action_lower,  # typo in original prompt
        "solve an issue reported" in action_lower,
        "here is the issue:" in action_lower,
        "please solve the following" in action_lower,
        "fix the following issue" in action_lower,
        "the following issue has been" in action_lower,
        "--- begin issue ---" in action_lower,
        "--- end issue ---" in action_lower,
        "begin issue" in action_lower and "end issue" in action_lower,
        # More issue patterns from various agent formats
        "solve this issue:" in action_lower,
        '"solve this issue' in action_lower,
        "<issue>" in action_lower,
        "</issue>" in action_lower,
        "consider the following issue" in action_lower,
        "issue description" in action_lower and "consider" in action_lower,
    ]
    if any(issue_patterns):
        return False

    # Planning/thinking/commentary messages (agent talking, not doing)
    thinking_patterns = [
        # Planning
        action_lower.startswith("let me think"),
        action_lower.startswith("let me ") and not any(x in action_lower for x in ['run', 'execute', 'check', 'look at the']),
        action_lower.startswith("let's ") and not any(x in action_lower for x in ['run', 'execute']),
        action_lower.startswith("i'll help"),
        action_lower.startswith("i will help"),
        action_lower.startswith("i need to"),
        action_lower.startswith("i should"),
        action_lower.startswith("i want to"),
        action_lower.startswith("i'm going to"),
        action_lower.startswith("my plan"),
        action_lower.startswith("plan:"),
        action_lower.startswith("next steps"),
        action_lower.startswith("first,"),
        action_lower.startswith("now,"),
        action_lower.startswith("now let"),
        action_lower.startswith("now that"),
        action_lower.startswith("to fix"),
        action_lower.startswith("to solve"),
        action_lower.startswith("we need to"),
        action_lower.startswith("we should"),
        "let's first" in action_lower,
        "i'll start by" in action_lower,
        "i will start by" in action_lower,

        # Commentary about results/observations
        action_lower.startswith("we are"),
        action_lower.startswith("we can see"),
        action_lower.startswith("we can confirm"),
        action_lower.startswith("it looks like"),
        action_lower.startswith("it seems"),
        action_lower.startswith("it appears"),
        action_lower.startswith("this shows"),
        action_lower.startswith("this indicates"),
        action_lower.startswith("this suggests"),
        action_lower.startswith("this confirms"),
        action_lower.startswith("the output"),
        action_lower.startswith("the result"),
        action_lower.startswith("the error"),
        action_lower.startswith("the code"),
        action_lower.startswith("the changes"),
        action_lower.startswith("the test"),
        action_lower.startswith("the fix"),
        action_lower.startswith("the issue"),
        action_lower.startswith("the problem"),
        action_lower.startswith("this issue"),
        action_lower.startswith("based on"),
        action_lower.startswith("looking at"),
        action_lower.startswith("as we can see"),
        action_lower.startswith("as shown"),
        action_lower.startswith("great!"),
        action_lower.startswith("perfect!"),
        action_lower.startswith("excellent!"),
        action_lower.startswith("good!"),
        action_lower.startswith("i apologize"),
        action_lower.startswith("i see"),
        action_lower.startswith("i notice"),
        action_lower.startswith("i found"),
        action_lower.startswith("summary"),
        action_lower.startswith("my edit"),
        "the issue suggests" in action_lower,
    ]
    if any(thinking_patterns):
        return False

    # Environment reminders
    if "environment reminder" in action_lower or "turns left" in action_lower:
        return False

    # === INCLUSION PATTERNS (keep these) ===

    # 1. Direct command patterns (SWE-agent style)
    command_patterns = [
        action_lower.startswith('edit '),
        action_lower.startswith('create '),
        action_lower.startswith('write '),
        action_lower.startswith('open '),
        action_lower.startswith('cat '),
        action_lower.startswith('view '),
        action_lower.startswith('find '),
        action_lower.startswith('grep '),
        action_lower.startswith('ls '),
        action_lower.startswith('cd '),
        action_lower.startswith('pwd'),
        action_lower.startswith('mkdir '),
        action_lower.startswith('rm '),
        action_lower.startswith('mv '),
        action_lower.startswith('cp '),
        action_lower.startswith('touch '),
        action_lower.startswith('head '),
        action_lower.startswith('tail '),
        action_lower.startswith('sed '),
        action_lower.startswith('awk '),
        action_lower.startswith('python'),
        action_lower.startswith('pytest'),
        action_lower.startswith('pip '),
        action_lower.startswith('make '),
        action_lower.startswith('bash '),
        action_lower.startswith('run '),
        action_lower.startswith('execute'),
        action_lower.startswith('git '),
        action_lower.startswith('str_replace'),
        action_lower.startswith('insert'),
        action_lower.startswith('append'),
        action_lower.startswith('search_file'),
        action_lower.startswith('search_dir'),
        action_lower.startswith('find_file'),
        action_lower.startswith('goto '),
        action_lower.startswith('scroll'),
        action_lower.startswith('submit'),
    ]
    if any(command_patterns):
        return True

    # 2. Terminal/shell indicators
    terminal_patterns = [
        '$ ' in action,  # Shell prompt
        action.strip().startswith('./'),  # Script execution
        action.strip().startswith('/'),   # Absolute path command
        '[?2004h' in action,  # Terminal escape codes
        'ubuntu@' in action,  # SSH prompt
        'root@' in action,    # Root prompt
        '(venv)' in action or '(env)' in action,  # Virtual env
    ]
    if any(terminal_patterns):
        return True

    # 3. Observation indicators (actual output)
    observation_patterns = [
        'observation:' in obs_lower,
        '[file:' in obs_lower,
        'lines total' in obs_lower,
        'lines above' in obs_lower,
        'lines below' in obs_lower,
        'traceback' in obs_lower,
        'error:' in obs_lower and ('line ' in obs_lower or 'file ' in obs_lower),
        'syntaxerror' in obs_lower,
        'modulenotfounderror' in obs_lower,
        'importerror' in obs_lower,
        'passed' in obs_lower and 'failed' in obs_lower,  # Test output
        '====' in observation and 'test' in obs_lower,    # Pytest output
        'exit code' in obs_lower,
    ]
    if any(observation_patterns):
        return True

    # 4. File path indicators (suggests file operation)
    file_patterns = [
        '.py' in combined,
        '.js' in combined,
        '.ts' in combined,
        '.java' in combined,
        '.go' in combined,
        '.rs' in combined,
        '.cpp' in combined,
        '.c' in combined,
        '.rb' in combined,
        '/src/' in combined,
        '/test' in combined,
        '/lib/' in combined,
    ]
    # Only match file patterns if there's some action context
    if any(file_patterns) and (len(action) < 500 or _has_code_content(observation)):
        # Check it's not just mentioning files in prose
        if any(c in action_lower for c in ['file', 'open', 'edit', 'read', 'write', 'view', 'cat', 'look']):
            return True

    # 5. Code modification indicators
    code_mod_patterns = [
        'end_of_edit' in action_lower,
        '<<<' in action and '>>>' in action,  # Diff markers
        'diff --git' in action,
        '+++ ' in action and '--- ' in action,
        '@@ ' in action,  # Diff hunk markers
        'write committed' in obs_lower,
        'file written' in obs_lower,
        'file created' in obs_lower,
        'file modified' in obs_lower,
    ]
    if any(code_mod_patterns):
        return True

    # 6. JSON tool use patterns (OpenHands, etc.)
    tool_patterns = [
        '"action"' in action_lower and '"args"' in action_lower,
        '"command"' in action_lower,
        '"tool"' in action_lower,
        'execute_bash' in action_lower,
        'str_replace_editor' in action_lower,
    ]
    if any(tool_patterns):
        return True

    # 7. Actual code content in action (inline edits)
    if _has_code_content(action) and len(action) > 50:
        return True

    return False


def _has_code_content(text: str) -> bool:
    """Check if text contains actual code/file content."""
    indicators = [
        'def ' in text and '(' in text,
        'class ' in text and ':' in text,
        'import ' in text,
        'from ' in text and ' import ' in text,
        'function ' in text,
        'const ' in text or 'let ' in text or 'var ' in text,
        'return ' in text,
        '#!/' in text,
        'if __name__' in text,
        '[File:' in text,
        'lines total]' in text,
    ]
    return any(indicators)


def _is_observation_content(text: str) -> bool:
    """
    Check if text contains observation/output content (sonar-agent style where USER=observation).

    This detects command outputs, file contents, test results, etc. that would typically
    appear as observations but are labeled as USER messages in some formats.
    """
    text_lower = text.lower()

    # File content indicators
    file_content_patterns = [
        "here's the result of running" in text_lower,
        "here's the files and directories" in text_lower,
        'cat -n' in text_lower,
        'lines total]' in text_lower,
        '[file:' in text_lower,
        'the file' in text_lower and 'has been edited' in text_lower,
        'file written' in text_lower,
        'file created' in text_lower,
    ]
    if any(file_content_patterns):
        return True

    # Command output indicators
    output_patterns = [
        text.strip().startswith('./'),  # File path listing
        text.strip().startswith('/testbed'),
        text.strip().startswith('/workspace'),
        'traceback (most recent call last)' in text_lower,
        'error:' in text_lower and 'line ' in text_lower,
        'test ' in text_lower and ('passed' in text_lower or 'failed' in text_lower),
        '====' in text and 'test' in text_lower,
        'observation:' in text_lower,
    ]
    if any(output_patterns):
        return True

    # Numbered line output (like cat -n)
    if re.search(r'^\s*\d+[\t\|]', text[:100]):
        return True

    # Multiple file paths listed
    if text.count('.py') > 2 and text.count('/') > 3 and len(text) < 2000:
        return True

    return False


def _is_action_description(text: str) -> bool:
    """
    Check if assistant text describes COMPLETED actions (Amazon-Q style narratives).

    Only matches PAST TENSE descriptions of actions already taken, not:
    - Future plans ("I will open", "Let's edit")
    - Analysis ("The method is...", "Looking at the code...")
    - Thinking ("I need to...", "We should...")
    - Summary/commentary about results

    These are for assistant-only trajectories that describe behavioral choices
    in narrative form rather than command/output pairs.
    """
    text_lower = text.lower()

    # === EXCLUSION: Planning/thinking (not completed actions) ===
    planning_patterns = [
        text_lower.startswith("let me"),
        text_lower.startswith("let's"),
        text_lower.startswith("i'll"),
        text_lower.startswith("i will"),
        text_lower.startswith("i need"),
        text_lower.startswith("i should"),
        text_lower.startswith("i want"),
        text_lower.startswith("we need"),
        text_lower.startswith("we should"),
        text_lower.startswith("first,"),
        text_lower.startswith("now,"),
        text_lower.startswith("next,"),
        text_lower.startswith("to fix"),
        text_lower.startswith("to solve"),
        text_lower.startswith("the issue"),
        text_lower.startswith("the problem"),
        text_lower.startswith("based on"),
        text_lower.startswith("looking at"),
        text_lower.startswith("analysis:"),
        "i'm going to" in text_lower,
    ]
    if any(planning_patterns):
        return False

    # === EXCLUSION: Summary/commentary (not action descriptions) ===
    summary_patterns = [
        text_lower.startswith("great!"),
        text_lower.startswith("perfect!"),
        text_lower.startswith("excellent!"),
        text_lower.startswith("good!"),
        text_lower.startswith("done!"),
        text_lower.startswith("ok!"),
        text_lower.startswith("this"),
        text_lower.startswith("the fix"),
        text_lower.startswith("the change"),
        text_lower.startswith("the modification"),
        text_lower.startswith("to summarize"),
        text_lower.startswith("in summary"),
        text_lower.startswith("summary:"),
        "to summarize" in text_lower,
        "the issue has been" in text_lower,
        "the problem has been" in text_lower,
        "this solution" in text_lower,
        "this fix" in text_lower,
        "let's analyze" in text_lower,
        "we can see" in text_lower,
        "we can confirm" in text_lower,
        "as shown" in text_lower,
        "as we can see" in text_lower,
    ]
    if any(summary_patterns):
        return False

    # === INCLUSION: Past tense action descriptions ===
    # Only match COMPLETED actions (past tense)
    completed_action_patterns = [
        'opened file' in text_lower,
        'opened the file' in text_lower,
        'edited file' in text_lower,
        'edited the file' in text_lower,
        'modified the' in text_lower,
        'updated the' in text_lower,
        'created file' in text_lower,
        'created the file' in text_lower,
        'wrote to' in text_lower,
        'ran the test' in text_lower,
        'executed the' in text_lower,
        'inspected the' in text_lower,
        'searched for' in text_lower,
        'found the' in text_lower and ('file' in text_lower or 'method' in text_lower or 'class' in text_lower or 'issue' in text_lower),
        'located the' in text_lower,
        'closed the file' in text_lower,
        'selected the' in text_lower,
        'i have successfully' in text_lower,
        'successfully updated' in text_lower,
        'successfully modified' in text_lower,
        'successfully fixed' in text_lower,
        'applied the' in text_lower and ('patch' in text_lower or 'fix' in text_lower or 'change' in text_lower),
        'have inspected' in text_lower,
        'have opened' in text_lower,
        'have edited' in text_lower,
        'have modified' in text_lower,
        'have updated' in text_lower,
    ]

    if any(completed_action_patterns):
        return True

    # Specific command descriptions with file paths (Amazon-Q style)
    # "- open file\n  - file path: django/forms/fields.py"
    if 'file path:' in text_lower and ('open file' in text_lower or 'edit file' in text_lower or 'close file' in text_lower):
        return True

    # "Here are the set of commands:" followed by actual commands
    if 'here are the set of commands' in text_lower:
        return True

    # === INCLUSION: Patch/diff output (actual code changes) ===
    # Some agents output patches directly as their response
    patch_patterns = [
        text_lower.startswith('diff --git'),
        text_lower.startswith('[repair]'),
        text_lower.startswith('patch:'),
        text_lower.startswith('--- a/'),
        text_lower.startswith('+++ b/'),
        'diff --git' in text_lower[:500],  # Patch within first 500 chars
    ]
    if any(patch_patterns):
        return True

    # === INCLUSION: JSON tool call patterns (moatless, OpenHands style) ===
    # Messages that are JSON objects representing tool calls/actions
    json_action_patterns = [
        text.strip().startswith('{') and 'action_args_class' in text_lower,
        text.strip().startswith('{') and ('file_path' in text_lower or 'directory' in text_lower or 'pattern' in text_lower),
        '"action"' in text_lower and '"args"' in text_lower,
    ]
    if any(json_action_patterns):
        return True

    # === INCLUSION: Stage logs with processing content ===
    # Some agents output logs that describe their processing steps
    # But exclude logs that primarily contain issue descriptions
    log_patterns = [
        text_lower.startswith('[file_level_stage'),
        text_lower.startswith('[related_elements_stage'),
        text_lower.startswith('[repair_stage'),
        text_lower.startswith('[repair_log]'),
        text_lower.startswith('[reproduce_log'),
    ]
    # Check if log contains issue description (should filter these)
    issue_in_log = (
        "we are currently solving the following issue" in text_lower or
        "we're currently solving the following issue" in text_lower or
        "consider the following issue" in text_lower or
        "<issue>" in text_lower
    )
    if any(log_patterns) and not issue_in_log:
        return True

    return False


# Keep old name as alias for backwards compatibility
is_edit_related_action = is_codebase_interaction


def detect_format(data: dict) -> str:
    """
    Detect the trajectory format.

    Returns:
        'unified' for new unified format with messages
        'legacy' for old SWE-agent format with trajectory
        'unknown' otherwise
    """
    if 'messages' in data and isinstance(data['messages'], list):
        # Check if messages have role/content structure
        if data['messages'] and isinstance(data['messages'][0], dict):
            if 'role' in data['messages'][0] and 'content' in data['messages'][0]:
                return 'unified'

    if 'trajectory' in data and isinstance(data['trajectory'], list):
        # Check if trajectory has action/observation structure
        if data['trajectory'] and isinstance(data['trajectory'][0], dict):
            if 'action' in data['trajectory'][0] or 'observation' in data['trajectory'][0]:
                return 'legacy'

    return 'unknown'


def filter_unified_trajectory(
    trajectory: dict,
    redact_models: bool = True,
    keep_system: bool = False,
) -> dict:
    """
    Filter a unified format trajectory to keep only editing-related messages.

    Args:
        trajectory: Unified trajectory dict with {messages: [{role, content}]}
        redact_models: Whether to redact model names from content
        keep_system: Whether to keep system messages

    Returns:
        Filtered trajectory dict
    """
    messages = trajectory.get('messages', [])
    filtered_messages = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get('role', '')
        content = msg.get('content', '')

        # Handle system messages
        if role == 'system':
            if keep_system:
                if redact_models:
                    content = redact_model_info(content)
                filtered_messages.append({
                    'role': role,
                    'content': content,
                    'timestamp': msg.get('timestamp'),
                })
            i += 1
            continue

        # For user messages, check if it's an action OR an observation (some formats invert roles)
        if role == 'user':
            # Handle nested content
            user_text = content
            if isinstance(content, list):
                user_text = content[0].get('text', '') if content else ''

            # Skip empty messages
            if not user_text or not user_text.strip():
                i += 1
                continue

            # Look ahead for the observation (assistant response)
            observation = ''
            if i + 1 < len(messages) and messages[i + 1].get('role') == 'assistant':
                observation = messages[i + 1].get('content', '')
                if isinstance(observation, list):
                    observation = observation[0].get('text', '') if observation else ''

            # Check if this user message is actually an observation (sonar-agent style)
            # These contain file contents, command outputs, etc.
            if _is_observation_content(user_text):
                if redact_models:
                    user_text = redact_model_info(user_text)
                filtered_messages.append({
                    'role': 'user',
                    'content': user_text,
                    'timestamp': msg.get('timestamp'),
                })
                i += 1
                continue

            # Standard case: user message is an action
            if is_codebase_interaction(user_text, observation):
                action_content = user_text
                if redact_models:
                    action_content = redact_model_info(action_content)

                filtered_messages.append({
                    'role': 'user',
                    'content': action_content,
                    'timestamp': msg.get('timestamp'),
                })

            # Don't automatically include the assistant message - it will be evaluated
            # independently when we reach it (via _is_action_description check)
            i += 1
            continue

        # For standalone assistant messages, check if they describe actions (Amazon-Q style)
        if role == 'assistant':
            # Handle nested content
            assistant_text = content
            if isinstance(content, list):
                assistant_text = content[0].get('text', '') if content else ''

            # Check if this assistant message describes behavioral actions
            if _is_action_description(assistant_text):
                if redact_models:
                    assistant_text = redact_model_info(assistant_text)

                filtered_messages.append({
                    'role': 'assistant',
                    'content': assistant_text,
                    'timestamp': msg.get('timestamp'),
                })

            i += 1
            continue

        # Skip other messages
        i += 1

    # Build filtered trajectory
    result = {
        'task_id': trajectory.get('task_id', ''),
        'agent': trajectory.get('agent', ''),
        'resolved': trajectory.get('resolved', False),
        'messages': filtered_messages,
        'metadata': trajectory.get('metadata', {}).copy(),
    }

    # Add filtering metadata
    result['metadata']['_filtered'] = True
    result['metadata']['_original_messages'] = len(messages)
    result['metadata']['_filtered_messages'] = len(filtered_messages)

    return result


def filter_legacy_trajectory(
    trajectory: dict,
    redact_models: bool = True,
    keep_thoughts: bool = False,
) -> dict:
    """
    Filter a legacy SWE-agent format trajectory to keep only editing-related actions.

    Args:
        trajectory: Raw SWE-bench trajectory dict with {trajectory: [{action, observation}]}
        redact_models: Whether to redact model names from observations
        keep_thoughts: Whether to keep 'thought' and 'response' fields

    Returns:
        Filtered trajectory dict with only edit-related steps
    """
    traj_list = trajectory.get('trajectory', [])
    filtered_steps = []

    for step in traj_list:
        action = step.get('action', '')
        observation = step.get('observation', '')

        if is_edit_related_action(action, observation):
            # Process observation
            if redact_models:
                observation = redact_model_info(observation)

            filtered_step = {
                'action': action,
                'observation': observation,
            }

            # Optionally keep thought/response (usually omitted to avoid model signatures)
            if keep_thoughts:
                if 'thought' in step:
                    thought = step['thought']
                    if redact_models:
                        thought = redact_model_info(thought)
                    filtered_step['thought'] = thought
                if 'response' in step:
                    response = step['response']
                    if redact_models:
                        response = redact_model_info(response)
                    filtered_step['response'] = response

            filtered_steps.append(filtered_step)

    return {
        'trajectory': filtered_steps,
        'info': trajectory.get('info', {}),
        'environment': trajectory.get('environment', ''),
        '_filtered': True,
        '_original_steps': len(traj_list),
        '_filtered_steps': len(filtered_steps),
    }


def filter_trajectory(
    trajectory: dict,
    redact_models: bool = True,
    keep_thoughts: bool = False,
) -> dict:
    """
    Filter a trajectory to keep only editing-related actions.
    Auto-detects format (unified or legacy).

    Args:
        trajectory: Raw trajectory dict (unified or legacy format)
        redact_models: Whether to redact model names from observations
        keep_thoughts: Whether to keep 'thought' and 'response' fields (legacy only)

    Returns:
        Filtered trajectory dict with only edit-related steps
    """
    fmt = detect_format(trajectory)

    if fmt == 'unified':
        return filter_unified_trajectory(trajectory, redact_models=redact_models)
    elif fmt == 'legacy':
        return filter_legacy_trajectory(trajectory, redact_models=redact_models, keep_thoughts=keep_thoughts)
    else:
        # Try legacy format as fallback
        return filter_legacy_trajectory(trajectory, redact_models=redact_models, keep_thoughts=keep_thoughts)


def load_trajectory(path: Path) -> dict:
    """Load a trajectory from a .traj file."""
    with open(path) as f:
        return json.load(f)


def save_trajectory(trajectory: dict, path: Path) -> None:
    """Save a trajectory to a JSON file."""
    with open(path, 'w') as f:
        json.dump(trajectory, f, indent=2)


def filter_agent_trajectories(
    agent_dir: Path,
    output_dir: Path,
    redact_models: bool = True,
    keep_thoughts: bool = False,
) -> dict:
    """
    Filter all trajectories for an agent.

    Returns:
        Summary dict with counts
    """
    trajs_dir = agent_dir / 'trajs'
    if not trajs_dir.exists():
        raise FileNotFoundError(f"No trajectories found at {trajs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'total': 0,
        'filtered': 0,
        'original_steps': 0,
        'filtered_steps': 0,
    }

    for traj_path in sorted(trajs_dir.glob('*.traj')):
        traj = load_trajectory(traj_path)
        filtered = filter_trajectory(traj, redact_models=redact_models, keep_thoughts=keep_thoughts)

        output_path = output_dir / traj_path.name
        save_trajectory(filtered, output_path)

        summary['total'] += 1
        summary['filtered'] += 1
        summary['original_steps'] += filtered['_original_steps']
        summary['filtered_steps'] += filtered['_filtered_steps']

    return summary


def print_filtered_trajectory(trajectory: dict, max_obs_length: int = 500) -> None:
    """Pretty-print a filtered trajectory. Handles both unified and legacy formats."""
    fmt = detect_format(trajectory)

    if fmt == 'unified' or 'messages' in trajectory:
        # Unified format
        messages = trajectory.get('messages', [])
        metadata = trajectory.get('metadata', {})
        original = metadata.get('_original_messages', metadata.get('total_steps', '?'))
        filtered = metadata.get('_filtered_messages', len(messages))

        print(f"Task: {trajectory.get('task_id', 'unknown')}")
        print(f"Agent: {trajectory.get('agent', 'unknown')}")
        print(f"Resolved: {trajectory.get('resolved', '?')}")
        print(f"Filtered messages: {filtered}/{original}")
        print("=" * 80)

        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')

            # Truncate long content
            if len(content) > max_obs_length:
                content = content[:max_obs_length] + f"... [{len(content) - max_obs_length} more chars]"

            print(f"\n[{i+1}] {role}")
            print(content)
    else:
        # Legacy format
        steps = trajectory.get('trajectory', [])
        original = trajectory.get('_original_steps', '?')
        filtered = trajectory.get('_filtered_steps', len(steps))

        print(f"Filtered trajectory: {filtered}/{original} steps kept")
        print("=" * 80)

        for i, step in enumerate(steps):
            action = step.get('action', '')
            observation = step.get('observation', '')

            # Truncate long observations
            if len(observation) > max_obs_length:
                observation = observation[:max_obs_length] + f"... [{len(observation) - max_obs_length} more chars]"

            print(f"\n[Step {i+1}]")
            print(f"ACTION: {action[:200]}{'...' if len(action) > 200 else ''}")
            print(f"OBSERVATION: {observation}")


def filter_unified_directory(
    input_dir: Path,
    output_dir: Path,
    redact_models: bool = True,
) -> dict:
    """
    Filter all unified format trajectories in a directory.

    Returns:
        Summary dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'total': 0,
        'filtered': 0,
        'original_messages': 0,
        'filtered_messages': 0,
    }

    for traj_path in sorted(input_dir.glob('*.json')):
        traj = load_trajectory(traj_path)

        # Skip non-unified format
        if detect_format(traj) != 'unified':
            continue

        filtered = filter_unified_trajectory(traj, redact_models=redact_models)

        output_path = output_dir / traj_path.name
        save_trajectory(filtered, output_path)

        summary['total'] += 1
        summary['filtered'] += 1
        summary['original_messages'] += filtered['metadata'].get('_original_messages', 0)
        summary['filtered_messages'] += filtered['metadata'].get('_filtered_messages', 0)

    return summary


def filter_all_unified(
    input_base_dir: Path,
    output_base_dir: Path,
    redact_models: bool = True,
) -> dict:
    """
    Filter all unified format trajectories across all agent directories.

    Args:
        input_base_dir: Base directory containing agent subdirectories (e.g., trajectory_data/unified_trajs/)
        output_base_dir: Base output directory
        redact_models: Whether to redact model names

    Returns:
        Summary dict with counts per agent and totals
    """
    # Find all agent directories
    agent_dirs = sorted([
        d for d in input_base_dir.iterdir()
        if d.is_dir() and not d.name.startswith('_') and not d.name.startswith('.')
    ])

    all_summary = {
        'agents_processed': 0,
        'total_trajectories': 0,
        'total_original_messages': 0,
        'total_filtered_messages': 0,
        'agents': {},
    }

    print(f"Found {len(agent_dirs)} agent directories")

    for i, agent_dir in enumerate(agent_dirs):
        agent_name = agent_dir.name

        # Check if has JSON files
        json_files = list(agent_dir.glob('*.json'))
        if not json_files:
            continue

        print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name} ({len(json_files)} files)")

        output_dir = output_base_dir / agent_name

        try:
            summary = filter_unified_directory(
                agent_dir,
                output_dir,
                redact_models=redact_models,
            )

            all_summary['agents_processed'] += 1
            all_summary['total_trajectories'] += summary['filtered']
            all_summary['total_original_messages'] += summary['original_messages']
            all_summary['total_filtered_messages'] += summary['filtered_messages']

            all_summary['agents'][agent_name] = {
                'trajectories': summary['filtered'],
                'original_messages': summary['original_messages'],
                'filtered_messages': summary['filtered_messages'],
            }

            reduction = 0
            if summary['original_messages'] > 0:
                reduction = 100 * (1 - summary['filtered_messages'] / summary['original_messages'])
            print(f"  -> {summary['filtered']} trajectories, {reduction:.1f}% reduction")

        except Exception as e:
            print(f"  -> Error: {e}")

    return all_summary


def main():
    parser = argparse.ArgumentParser(
        description='Filter SWE-bench trajectories to editing behavior only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View filtered trajectory (auto-detects format)
  python llm_judge/trajectory_filter.py --traj experiments/evaluation/verified/20240620_sweagent_claude3.5sonnet/trajs/django__django-10880.traj

  # Filter unified format trajectory
  python llm_judge/trajectory_filter.py --traj chris_output/unified_trajs/agent/task.json --output filtered.json

  # Save filtered trajectory
  python llm_judge/trajectory_filter.py --traj path/to/task.traj --output filtered.json

  # Filter all trajectories for an agent (legacy format)
  python llm_judge/trajectory_filter.py --agent 20240620_sweagent_claude3.5sonnet --output_dir chris_output/filtered_trajs/

  # Filter a directory of unified format trajectories
  python llm_judge/trajectory_filter.py --unified_dir trajectory_data/unified_trajs/agent --output_dir chris_output/filtered_unified/

  # Filter ALL unified trajectories across all agents
  python llm_judge/trajectory_filter.py --all_unified --output_dir trajectory_data/filtered_unified/
        """
    )

    parser.add_argument('--traj', type=str, help='Path to a single trajectory file (any format)')
    parser.add_argument('--agent', type=str, help='Agent name to filter all legacy trajectories for')
    parser.add_argument('--unified_dir', type=str, help='Directory of unified format trajectories to filter')
    parser.add_argument('--all_unified', action='store_true', help='Filter ALL unified trajectories in trajectory_data/unified_trajs/')
    parser.add_argument('--unified_base', type=str, default='trajectory_data/unified_trajs', help='Base directory for --all_unified (default: trajectory_data/unified_trajs)')
    parser.add_argument('--output', type=str, help='Output path for single trajectory')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch filtering')
    parser.add_argument('--no_redact', action='store_true', help='Disable model name redaction')
    parser.add_argument('--keep_thoughts', action='store_true', help='Keep thought/response fields (legacy only)')
    parser.add_argument('--keep_system', action='store_true', help='Keep system messages (unified only)')
    parser.add_argument('--max_obs', type=int, default=500, help='Max observation length to print (default: 500)')

    args = parser.parse_args()

    if args.traj:
        # Filter single trajectory (auto-detect format)
        traj_path = Path(args.traj)
        if not traj_path.exists():
            print(f"Error: File not found: {traj_path}")
            return

        traj = load_trajectory(traj_path)
        fmt = detect_format(traj)
        print(f"Detected format: {fmt}")

        filtered = filter_trajectory(
            traj,
            redact_models=not args.no_redact,
            keep_thoughts=args.keep_thoughts
        )

        if args.output:
            save_trajectory(filtered, Path(args.output))
            print(f"Saved filtered trajectory to {args.output}")

            # Print summary based on format
            if fmt == 'unified':
                orig = filtered['metadata'].get('_original_messages', '?')
                filt = filtered['metadata'].get('_filtered_messages', '?')
                print(f"  Original messages: {orig}")
                print(f"  Filtered messages: {filt}")
            else:
                print(f"  Original steps: {filtered.get('_original_steps', '?')}")
                print(f"  Filtered steps: {filtered.get('_filtered_steps', '?')}")
        else:
            print_filtered_trajectory(filtered, max_obs_length=args.max_obs)

    elif args.all_unified:
        # Filter ALL unified trajectories across all agents
        input_base = Path(args.unified_base)
        if not input_base.exists():
            print(f"Error: Directory not found: {input_base}")
            return

        output_base = Path(args.output_dir) if args.output_dir else Path('trajectory_data/filtered_unified')

        print(f"Filtering ALL unified trajectories in {input_base}...")
        print(f"Output directory: {output_base}")
        print("=" * 60)

        summary = filter_all_unified(
            input_base,
            output_base,
            redact_models=not args.no_redact,
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Agents processed: {summary['agents_processed']}")
        print(f"Total trajectories: {summary['total_trajectories']}")
        print(f"Total original messages: {summary['total_original_messages']}")
        print(f"Total filtered messages: {summary['total_filtered_messages']}")
        if summary['total_original_messages'] > 0:
            reduction = 100 * (1 - summary['total_filtered_messages'] / summary['total_original_messages'])
            print(f"Overall reduction: {reduction:.1f}%")
        print(f"Output directory: {output_base}")

        # Save summary JSON
        summary_path = output_base / '_filter_summary.json'
        output_base.mkdir(parents=True, exist_ok=True)
        import json as json_module
        with open(summary_path, 'w') as f:
            json_module.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    elif args.unified_dir:
        # Filter directory of unified format trajectories
        input_dir = Path(args.unified_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f'{input_dir.name}_filtered'

        print(f"Filtering unified trajectories in {input_dir}...")
        summary = filter_unified_directory(
            input_dir,
            output_dir,
            redact_models=not args.no_redact,
        )

        print(f"\nFiltered {summary['filtered']} trajectories")
        print(f"  Original messages: {summary['original_messages']}")
        print(f"  Filtered messages: {summary['filtered_messages']}")
        if summary['original_messages'] > 0:
            print(f"  Reduction: {100 * (1 - summary['filtered_messages'] / summary['original_messages']):.1f}%")
        print(f"  Output directory: {output_dir}")

    elif args.agent:
        # Filter all trajectories for agent (legacy format)
        experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'
        agent_dir = experiments_dir / 'evaluation' / 'verified' / args.agent

        if not agent_dir.exists():
            print(f"Error: Agent directory not found: {agent_dir}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path(f'chris_output/filtered_trajs/{args.agent}')

        print(f"Filtering trajectories for {args.agent}...")
        summary = filter_agent_trajectories(
            agent_dir,
            output_dir,
            redact_models=not args.no_redact,
            keep_thoughts=args.keep_thoughts
        )

        print(f"\nFiltered {summary['filtered']} trajectories")
        print(f"  Original steps: {summary['original_steps']}")
        print(f"  Filtered steps: {summary['filtered_steps']}")
        if summary['original_steps'] > 0:
            print(f"  Reduction: {100 * (1 - summary['filtered_steps'] / summary['original_steps']):.1f}%")
        print(f"  Output directory: {output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
