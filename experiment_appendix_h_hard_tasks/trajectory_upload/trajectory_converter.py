"""
Convert SWE-bench trajectories from all formats to unified JSON.

Supported formats:
- traj_sweagent: .traj files with {trajectory: [{action, observation}]}
- yaml_history: .yaml files with {history: [{role, content}]}
- json_traj_like: .json with same structure as .traj
- json_chat_list: .json as [{role, content}]
- json_solver: .json as [{event_type, event_data}]
- json_langchain: .json with nested langchain message format
- log_text: .log text files with sections
- txt_amazon_q: no extension, amazon-q format
- txt_artemis: .txt with timestamps
- jsonl_navie: .jsonl with {type, message}

Usage:
    # Convert ALL agents
    python llm_judge/trajectory_converter.py --all

    # Convert single agent
    python llm_judge/trajectory_converter.py --agent 20240620_sweagent_claude3.5sonnet

    # Preview a file
    python llm_judge/trajectory_converter.py --input path/to/file --preview
"""

import argparse
import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Message:
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[str] = None


@dataclass
class UnifiedTrajectory:
    task_id: str
    agent: str
    resolved: bool
    messages: list[Message]
    metadata: dict


# Format detection - order matters (most specific first)
def detect_format(file_path: Path, data: any = None) -> str:
    """Detect the trajectory format."""
    ext = file_path.suffix.lower()

    # Text-based formats (detect before loading)
    if ext == '.log':
        return 'log_text'
    if ext == '.txt':
        return 'txt_text'
    if ext == '' or ext == file_path.name:  # No extension
        return 'txt_noext'
    if ext == '.md':
        return 'markdown'
    if ext == '.jsonl':
        return 'jsonl_navie'

    # For JSON/YAML, we need the data
    if data is None:
        return 'unknown'

    # Dict-based formats
    if isinstance(data, dict):
        # testCommandTrajectory (wandb)
        if 'testCommandTrajectory' in data:
            return 'json_wandb'

        # attempt_N with traj.UUID nested format (epam 20250228)
        if any(k.startswith('attempt_') for k in data.keys()):
            return 'json_attempt_uuid'

        # instance_id + messages (Skywork, SAGE_OpenHands)
        if 'instance_id' in data and 'messages' in data:
            return 'json_instance_messages'

        # llm_call_data (epam variants)
        if 'llm_call_data' in data:
            return 'json_llm_call_data'

        # history with action/message (zai_glm)
        if 'history' in data and isinstance(data['history'], list):
            if data['history'] and isinstance(data['history'][0], dict):
                if 'action' in data['history'][0] or 'message' in data['history'][0]:
                    return 'json_zai_history'
            # Regular yaml-style history
            if data['history'] and isinstance(data['history'][0], dict) and 'role' in data['history'][0]:
                return 'yaml_history'

        # gru format: trajectory is a dict with content
        if 'trajectory' in data and isinstance(data['trajectory'], dict):
            if 'content' in data['trajectory']:
                return 'json_gru'

        # Standard sweagent trajectory (list of action/observation)
        if 'trajectory' in data and isinstance(data['trajectory'], list):
            if data['trajectory'] and isinstance(data['trajectory'][0], dict):
                if 'action' in data['trajectory'][0]:
                    return 'traj_sweagent'

        # Single content field (aime_coder) - has rationale, tool, parameters, etc.
        if 'content' in data and ('rationale' in data or 'tool' in data):
            return 'json_single_content'

    # List-based formats
    if isinstance(data, list) and len(data) > 0:
        first = data[0]

        # List of strings (learn_by_interact)
        if isinstance(first, str):
            return 'json_string_list'

        if isinstance(first, dict):
            # step_idx format (frogboss)
            if 'step_idx' in first and 'thought' in first:
                return 'json_frogboss'

            # blocks format (sonar-foundation)
            if 'role' in first and 'blocks' in first:
                return 'json_blocks'

            # role/content chat list
            if 'role' in first and 'content' in first:
                return 'json_chat_list'

            # event_type (solver)
            if 'event_type' in first:
                return 'json_solver'

            # problem_statement format (SWE-Exp_DeepSeek)
            if 'problem_statement' in first:
                return 'json_problem_statement'

            # UUID keys (epam variant)
            keys = list(first.keys())
            if keys and len(keys[0]) > 30 and '-' in keys[0]:  # UUID-like
                return 'json_uuid_messages'

        # List of lists (langchain)
        if isinstance(first, list):
            return 'json_langchain'

    return 'unknown'


def load_file(file_path: Path) -> tuple[any, str]:
    """Load a trajectory file and detect its format."""
    ext = file_path.suffix.lower()

    # Text-based formats
    if ext in ('.log', '.txt', '.md', '') or ext == file_path.name:
        with open(file_path, 'r', errors='replace') as f:
            content = f.read()
        fmt = detect_format(file_path)
        return content, fmt

    # JSONL - detect actual format from first line
    if ext == '.jsonl':
        lines = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        # Detect JSONL format based on first entry
        if lines:
            first = lines[0]
            if isinstance(first, dict):
                # SWE-Fixer format: {instance_id, prediction} or {instance_id, model_patch}
                if 'instance_id' in first and ('prediction' in first or 'model_patch' in first or 'model_reasoning' in first):
                    return lines, 'jsonl_swefixer'
                # Navie format: {type, message: {role, content}}
                if 'message' in first and isinstance(first.get('message'), dict):
                    return lines, 'jsonl_navie'
        return lines, 'jsonl_navie'

    # JSON
    if ext == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        fmt = detect_format(file_path, data)
        return data, fmt

    # YAML
    if ext == '.yaml':
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        fmt = detect_format(file_path, data)
        return data, fmt

    # .traj (JSON - but detect actual format from content)
    if ext == '.traj':
        with open(file_path, 'r', errors='replace') as f:
            content = f.read()
        # Check if it's plain text (Lingxi format) vs JSON
        if content.strip().startswith('<') or content.strip().startswith('```') or not (content.strip().startswith('{') or content.strip().startswith('[')):
            return content, 'traj_text'
        try:
            data = json.loads(content)
            # Some .traj files have llm_call_data format instead of trajectory
            fmt = detect_format(file_path, data)
            if fmt == 'unknown':
                fmt = 'traj_sweagent'  # Default for .traj
            return data, fmt
        except json.JSONDecodeError:
            return content, 'traj_text'

    return None, 'unknown'


# ============== Format Converters ==============

def convert_traj_sweagent(data: dict, file_path: Path) -> list[Message]:
    """Convert SWE-agent .traj format."""
    messages = []

    for step in data.get('trajectory', []):
        action = step.get('action', '')
        observation = step.get('observation', '')

        if action:
            messages.append(Message(role='user', content=action))
        if observation:
            messages.append(Message(role='assistant', content=observation))

    return messages


def _normalize_content(content) -> str:
    """Normalize message content to string.

    Some agents (OpenHands, codesweep) store content as a list of content blocks
    like [{'type': 'text', 'text': '...'}] instead of a plain string.
    """
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", str(item)))
            else:
                text_parts.append(str(item))
        return " ".join(text_parts)
    return str(content) if content else ""


def convert_yaml_history(data: dict, file_path: Path) -> list[Message]:
    """Convert YAML history format."""
    messages = []

    for msg in data.get('history', []):
        role = msg.get('role', 'user')
        content = _normalize_content(msg.get('content', ''))

        if role == 'system':
            messages.append(Message(role='system', content=content))
        elif role == 'assistant':
            messages.append(Message(role='assistant', content=content))
        else:
            messages.append(Message(role='user', content=content))

    return messages


def convert_json_chat_list(data: list, file_path: Path) -> list[Message]:
    """Convert JSON chat list format [{role, content}]."""
    messages = []

    for msg in data:
        if not isinstance(msg, dict):
            continue
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        # Normalize role
        if role in ('system',):
            messages.append(Message(role='system', content=str(content)))
        elif role in ('assistant', 'ai'):
            messages.append(Message(role='assistant', content=str(content)))
        else:
            messages.append(Message(role='user', content=str(content)))

    return messages


def convert_json_solver(data: list, file_path: Path) -> list[Message]:
    """Convert solver JSON format [{event_type, event_data}]."""
    messages = []

    for event in data:
        event_type = event.get('event_type', '')
        event_data = event.get('event_data', {})

        if event_type == 'agent_thought':
            content = event_data.get('thought', '') or event_data.get('content', '')
            if content:
                messages.append(Message(role='assistant', content=str(content)))
        elif event_type == 'tool_call':
            content = json.dumps(event_data) if isinstance(event_data, dict) else str(event_data)
            messages.append(Message(role='user', content=content))
        elif event_type == 'tool_result':
            content = event_data.get('result', '') or str(event_data)
            messages.append(Message(role='assistant', content=str(content)))
        elif event_type in ('retrieval', 'solver_log'):
            content = str(event_data) if event_data else event_type
            messages.append(Message(role='assistant', content=content))

    return messages


def convert_json_langchain(data: list, file_path: Path) -> list[Message]:
    """Convert langchain nested format.

    Handles formats like:
    - [[{kwargs: {content, type}}]] (double nested)
    - [[[{kwargs: ...}]]] (triple nested - composio_swekit)
    - [{type: "constructor", id: [...], kwargs: {content}}]
    """
    messages = []

    def extract_message(item):
        if isinstance(item, dict):
            # Langchain constructor format with 'id' array
            if 'type' in item and 'id' in item and 'kwargs' in item:
                kwargs = item['kwargs']
                content = kwargs.get('content', '')
                # Get role from id array (e.g., ["langchain", "schema", "messages", "SystemMessage"])
                msg_type = item['id'][-1] if isinstance(item['id'], list) else ''
                if 'System' in msg_type:
                    role = 'system'
                elif 'AI' in msg_type or 'Assistant' in msg_type:
                    role = 'assistant'
                else:
                    role = 'user'
                return content, role
            # Standard format with kwargs
            elif 'kwargs' in item:
                kwargs = item['kwargs']
                content = kwargs.get('content', '')
                role = kwargs.get('type', kwargs.get('role', 'user'))
                return content, role
            elif 'content' in item:
                return item.get('content', ''), item.get('role', 'user')
        return None, None

    def process_items(items, depth=0):
        """Recursively process nested lists up to depth 3."""
        if depth > 3:
            return
        for item in items:
            if isinstance(item, list):
                process_items(item, depth + 1)
            else:
                content, role = extract_message(item)
                if content:
                    role_norm = 'assistant' if role in ('ai', 'assistant', 'AIMessage') else 'system' if role == 'system' else 'user'
                    messages.append(Message(role=role_norm, content=str(content)[:10000]))

    process_items(data)
    return messages


def convert_jsonl_navie(data: list, file_path: Path) -> list[Message]:
    """Convert JSONL navie format [{type, message}]."""
    messages = []

    for item in data:
        msg = item.get('message', {})
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            timestamp = item.get('timestamp')

            role_norm = 'assistant' if role == 'assistant' else 'system' if role == 'system' else 'user'
            messages.append(Message(role=role_norm, content=str(content), timestamp=timestamp))

    return messages


def convert_jsonl_swefixer(data: list, file_path: Path) -> list[Message]:
    """Convert SWE-Fixer JSONL format [{instance_id, prediction/model_patch, model_reasoning}].

    Each line is a separate task, so we concatenate all into messages.
    """
    messages = []

    for item in data:
        if not isinstance(item, dict):
            continue

        instance_id = item.get('instance_id', '')

        # Get the reasoning/prediction content
        reasoning = item.get('model_reasoning', '') or item.get('prediction', '')
        if isinstance(reasoning, dict):
            # prediction might be a JSON string with reasoning_process
            reasoning = reasoning.get('reasoning process', '') or str(reasoning)

        patch = item.get('model_patch', '')

        # Add as messages
        if reasoning:
            messages.append(Message(role='assistant', content=f"[{instance_id}] {str(reasoning)[:10000]}"))
        if patch:
            messages.append(Message(role='assistant', content=f"[{instance_id}] Patch:\n{str(patch)[:5000]}"))

    return messages


def convert_log_text(content: str, file_path: Path) -> list[Message]:
    """Convert text log format."""
    messages = []

    # Check if this is a JSON log format (amazon.nova-premier style)
    # Format: {"iteration": N, "input": [{role, content}], "response": "..."}
    if content.strip().startswith('{') and '"iteration"' in content:
        try:
            # Parse as JSON (may be one line or multiple iterations)
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    entry = json.loads(line)
                    # Extract input messages
                    for msg in entry.get('input', []):
                        if isinstance(msg, dict) and msg.get('content'):
                            role = msg.get('role', 'user')
                            content_val = msg['content']
                            # Handle list content (multimodal)
                            if isinstance(content_val, list):
                                content_val = ' '.join(
                                    str(c.get('text', c)) if isinstance(c, dict) else str(c)
                                    for c in content_val
                                )
                            if role == 'system':
                                role = 'user'  # Normalize
                            messages.append(Message(role=role, content=str(content_val)[:10000]))
                    # Extract response
                    if entry.get('response'):
                        messages.append(Message(role='assistant', content=str(entry['response'])[:10000]))
                except json.JSONDecodeError:
                    continue
            if messages:
                return messages
        except Exception:
            pass  # Fall through to regular log parsing

    # Split by common section delimiters
    # MASAI format: ========== timestamp Section:
    # Agentless format: ### Section or timestamp - LEVEL - message

    sections = re.split(r'(?:={5,}|#{3,})\s*', content)

    for section in sections:
        section = section.strip()
        if not section or len(section) < 10:
            continue

        # Try to determine if it's an action or observation
        if any(kw in section.lower() for kw in ['output:', 'result:', 'response:', 'observation:']):
            messages.append(Message(role='assistant', content=section[:10000]))
        else:
            messages.append(Message(role='user', content=section[:10000]))

    return messages


def convert_txt_noext(content: str, file_path: Path) -> list[Message]:
    """Convert text files without extension (amazon-q, lingma, etc.)."""
    messages = []

    # Amazon-Q format: "assistant: ... \n- command\n  - param: value\n======"
    # Try to split by role markers

    parts = re.split(r'\n(?=assistant:|user:|human:)', content, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.lower().startswith('assistant:'):
            messages.append(Message(role='assistant', content=part[10:].strip()[:10000]))
        elif part.lower().startswith('user:') or part.lower().startswith('human:'):
            prefix_len = 5 if part.lower().startswith('user:') else 6
            messages.append(Message(role='user', content=part[prefix_len:].strip()[:10000]))
        else:
            # Heuristic: if contains code or commands, likely user action
            messages.append(Message(role='user', content=part[:10000]))

    return messages


def convert_txt_text(content: str, file_path: Path) -> list[Message]:
    """Convert .txt format (artemis, etc.)."""
    messages = []

    # Artemis format: timestamp | LEVEL | module:func:line - content
    # Try to extract meaningful content

    lines = content.split('\n')
    current_content = []

    for line in lines:
        # Skip timestamp/log prefix
        if ' | ' in line:
            # Extract content after the log prefix
            parts = line.split(' - ', 1)
            if len(parts) > 1:
                current_content.append(parts[1])
            else:
                current_content.append(line)
        else:
            current_content.append(line)

    # Join and create a single message or split by patterns
    full_content = '\n'.join(current_content)
    if full_content.strip():
        messages.append(Message(role='assistant', content=full_content[:50000]))

    return messages


def convert_json_instance_messages(data: dict, file_path: Path) -> list[Message]:
    """Convert {instance_id, messages} format (Skywork, SAGE_OpenHands)."""
    messages = []
    for msg in data.get('messages', []):
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            role_norm = 'assistant' if role in ('assistant', 'ai') else 'system' if role == 'system' else 'user'
            messages.append(Message(role=role_norm, content=str(content)))
    return messages


def convert_json_llm_call_data(data: dict, file_path: Path) -> list[Message]:
    """Convert {llm_call_data: [{role, content}]} format (epam variants).

    Content can be:
    - A string
    - A list of dicts with {type, text} or {type, tool_use, input}
    """
    messages = []
    for msg in data.get('llm_call_data', []):
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Handle complex content (list of content blocks)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if 'text' in block:
                            text_parts.append(block['text'])
                        elif block.get('type') == 'tool_use':
                            tool_name = block.get('name', 'tool')
                            tool_input = block.get('input', {})
                            text_parts.append(f"[Tool: {tool_name}] {json.dumps(tool_input)[:500]}")
                        elif block.get('type') == 'tool_result':
                            text_parts.append(f"[Tool Result] {str(block.get('content', ''))[:500]}")
                content = '\n'.join(text_parts)

            role_norm = 'assistant' if role in ('assistant', 'ai') else 'system' if role == 'system' else 'user'
            if content:
                messages.append(Message(role=role_norm, content=str(content)[:10000]))
    return messages


def convert_json_zai_history(data: dict, file_path: Path) -> list[Message]:
    """Convert {history: [{action, message, ...}]} format (zai_glm)."""
    messages = []
    for item in data.get('history', []):
        if isinstance(item, dict):
            action = item.get('action', '')
            message = item.get('message', '')
            source = item.get('source', '')

            # Determine role based on source or action
            if source == 'agent' or action:
                content = f"{action}: {message}" if action and message else (action or message)
                messages.append(Message(role='assistant', content=str(content)[:10000]))
            elif message:
                messages.append(Message(role='user', content=str(message)[:10000]))
    return messages


def convert_json_gru(data: dict, file_path: Path) -> list[Message]:
    """Convert gru format where trajectory is a dict with content."""
    messages = []
    traj = data.get('trajectory', {})
    if isinstance(traj, dict):
        content = traj.get('content', '')
        if content:
            # Content is usually a long string - split into chunks or keep as one
            messages.append(Message(role='assistant', content=str(content)[:50000]))
    return messages


def convert_json_single_content(data: dict, file_path: Path) -> list[Message]:
    """Convert {content: "..."} format (aime_coder)."""
    content = data.get('content', '')
    if content:
        return [Message(role='assistant', content=str(content)[:50000])]
    return []


def convert_json_string_list(data: list, file_path: Path) -> list[Message]:
    """Convert list of strings format (learn_by_interact)."""
    messages = []
    for i, item in enumerate(data):
        if isinstance(item, str) and item.strip():
            # Alternate between user and assistant, or use heuristics
            role = 'assistant' if i % 2 == 1 else 'user'
            messages.append(Message(role=role, content=item[:10000]))
    return messages


def convert_json_frogboss(data: list, file_path: Path) -> list[Message]:
    """Convert [{step_idx, thought, action, observation}] format (frogboss)."""
    messages = []
    for step in data:
        if isinstance(step, dict):
            thought = step.get('thought', '')
            action = step.get('action', '')
            observation = step.get('observation', '')

            # Thought/action as assistant, observation as result
            if thought:
                messages.append(Message(role='assistant', content=str(thought)[:10000]))
            if action:
                messages.append(Message(role='user', content=str(action)[:10000]))
            if observation:
                messages.append(Message(role='assistant', content=str(observation)[:10000]))
    return messages


def convert_json_uuid_messages(data: list, file_path: Path) -> list[Message]:
    """Convert [{uuid: {author_name, message}}] format (epam variant)."""
    messages = []
    for item in data:
        if isinstance(item, dict):
            # Get first (and usually only) value
            for uuid_key, msg_data in item.items():
                if isinstance(msg_data, dict):
                    author = msg_data.get('author_name', '')
                    message = msg_data.get('message', '')
                    role = 'assistant' if 'thought' in author.lower() or 'agent' in author.lower() else 'user'
                    if message:
                        messages.append(Message(role=role, content=str(message)[:10000]))
    return messages


def convert_markdown(content: str, file_path: Path) -> list[Message]:
    """Convert markdown format (usually README, not real trajectory)."""
    if 'README' in file_path.name.upper():
        return []  # Skip README files

    # Treat as documentation
    return [Message(role='system', content=content[:10000])]


def convert_json_wandb(data: dict, file_path: Path) -> list[Message]:
    """Convert wandb format {testCommandTrajectory: [{role, content, tool_calls}]}."""
    messages = []
    for msg in data.get('testCommandTrajectory', []):
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Handle tool_calls
            tool_calls = msg.get('tool_calls', [])
            if tool_calls and not content:
                tool_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get('function', {})
                        name = func.get('name', 'tool')
                        args = func.get('arguments', '')
                        tool_parts.append(f"[{name}] {args[:500]}")
                content = '\n'.join(tool_parts)

            role_norm = 'assistant' if role in ('assistant', 'ai') else 'system' if role == 'system' else 'user'
            if content:
                messages.append(Message(role=role_norm, content=str(content)[:10000]))
    return messages


def convert_json_attempt_uuid(data: dict, file_path: Path) -> list[Message]:
    """Convert attempt_N/traj/UUID format (epam 20250228)."""
    messages = []
    # Iterate over attempts
    for attempt_key, attempt_data in data.items():
        if not attempt_key.startswith('attempt_'):
            continue
        traj = attempt_data.get('traj', {}) if isinstance(attempt_data, dict) else {}
        # traj has UUID keys
        for uuid_key, msg_data in traj.items():
            if isinstance(msg_data, dict):
                author = msg_data.get('author_name', '')
                message = msg_data.get('message', '')
                role = 'assistant' if 'thought' in author.lower() or 'agent' in author.lower() else 'user'
                if message:
                    messages.append(Message(role=role, content=str(message)[:10000]))
    return messages


def convert_json_blocks(data: list, file_path: Path) -> list[Message]:
    """Convert blocks format (sonar-foundation): [{role, blocks: [{block_type, text}]}]."""
    messages = []
    for msg in data:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            blocks = msg.get('blocks', [])
            text_parts = []
            for block in blocks:
                if isinstance(block, dict):
                    text = block.get('text', '')
                    if text:
                        text_parts.append(text)
            content = '\n'.join(text_parts)
            role_norm = 'assistant' if role in ('assistant', 'ai') else 'system' if role == 'system' else 'user'
            if content:
                messages.append(Message(role=role_norm, content=content[:10000]))
    return messages


def convert_json_problem_statement(data: list, file_path: Path) -> list[Message]:
    """Convert problem_statement format (SWE-Exp_DeepSeek): [{problem_statement, ...}]."""
    messages = []
    for item in data:
        if isinstance(item, dict):
            problem = item.get('problem_statement', '')
            model_patch = item.get('model_patch', '')
            if problem:
                messages.append(Message(role='user', content=str(problem)[:10000]))
            if model_patch:
                messages.append(Message(role='assistant', content=str(model_patch)[:10000]))
    return messages


def convert_traj_text(content: str, file_path: Path) -> list[Message]:
    """Convert plain text .traj files (Lingxi format - XML-like with issue/response)."""
    messages = []

    # Try to extract sections from XML-like format
    # <issue_description>...</issue_description>
    # <response>...</response>
    import re

    issue_match = re.search(r'<issue_description>(.*?)</issue_description>', content, re.DOTALL)
    if issue_match:
        messages.append(Message(role='user', content=issue_match.group(1).strip()[:10000]))

    response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
    if response_match:
        messages.append(Message(role='assistant', content=response_match.group(1).strip()[:10000]))

    # If no XML tags, treat as plain text
    if not messages and content.strip():
        messages.append(Message(role='assistant', content=content.strip()[:50000]))

    return messages


# ============== Main Conversion Logic ==============

CONVERTERS = {
    'traj_sweagent': convert_traj_sweagent,
    'yaml_history': convert_yaml_history,
    'json_chat_list': convert_json_chat_list,
    'json_solver': convert_json_solver,
    'json_langchain': convert_json_langchain,
    'jsonl_navie': convert_jsonl_navie,
    'log_text': convert_log_text,
    'txt_noext': convert_txt_noext,
    'txt_text': convert_txt_text,
    'markdown': convert_markdown,
    # New formats added for previously failing agents
    'json_instance_messages': convert_json_instance_messages,
    'json_llm_call_data': convert_json_llm_call_data,
    'json_zai_history': convert_json_zai_history,
    'json_gru': convert_json_gru,
    'json_single_content': convert_json_single_content,
    'json_string_list': convert_json_string_list,
    'json_frogboss': convert_json_frogboss,
    'json_uuid_messages': convert_json_uuid_messages,
    'json_wandb': convert_json_wandb,
    'json_attempt_uuid': convert_json_attempt_uuid,
    'json_blocks': convert_json_blocks,
    'json_problem_statement': convert_json_problem_statement,
    'traj_text': convert_traj_text,
    'jsonl_swefixer': convert_jsonl_swefixer,
}


def convert_trajectory(
    file_path: Path,
    agent_name: str,
    task_id: Optional[str] = None,
    resolved: bool = False,
) -> Optional[UnifiedTrajectory]:
    """Convert a trajectory file to unified format."""

    if task_id is None:
        task_id = file_path.stem
        # Clean up task_id (remove _traj suffix, instance_ prefix, etc.)
        task_id = re.sub(r'_traj$', '', task_id)
        # Handle instance_ prefix (seen in blackboxai agent)
        if task_id.startswith('instance_'):
            task_id = task_id[9:]  # Strip "instance_" prefix

    try:
        data, fmt = load_file(file_path)
    except Exception as e:
        print(f"  Error loading {file_path.name}: {e}")
        return None

    if fmt == 'unknown' or data is None:
        print(f"  Unknown format: {file_path.name}")
        return None

    converter = CONVERTERS.get(fmt)
    if converter is None:
        print(f"  No converter for format: {fmt}")
        return None

    try:
        messages = converter(data, file_path)
    except Exception as e:
        print(f"  Error converting {file_path.name}: {e}")
        return None

    if not messages:
        return None

    # Build metadata
    metadata = {
        'source_format': fmt,
        'source_file': str(file_path),
        'total_steps': len(messages),
    }

    # Try to extract environment info
    if isinstance(data, dict):
        if 'environment' in data:
            metadata['environment'] = data['environment']

    return UnifiedTrajectory(
        task_id=task_id,
        agent=agent_name,
        resolved=resolved,
        messages=messages,
        metadata=metadata,
    )


def trajectory_to_dict(traj: UnifiedTrajectory) -> dict:
    """Convert to JSON-serializable dict."""
    return {
        'task_id': traj.task_id,
        'agent': traj.agent,
        'resolved': traj.resolved,
        'messages': [asdict(m) for m in traj.messages],
        'metadata': traj.metadata,
    }


def load_results(agent_dir: Path) -> dict[str, bool]:
    """Load results.json to get resolved status."""
    results_path = agent_dir / 'results' / 'results.json'
    if not results_path.exists():
        return {}

    try:
        with open(results_path) as f:
            results = json.load(f)
        resolved_set = set(results.get('resolved', []))
        return {task: True for task in resolved_set}
    except Exception:
        return {}


def convert_nested_directory(
    task_dir: Path,
    agent_name: str,
    task_id: str,
    resolved: bool,
) -> Optional[UnifiedTrajectory]:
    """Convert a nested directory (where each task is a subdirectory) to unified format.

    Supports multiple nested formats:
    - lingma: debug_agent_write_patch_1.json with [{role, content}]
    - autocoderover: attempt_0/*.json with [{role, content}]
    - cortexa: 0_file_retrieval.json, 1_direct_prompt.json, etc.
    - moatless: trajectory.json with {nodes: [{user_message, action_steps}]}
    - JoyCode: completed_trajectory.json with {instance_id, test_generation, ...}
    - nfactorial: *.logs files
    - patchpilot: all task logs in one directory
    """
    messages = []
    source_format = 'nested_unknown'

    # Check for specific file patterns
    files = list(task_dir.iterdir())
    file_names = [f.name for f in files]

    # moatless: trajectory.json
    if 'trajectory.json' in file_names:
        source_format = 'nested_moatless'
        try:
            with open(task_dir / 'trajectory.json') as f:
                data = json.load(f)
            for node in data.get('nodes', []):
                if node.get('user_message'):
                    messages.append(Message(role='user', content=str(node['user_message'])[:10000]))
                for step in node.get('action_steps', []):
                    if isinstance(step, dict):
                        action = step.get('action', {})
                        if isinstance(action, dict):
                            content = action.get('thoughts', '') or action.get('output', '') or str(action)
                            messages.append(Message(role='assistant', content=str(content)[:10000]))
        except Exception as e:
            print(f"    Error parsing moatless: {e}")

    # JoyCode: completed_trajectory.json
    elif 'completed_trajectory.json' in file_names:
        source_format = 'nested_joycode'
        try:
            with open(task_dir / 'completed_trajectory.json') as f:
                data = json.load(f)
            # Extract all agent_trajectory_content sections
            sections = ['test_generation', 'diff_generation', 'code_repair']
            for section in sections:
                content = data.get(section, {}).get('agent_trajectory_content', '')
                if content:
                    messages.append(Message(role='assistant', content=f"[{section}]\n{str(content)[:30000]}"))
        except Exception as e:
            print(f"    Error parsing joycode: {e}")

    # swerl: resolved/*.txt or failed/*.txt with Prompt/Response/Patch sections
    elif any(f in ('resolved', 'failed') for f in file_names):
        source_format = 'nested_swerl'
        for subdir_name in ['resolved', 'failed']:
            subdir = task_dir / subdir_name
            if subdir.exists() and subdir.is_dir():
                for txt_file in sorted(subdir.glob('*.txt')):
                    try:
                        content = txt_file.read_text(errors='replace')
                        # Parse Prompt/Response/Patch sections
                        import re
                        sections = re.split(r'={10,}\s*(Prompt|Response|Patch)\s*={10,}', content)
                        for i in range(1, len(sections), 2):
                            section_name = sections[i]
                            section_content = sections[i + 1].strip() if i + 1 < len(sections) else ''
                            if section_content:
                                role = 'user' if section_name == 'Prompt' else 'assistant'
                                messages.append(Message(role=role, content=f"[{subdir_name}/{section_name}]\n{section_content[:15000]}"))
                    except Exception as e:
                        print(f"    Error parsing swerl {txt_file.name}: {e}")

    # sweagent nested traj: <task_id>.traj files inside task directory
    elif any(f.endswith('.traj') and not f.endswith('.traj.json') for f in file_names):
        source_format = 'nested_sweagent_traj'
        for f in sorted(files):
            if f.suffix == '.traj' and f.is_file():
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    # Same format as regular SWE-agent traj
                    if isinstance(data.get('trajectory'), list):
                        for step in data['trajectory']:
                            if isinstance(step, dict):
                                action = step.get('action', '')
                                observation = step.get('observation', '')
                                thought = step.get('thought', '')
                                if thought:
                                    messages.append(Message(role='assistant', content=f"[Thought]\n{thought[:5000]}"))
                                if action:
                                    messages.append(Message(role='user', content=f"[Action]\n{action[:5000]}"))
                                if observation:
                                    messages.append(Message(role='assistant', content=f"[Observation]\n{observation[:10000]}"))
                except Exception as e:
                    print(f"    Error parsing nested traj {f.name}: {e}")

    # cortexa: numbered json files (0_file_retrieval.json, etc.)
    # Must check BEFORE lingma since files like 2_localization_agent.json contain _agent.json
    elif any(f[0].isdigit() and f.endswith('.json') for f in file_names):
        source_format = 'nested_cortexa'
        for f in sorted(files, key=lambda x: x.name):
            if f.suffix == '.json' and f.name[0].isdigit():
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    # cortexa has model-specific keys
                    for model_name, model_data in data.items():
                        if isinstance(model_data, dict):
                            prompt = model_data.get('user_prompt', '') or model_data.get('system_prompt', '')
                            response = model_data.get('response', '')
                            if prompt:
                                messages.append(Message(role='user', content=str(prompt)[:10000]))
                            if response:
                                # Response can be a string or a list
                                if isinstance(response, list):
                                    response = '\n'.join(str(r) for r in response[:50])
                                messages.append(Message(role='assistant', content=str(response)[:10000]))
                except Exception as e:
                    print(f"    Error parsing cortexa {f.name}: {e}")

    # lingma: debug_agent_write_patch_*.json OR *_agent.json
    elif any('debug_agent' in f or f.endswith('_agent.json') for f in file_names):
        source_format = 'nested_lingma'
        for f in sorted(files):
            if f.suffix == '.json' and ('debug_agent' in f.name or f.name.endswith('_agent.json')):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        for msg in data:
                            if isinstance(msg, dict) and 'content' in msg:
                                role = msg.get('role', 'user')
                                role_norm = 'assistant' if role == 'assistant' else 'system' if role == 'system' else 'user'
                                messages.append(Message(role=role_norm, content=str(msg['content'])[:10000]))
                except Exception as e:
                    print(f"    Error parsing lingma {f.name}: {e}")

    # autocoderover: attempt_* directories
    elif any(f.startswith('attempt_') for f in file_names):
        source_format = 'nested_autocoderover'
        for attempt_dir in sorted(task_dir.iterdir()):
            if attempt_dir.is_dir() and attempt_dir.name.startswith('attempt_'):
                for json_file in sorted(attempt_dir.glob('*.json')):
                    if json_file.name in ('regression_test_result_0.json',):
                        continue
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for msg in data:
                                if isinstance(msg, dict) and 'content' in msg:
                                    role = msg.get('role', 'user')
                                    role_norm = 'assistant' if role == 'assistant' else 'system' if role == 'system' else 'user'
                                    messages.append(Message(role=role_norm, content=str(msg['content'])[:10000]))
                    except Exception as e:
                        print(f"    Error parsing autocoderover {json_file.name}: {e}")

    # nfactorial/patchpilot: .log or .logs files
    elif any(f.endswith('.log') or f.endswith('.logs') for f in file_names):
        source_format = 'nested_logs'
        for f in sorted(files):
            if f.suffix in ('.log', '.logs') and f.is_file():
                try:
                    content = f.read_text(errors='replace')
                    messages.append(Message(role='assistant', content=content[:50000]))
                except Exception as e:
                    print(f"    Error reading log {f.name}: {e}")

    # SAGE_bash_only / livesweagent: *.traj.json with {info: {submission}, trajectory: [...]}
    elif any(f.endswith('.traj.json') for f in file_names):
        source_format = 'nested_traj_json'
        for f in sorted(files):
            if f.name.endswith('.traj.json'):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    # Extract trajectory if present
                    if isinstance(data.get('trajectory'), list):
                        for step in data['trajectory']:
                            if isinstance(step, dict):
                                action = step.get('action', '')
                                observation = step.get('observation', '')
                                if action:
                                    messages.append(Message(role='user', content=str(action)[:10000]))
                                if observation:
                                    messages.append(Message(role='assistant', content=str(observation)[:10000]))
                    # If no trajectory, extract submission as fallback
                    elif data.get('info', {}).get('submission'):
                        messages.append(Message(role='assistant', content=str(data['info']['submission'])[:10000]))
                except Exception as e:
                    print(f"    Error parsing traj.json {f.name}: {e}")

    if not messages:
        return None

    metadata = {
        'source_format': source_format,
        'source_file': str(task_dir),
        'total_steps': len(messages),
        'has_patch': False,
    }

    return UnifiedTrajectory(
        task_id=task_id,
        agent=agent_name,
        resolved=resolved,
        messages=messages,
        metadata=metadata,
    )


def convert_agent_trajectories(
    agent_dir: Path,
    output_dir: Path,
    verified_task_ids: set | None = None,
) -> dict:
    """Convert all trajectories for an agent.

    Args:
        agent_dir: Path to agent directory containing trajs/
        output_dir: Path to output directory
        verified_task_ids: Optional set of task IDs to include (filters to Verified only)
    """
    agent_name = agent_dir.name
    trajs_dir = agent_dir / 'trajs'

    if not trajs_dir.exists():
        return {'total': 0, 'converted': 0, 'errors': 0, 'formats': {}}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(agent_dir)

    summary = {
        'total': 0,
        'converted': 0,
        'errors': 0,
        'skipped': 0,
        'filtered': 0,  # Filtered out by verified_task_ids
        'formats': {},
    }

    # Check if this is a nested directory format (subdirectories per task)
    items = list(trajs_dir.iterdir())
    has_task_subdirs = any(
        item.is_dir() and '__' in item.name  # task IDs have format repo__issue
        for item in items
    )

    # Check for category subdirectories (patchpilot, agentless_MCTS style)
    # Structure: trajs/category_dir/<task_id>.log or <task_id>.jsonl
    # Includes: patchpilot (localization, repair, reproduce, validation dirs)
    #           agentless_MCTS (file_level_stage_logs, related_elements_stage_logs, etc.)
    non_hidden_items = [item for item in items if not item.name.startswith('.')]
    has_category_subdirs = (
        not has_task_subdirs and
        len(non_hidden_items) > 0 and
        all(item.is_dir() for item in non_hidden_items)
    )

    if has_task_subdirs:
        # Nested directory format - each task is a subdirectory
        for task_dir in items:
            if not task_dir.is_dir():
                continue
            if task_dir.name.startswith('.'):
                continue

            summary['total'] += 1
            task_id = task_dir.name

            # Filter by verified task IDs if specified
            if verified_task_ids is not None and task_id not in verified_task_ids:
                summary['filtered'] += 1
                continue

            resolved = results.get(task_id, False)

            converted = convert_nested_directory(task_dir, agent_name, task_id, resolved)

            if converted is None:
                summary['errors'] += 1
                continue

            fmt = converted.metadata.get('source_format', 'unknown')
            summary['formats'][fmt] = summary['formats'].get(fmt, 0) + 1

            output_path = output_dir / f"{task_id}.json"
            with open(output_path, 'w') as f:
                json.dump(trajectory_to_dict(converted), f, indent=2)

            summary['converted'] += 1
    elif has_category_subdirs:
        # Category subdirectory format (patchpilot, agentless_MCTS)
        # Structures supported:
        #   1. trajs/category_dir/<task_id>.log (patchpilot-v1.1)
        #   2. trajs/category_dir/output_*.jsonl with instance_id (patchpilot_Co-PatcheR)
        #   3. trajs/category_dir/nested_logs_dir/<task_id>.log (agentless_MCTS)
        task_data: dict[str, list[tuple[str, str]]] = {}  # task_id -> [(category, content)]

        for category_dir in non_hidden_items:
            if not category_dir.is_dir():
                continue
            category_name = category_dir.name

            # Check for direct .log files (patchpilot-v1.1 style)
            for log_file in category_dir.iterdir():
                if log_file.is_file() and log_file.suffix == '.log':
                    task_id = log_file.stem
                    try:
                        content = log_file.read_text(errors='replace')
                        task_data.setdefault(task_id, []).append((category_name, content[:30000]))
                    except Exception as e:
                        print(f"    Error reading {log_file}: {e}")

                # Check for .jsonl files (patchpilot_Co-PatcheR style)
                elif log_file.is_file() and log_file.suffix == '.jsonl':
                    try:
                        for line in log_file.read_text(errors='replace').strip().split('\n'):
                            if not line.strip():
                                continue
                            try:
                                entry = json.loads(line)
                                if isinstance(entry, dict) and 'instance_id' in entry:
                                    task_id = entry['instance_id']
                                    # Extract patch or relevant content
                                    patch = entry.get('model_patch', '') or entry.get('prediction', '')
                                    if patch:
                                        task_data.setdefault(task_id, []).append((category_name, f"Patch:\n{patch[:10000]}"))
                            except json.JSONDecodeError:
                                continue
                    except Exception as e:
                        print(f"    Error reading {log_file}: {e}")

                # Check for nested log directories (agentless_MCTS style)
                elif log_file.is_dir():
                    nested_dir = log_file
                    for nested_file in nested_dir.iterdir():
                        if nested_file.is_file() and nested_file.suffix == '.log':
                            task_id = nested_file.stem
                            try:
                                content = nested_file.read_text(errors='replace')
                                full_category = f"{category_name}/{nested_dir.name}"
                                task_data.setdefault(task_id, []).append((full_category, content[:30000]))
                            except Exception as e:
                                print(f"    Error reading {nested_file}: {e}")

        for task_id, category_contents in task_data.items():
            summary['total'] += 1

            # Filter by verified task IDs if specified
            if verified_task_ids is not None and task_id not in verified_task_ids:
                summary['filtered'] += 1
                continue

            resolved = results.get(task_id, False)

            # Combine all content for this task
            messages = []
            for category, content in sorted(category_contents):
                messages.append(Message(role='assistant', content=f"[{category}]\n{content}"))

            if not messages:
                summary['errors'] += 1
                continue

            metadata = {
                'source_format': 'category_logs',
                'source_file': str(trajs_dir),
                'total_steps': len(messages),
                'has_patch': False,
            }

            converted = UnifiedTrajectory(
                task_id=task_id,
                agent=agent_name,
                resolved=resolved,
                messages=messages,
                metadata=metadata,
            )

            summary['formats']['category_logs'] = summary['formats'].get('category_logs', 0) + 1

            output_path = output_dir / f"{task_id}.json"
            with open(output_path, 'w') as f:
                json.dump(trajectory_to_dict(converted), f, indent=2)

            summary['converted'] += 1
    else:
        # Standard format - files in trajs/
        traj_files = [f for f in items if f.is_file()]

        for traj_path in traj_files:
            summary['total'] += 1

            # Skip certain files
            if traj_path.name.startswith('.') or traj_path.name == 'README.md':
                summary['skipped'] += 1
                continue

            task_id = traj_path.stem
            task_id = re.sub(r'_traj$', '', task_id)
            # Handle instance_ prefix (seen in blackboxai agent)
            if task_id.startswith('instance_'):
                task_id = task_id[9:]

            # Filter by verified task IDs if specified
            if verified_task_ids is not None and task_id not in verified_task_ids:
                summary['filtered'] += 1
                continue

            resolved = results.get(task_id, False)

            converted = convert_trajectory(traj_path, agent_name, task_id, resolved)

            if converted is None:
                summary['errors'] += 1
                continue

            # Track format
            fmt = converted.metadata.get('source_format', 'unknown')
            summary['formats'][fmt] = summary['formats'].get(fmt, 0) + 1

            # Save
            output_path = output_dir / f"{task_id}.json"
            with open(output_path, 'w') as f:
                json.dump(trajectory_to_dict(converted), f, indent=2)

            summary['converted'] += 1

    return summary


def check_trajectory_quality(output_dir: Path) -> dict:
    """Check if converted trajectories are useful or blob-style.

    Returns dict with:
    - is_blob: True if all trajectories have <= 1 message
    - reason: Explanation if is_blob
    - sample_messages: Message count from sample
    """
    traj_files = list(output_dir.glob('*.json'))
    if not traj_files:
        return {'is_blob': True, 'reason': 'No trajectories', 'sample_messages': 0}

    # Sample up to 5 trajectories
    sample_files = sorted(traj_files)[:5]
    message_counts = []

    for f in sample_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            message_counts.append(len(data.get('messages', [])))
        except Exception:
            message_counts.append(0)

    avg_messages = sum(message_counts) / len(message_counts) if message_counts else 0
    max_messages = max(message_counts) if message_counts else 0

    # Blob criteria: all sampled trajectories have <= 1 message
    if max_messages <= 1:
        return {
            'is_blob': True,
            'reason': 'Single message trajectories (patch-only or infrastructure logs)',
            'sample_messages': max_messages
        }

    return {'is_blob': False, 'reason': None, 'sample_messages': max_messages}


def cleanup_trajectories(output_dir: Path) -> dict:
    """Clean up converted trajectories by removing invalid ones.

    Removes:
    - Single-message trajectories (incomplete/failed runs with only server logs)

    Returns cleanup summary.
    """
    summary = {
        'deleted_single_message': 0,
        'total_checked': 0,
        'remaining': 0,
    }

    traj_files = list(output_dir.glob('*.json'))
    summary['total_checked'] = len(traj_files)

    for f in traj_files:
        if f.name.startswith('_'):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            num_messages = len(data.get('messages', []))

            # Delete single-message trajectories
            if num_messages <= 1:
                f.unlink()
                summary['deleted_single_message'] += 1
        except Exception:
            pass

    summary['remaining'] = summary['total_checked'] - summary['deleted_single_message']
    return summary


def check_agent_role_quality(output_dir: Path) -> dict:
    """Check if an agent's trajectories have proper assistant messages.

    Some agents (devlo, artemis) have non-standard formats where all content
    is in 'user' role messages with no proper 'assistant' turns.

    Returns:
    - has_assistant_messages: True if >50% of trajectories have assistant messages
    - no_assistant_count: Number of trajectories without assistant messages
    - total_checked: Total trajectories checked
    """
    traj_files = list(output_dir.glob('*.json'))
    traj_files = [f for f in traj_files if not f.name.startswith('_')]

    if not traj_files:
        return {'has_assistant_messages': True, 'no_assistant_count': 0, 'total_checked': 0}

    no_assistant_count = 0
    total_checked = len(traj_files)

    for f in traj_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            messages = data.get('messages', [])
            roles = [m.get('role') for m in messages if isinstance(m, dict)]

            if 'assistant' not in roles:
                no_assistant_count += 1
        except Exception:
            pass

    # Agent is invalid if >50% of trajectories have no assistant messages
    has_assistant_messages = no_assistant_count <= total_checked * 0.5

    return {
        'has_assistant_messages': has_assistant_messages,
        'no_assistant_count': no_assistant_count,
        'total_checked': total_checked,
    }


def convert_all_agents(
    experiments_dir: Path,
    output_base_dir: Path,
    verified_only: bool = False,
) -> dict:
    """Convert trajectories for ALL agents.

    Args:
        experiments_dir: Path to experiments directory
        output_base_dir: Base directory for output
        verified_only: If True, only convert trajectories for SWE-bench Verified tasks
    """
    verified_dir = experiments_dir / 'evaluation' / 'verified'

    # Agents to skip - these have non-meaningful trajectories:
    # - solver agents: only infrastructure logs (Request received, Solver finished)
    # - tools agents: problem statements misattributed as assistant messages
    # - SWE-Fixer agents: only 1-2 trajectories survive quality checks
    SKIP_AGENTS = {
        '20240920_solver',
        '20240924_solver',
        '20241028_solver',
        '20241022_tools_claude-3-5-haiku',
        '20241022_tools_claude-3-5-sonnet-updated',
        '20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128',
        '20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor',
    }

    # Load verified task IDs if filtering
    verified_task_ids = None
    if verified_only:
        try:
            from datasets import load_dataset
            ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            verified_task_ids = set(ex["instance_id"] for ex in ds)
            print(f"Filtering to {len(verified_task_ids)} SWE-bench Verified tasks")
        except Exception as e:
            print(f"Warning: Could not load SWE-bench Verified dataset: {e}")
            print("Proceeding without filtering")

    all_summary = {
        'agents_processed': 0,
        'total_trajectories': 0,
        'total_converted': 0,
        'total_filtered': 0,
        'total_errors': 0,
        'formats': {},
        'agents': {},
    }

    # Find all agent directories with trajs
    agent_dirs = sorted([
        d for d in verified_dir.iterdir()
        if d.is_dir() and (d / 'trajs').exists()
    ])

    print(f"Found {len(agent_dirs)} agent directories")

    for i, agent_dir in enumerate(agent_dirs):
        agent_name = agent_dir.name
        trajs_dir = agent_dir / 'trajs'

        # Skip agents with non-meaningful trajectories
        if agent_name in SKIP_AGENTS:
            print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name} - SKIPPED (non-meaningful trajectories)")
            continue

        # Check if has files or directories (nested formats have subdirs, not files)
        traj_items = list(trajs_dir.iterdir()) if trajs_dir.exists() else []
        traj_items = [f for f in traj_items if not f.name.startswith('.')]

        if not traj_items:
            continue

        print(f"\n[{i+1}/{len(agent_dirs)}] {agent_name} ({len(traj_items)} items)")

        output_dir = output_base_dir / agent_name

        try:
            summary = convert_agent_trajectories(agent_dir, output_dir, verified_task_ids)

            all_summary['agents_processed'] += 1
            all_summary['total_trajectories'] += summary['total']
            all_summary['total_converted'] += summary['converted']
            all_summary['total_filtered'] += summary.get('filtered', 0)
            all_summary['total_errors'] += summary['errors']

            # Merge format counts
            for fmt, count in summary['formats'].items():
                all_summary['formats'][fmt] = all_summary['formats'].get(fmt, 0) + count

            all_summary['agents'][agent_name] = {
                'converted': summary['converted'],
                'errors': summary['errors'],
                'format': list(summary['formats'].keys())[0] if summary['formats'] else 'unknown',
            }

            print(f"  -> Converted {summary['converted']}/{summary['total']}, errors: {summary['errors']}, format: {list(summary['formats'].keys())}")

            # Clean up single-message trajectories
            if summary['converted'] > 0:
                cleanup = cleanup_trajectories(output_dir)
                if cleanup['deleted_single_message'] > 0:
                    print(f"  -> Cleaned up {cleanup['deleted_single_message']} single-message trajectories")
                    summary['converted'] = cleanup['remaining']
                    all_summary['total_converted'] -= cleanup['deleted_single_message']

                # If all trajectories were cleaned up, remove the directory
                if cleanup['remaining'] == 0:
                    import shutil
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    all_summary['agents'][agent_name]['skipped_blob'] = True
                    all_summary['agents'][agent_name]['blob_reason'] = 'All trajectories were single-message'
                    print(f"  -> SKIPPED (blob-style): All trajectories were single-message")
                    continue

            # Check role quality - remove agents with mostly non-standard role assignments
            if summary['converted'] > 0 and output_dir.exists():
                role_quality = check_agent_role_quality(output_dir)
                if not role_quality['has_assistant_messages']:
                    import shutil
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    all_summary['agents'][agent_name]['removed'] = True
                    all_summary['agents'][agent_name]['removal_reason'] = (
                        f"Non-standard role format: {role_quality['no_assistant_count']}/{role_quality['total_checked']} "
                        "trajectories have no assistant messages"
                    )
                    all_summary['total_converted'] -= summary['converted']
                    print(f"  -> REMOVED: {role_quality['no_assistant_count']}/{role_quality['total_checked']} trajectories have no assistant messages")
                    continue

            # Check trajectory quality and skip blob-style agents
            if summary['converted'] > 0 and output_dir.exists():
                quality = check_trajectory_quality(output_dir)
                if quality['is_blob']:
                    import shutil
                    if output_dir.exists():
                        shutil.rmtree(output_dir)
                    all_summary['agents'][agent_name]['skipped_blob'] = True
                    all_summary['agents'][agent_name]['blob_reason'] = quality['reason']
                    all_summary['total_converted'] -= summary['converted']
                    print(f"  -> SKIPPED (blob-style): {quality['reason']}")

        except Exception as e:
            print(f"  -> Error: {e}")
            all_summary['total_errors'] += 1

    return all_summary


def preview_trajectory(traj: UnifiedTrajectory, max_len: int = 200):
    """Pretty-print a trajectory preview."""
    print(f"Task ID: {traj.task_id}")
    print(f"Agent: {traj.agent}")
    print(f"Resolved: {traj.resolved}")
    print(f"Format: {traj.metadata.get('source_format')}")
    print(f"Messages: {len(traj.messages)}")
    print("=" * 60)

    for i, msg in enumerate(traj.messages[:10]):
        content = msg.content
        if len(content) > max_len:
            content = content[:max_len] + f"... [{len(content) - max_len} more]"
        print(f"\n[{i+1}] {msg.role.upper()}")
        print(content)

    if len(traj.messages) > 10:
        print(f"\n... and {len(traj.messages) - 10} more messages")


def main():
    parser = argparse.ArgumentParser(description='Convert SWE-bench trajectories to unified format')
    parser.add_argument('--input', type=str, help='Single trajectory file')
    parser.add_argument('--agent', type=str, help='Agent name')
    parser.add_argument('--all', action='store_true', help='Convert ALL agents')
    parser.add_argument('--output', type=str, help='Output path (single file)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--preview', action='store_true', help='Preview without saving')
    parser.add_argument('--verified_only', action='store_true',
                        help='Only convert trajectories for SWE-bench Verified tasks (500 tasks)')

    args = parser.parse_args()
    experiments_dir = Path(__file__).resolve().parents[1] / 'experiments'

    if args.all:
        output_dir = Path(args.output_dir) if args.output_dir else Path('chris_output/unified_trajs')

        print("Converting ALL agent trajectories...")
        if args.verified_only:
            print("Filtering to SWE-bench Verified tasks only")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        summary = convert_all_agents(experiments_dir, output_dir, verified_only=args.verified_only)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Agents processed: {summary['agents_processed']}")
        print(f"Total trajectories: {summary['total_trajectories']}")
        print(f"Total converted: {summary['total_converted']}")
        if summary.get('total_filtered', 0) > 0:
            print(f"Total filtered (non-Verified): {summary['total_filtered']}")
        print(f"Total errors: {summary['total_errors']}")
        print(f"Formats: {summary['formats']}")
        print(f"Output: {output_dir}")

        # Save summary
        summary_path = output_dir / '_conversion_summary.json'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return

        agent_name = args.agent or input_path.parent.parent.name
        converted = convert_trajectory(input_path, agent_name)

        if converted is None:
            print("Failed to convert trajectory")
            return

        if args.preview or not args.output:
            preview_trajectory(converted)
        else:
            with open(args.output, 'w') as f:
                json.dump(trajectory_to_dict(converted), f, indent=2)
            print(f"Saved to {args.output}")

    elif args.agent:
        agent_dir = experiments_dir / 'evaluation' / 'verified' / args.agent
        if not agent_dir.exists():
            print(f"Error: Agent not found: {agent_dir}")
            return

        output_dir = Path(args.output_dir) if args.output_dir else Path(f'chris_output/unified_trajs/{args.agent}')

        print(f"Converting {args.agent}...")
        summary = convert_agent_trajectories(agent_dir, output_dir)

        print(f"\nConverted {summary['converted']}/{summary['total']}")
        print(f"Errors: {summary['errors']}")
        print(f"Formats: {summary['formats']}")
        print(f"Output: {output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
