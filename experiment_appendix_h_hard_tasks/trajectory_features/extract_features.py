"""Extract behavioral features from agent trajectories using LLM judge.

Supports parallel API calls for faster extraction.

Usage:
    # Exploration run (100 trajectories, parallel)
    python -m experiment_appendix_h_hard_tasks.trajectory_features.extract_features --num_trajectories 100 --parallel 10

    # Single trajectory
    python -m experiment_appendix_h_hard_tasks.trajectory_features.extract_features \
        --agent 20250807_openhands_gpt5 --task_id django__django-11179

    # Full run with specific model
    python -m experiment_appendix_h_hard_tasks.trajectory_features.extract_features \
        --model claude-opus-4-5-20251101 --parallel 20
"""

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import pandas as pd
from anthropic import AsyncAnthropic

from .config import AGENT_NAMES, FEATURE_NAMES, SELECTED_AGENTS
from .prompts import TRAJECTORY_FEATURE_PROMPT, format_trajectory_for_prompt


@dataclass
class UsageStats:
    """Track API usage statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_errors: int = 0
    model: str = "claude-sonnet-4-5-20250929"

    def add_call(self, input_tokens: int, output_tokens: int):
        """Record a successful API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

    def add_error(self):
        """Record a failed API call."""
        self.total_errors += 1

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in USD based on model."""
        if "opus" in self.model.lower():
            input_rate = 15  # $15/1M input
            output_rate = 75  # $75/1M output
        else:
            input_rate = 3  # $3/1M input
            output_rate = 15  # $15/1M output

        input_cost = self.total_input_tokens * input_rate / 1_000_000
        output_cost = self.total_output_tokens * output_rate / 1_000_000
        return input_cost + output_cost

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"Calls: {self.total_calls}, Errors: {self.total_errors}, "
            f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out, "
            f"Cost: ${self.estimated_cost:.2f}"
        )


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from response text."""
    text = text.strip()

    # Try to extract from markdown code blocks
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    # Find JSON object boundaries
    if not text.startswith("{"):
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break

    return json.loads(text)


class TrajectoryFeatureExtractor:
    """Extract behavioral features from trajectories using Claude."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        trajectory_dir: str = "experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs",
        max_retries: int = 3,
    ):
        """Initialize the extractor.

        Args:
            model: Claude model to use.
            trajectory_dir: Directory containing unified trajectories.
            max_retries: Maximum retries on API errors.
        """
        self.model = model
        self.trajectory_dir = Path(trajectory_dir)
        self.max_retries = max_retries
        self.sync_client = anthropic.Anthropic()
        self.async_client = AsyncAnthropic()
        self.usage = UsageStats(model=model)
        self._lock = asyncio.Lock()

    def load_trajectory(self, agent: str, task_id: str) -> dict:
        """Load a trajectory file."""
        path = self.trajectory_dir / agent / f"{task_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trajectory not found: {path}")

        with open(path) as f:
            return json.load(f)

    def _build_prompt(
        self, agent: str, task_id: str, trajectory: dict, max_messages: int = 100
    ) -> str:
        """Build the feature extraction prompt."""
        trajectory_content = format_trajectory_for_prompt(
            trajectory, max_messages=max_messages
        )

        return TRAJECTORY_FEATURE_PROMPT.format(
            agent_name=agent,
            task_id=task_id,
            resolved="SUCCESS" if trajectory.get("resolved", False) else "FAILURE",
            trajectory_content=trajectory_content,
        )

    def extract_features_sync(
        self,
        agent: str,
        task_id: str,
        max_messages: int = 100,
    ) -> Dict[str, Any]:
        """Extract features from a single trajectory (synchronous)."""
        trajectory = self.load_trajectory(agent, task_id)
        prompt = self._build_prompt(agent, task_id, trajectory, max_messages)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.sync_client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                self.usage.add_call(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

                result = parse_json_response(response.content[0].text)
                result["agent"] = agent
                result["task_id"] = task_id
                result["resolved"] = trajectory.get("resolved", False)
                result["trajectory_length"] = len(trajectory.get("messages", []))
                result["model"] = self.model

                return result

            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (2**attempt) * 10
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

            except anthropic.APIError as e:
                last_error = e
                self.usage.add_error()
                wait_time = 2**attempt
                print(f"API error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

            except json.JSONDecodeError as e:
                last_error = e
                self.usage.add_error()
                break

        raise last_error or ValueError("Failed to extract features")

    async def extract_features_async(
        self,
        agent: str,
        task_id: str,
        max_messages: int = 100,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Extract features from a single trajectory (async)."""
        if semaphore:
            async with semaphore:
                return await self._extract_features_async_impl(
                    agent, task_id, max_messages
                )
        return await self._extract_features_async_impl(agent, task_id, max_messages)

    async def _extract_features_async_impl(
        self,
        agent: str,
        task_id: str,
        max_messages: int = 100,
    ) -> Dict[str, Any]:
        """Internal async implementation."""
        trajectory = self.load_trajectory(agent, task_id)
        prompt = self._build_prompt(agent, task_id, trajectory, max_messages)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )

                async with self._lock:
                    self.usage.add_call(
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    )

                result = parse_json_response(response.content[0].text)
                result["agent"] = agent
                result["task_id"] = task_id
                result["resolved"] = trajectory.get("resolved", False)
                result["trajectory_length"] = len(trajectory.get("messages", []))
                result["model"] = self.model

                return result

            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (2**attempt) * 10
                print(f"Rate limited ({agent}/{task_id}), waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

            except anthropic.APIError as e:
                last_error = e
                async with self._lock:
                    self.usage.add_error()
                wait_time = 2**attempt
                print(f"API error ({agent}/{task_id}): {e}, retrying...")
                await asyncio.sleep(wait_time)

            except json.JSONDecodeError as e:
                last_error = e
                async with self._lock:
                    self.usage.add_error()
                break

        raise last_error or ValueError(f"Failed: {agent}/{task_id}")

    def get_task_ids(self, agent: str) -> List[str]:
        """Get all task IDs available for an agent."""
        agent_dir = self.trajectory_dir / agent
        if not agent_dir.exists():
            raise FileNotFoundError(f"Agent directory not found: {agent_dir}")
        return [f.stem for f in agent_dir.glob("*.json")]

    async def extract_batch_async(
        self,
        agent_task_pairs: List[Tuple[str, str]],
        output_path: Optional[Path] = None,
        resume: bool = True,
        parallel: int = 10,
        save_interval: int = 20,
    ) -> pd.DataFrame:
        """Extract features for multiple trajectories in parallel.

        Args:
            agent_task_pairs: List of (agent, task_id) tuples.
            output_path: Path to save results incrementally.
            resume: If True, skip already-processed pairs.
            parallel: Number of parallel API calls.
            save_interval: Save results every N completions.

        Returns:
            DataFrame with extracted features.
        """
        results = []
        existing_pairs = set()

        # Load existing results if resuming
        if resume and output_path and output_path.exists():
            existing_df = pd.read_csv(output_path)
            existing_pairs = set(zip(existing_df["agent"], existing_df["task_id"]))
            results = existing_df.to_dict("records")
            print(f"Loaded {len(existing_pairs)} existing results")

        # Filter to remaining pairs
        pairs_to_process = [
            (a, t) for a, t in agent_task_pairs if (a, t) not in existing_pairs
        ]
        print(
            f"Processing {len(pairs_to_process)} trajectories with {parallel} parallel calls..."
        )

        semaphore = asyncio.Semaphore(parallel)
        completed = 0
        start_time = time.time()

        async def process_one(agent: str, task_id: str) -> Optional[Dict]:
            nonlocal completed
            try:
                result = await self.extract_features_async(
                    agent, task_id, semaphore=semaphore
                )
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"[{completed}/{len(pairs_to_process)}] "
                    f"{agent[:30]}... / {task_id} OK ({rate:.1f}/s)"
                )
                return result
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(pairs_to_process)}] {agent} / {task_id} ERROR: {e}")
                return None

        # Process in batches and save periodically
        batch_size = save_interval
        for batch_start in range(0, len(pairs_to_process), batch_size):
            batch = pairs_to_process[batch_start : batch_start + batch_size]

            tasks = [process_one(agent, task_id) for agent, task_id in batch]
            batch_results = await asyncio.gather(*tasks)

            # Collect successful results
            for r in batch_results:
                if r is not None:
                    results.append(r)

            # Save incrementally
            if output_path and results:
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                print(f"  Saved {len(results)} results to {output_path}")

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s ({len(pairs_to_process)/elapsed:.1f}/s)")
        print(f"Usage: {self.usage.summary()}")

        return pd.DataFrame(results)

    def extract_batch_sync(
        self,
        agent_task_pairs: List[Tuple[str, str]],
        output_path: Optional[Path] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        """Extract features sequentially (fallback for debugging)."""
        results = []
        existing_pairs = set()

        if resume and output_path and output_path.exists():
            existing_df = pd.read_csv(output_path)
            existing_pairs = set(zip(existing_df["agent"], existing_df["task_id"]))
            results = existing_df.to_dict("records")
            print(f"Loaded {len(existing_pairs)} existing results")

        pairs_to_process = [
            (a, t) for a, t in agent_task_pairs if (a, t) not in existing_pairs
        ]
        print(f"Processing {len(pairs_to_process)} trajectories (sequential)...")

        for i, (agent, task_id) in enumerate(pairs_to_process):
            print(f"[{i+1}/{len(pairs_to_process)}] {agent} / {task_id}...", end=" ")
            try:
                result = self.extract_features_sync(agent, task_id)
                results.append(result)
                print("OK")

                if output_path:
                    pd.DataFrame(results).to_csv(output_path, index=False)
            except Exception as e:
                print(f"ERROR: {e}")

        print(f"\nUsage: {self.usage.summary()}")
        return pd.DataFrame(results)


def sample_trajectories(
    n_per_agent: int = 5,
    agents: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Sample trajectories evenly across agents."""
    if agents is None:
        agents = AGENT_NAMES

    trajectory_dir = Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs")
    pairs = []

    for agent in agents:
        agent_dir = trajectory_dir / agent
        if not agent_dir.exists():
            print(f"Warning: Agent directory not found: {agent}")
            continue

        task_ids = sorted([f.stem for f in agent_dir.glob("*.json")])
        if len(task_ids) <= n_per_agent:
            sampled = task_ids
        else:
            step = len(task_ids) // n_per_agent
            sampled = [task_ids[i * step] for i in range(n_per_agent)]

        pairs.extend([(agent, task_id) for task_id in sampled])

    return pairs


def sample_by_task(
    n_tasks: int = 100,
    agents_per_task: int = 6,
    agents: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Sample trajectories task-by-task with fixed agents per task.

    Args:
        n_tasks: Number of tasks to sample
        agents_per_task: Number of agents to include per task
        agents: List of agents to use (default: AGENT_NAMES)

    Returns:
        List of (agent, task_id) pairs
    """
    if agents is None:
        agents = AGENT_NAMES

    trajectory_dir = Path("experiment_appendix_h_hard_tasks/trajectory_data/unified_trajs")

    # Build task -> available agents mapping
    task_agents: Dict[str, List[str]] = {}
    for agent in agents:
        agent_dir = trajectory_dir / agent
        if not agent_dir.exists():
            continue

        for task_file in agent_dir.glob("*.json"):
            task_id = task_file.stem
            if task_id not in task_agents:
                task_agents[task_id] = []
            task_agents[task_id].append(agent)

    # Filter to tasks with enough agents
    valid_tasks = [t for t, a in task_agents.items() if len(a) >= agents_per_task]
    print(f"Found {len(valid_tasks)} tasks with >= {agents_per_task} agents")

    # Sample tasks
    import random
    random.seed(42)  # Reproducible
    sampled_tasks = sorted(random.sample(valid_tasks, min(n_tasks, len(valid_tasks))))

    # For each task, select agents spanning the ability spectrum
    # Sort agents by their IRT ability and pick evenly spaced ones
    agent_abilities = {a.name: a.theta for a in SELECTED_AGENTS}

    pairs = []
    for task_id in sampled_tasks:
        available = task_agents[task_id]
        # Sort by ability
        available_with_ability = [(a, agent_abilities.get(a, 0)) for a in available]
        available_with_ability.sort(key=lambda x: x[1])

        # Pick evenly spaced agents
        if len(available_with_ability) <= agents_per_task:
            selected = [a for a, _ in available_with_ability]
        else:
            step = len(available_with_ability) / agents_per_task
            indices = [int(i * step) for i in range(agents_per_task)]
            selected = [available_with_ability[i][0] for i in indices]

        pairs.extend([(agent, task_id) for agent in selected])

    return pairs


async def async_main(args):
    """Async entry point."""
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = TrajectoryFeatureExtractor(model=args.model)
    print(f"Using model: {args.model}")

    if args.agent and args.task_id:
        # Single trajectory
        print(f"Extracting features for {args.agent} / {args.task_id}")
        result = extractor.extract_features_sync(args.agent, args.task_id)
        print(json.dumps(result, indent=2))
        print(f"\nUsage: {extractor.usage.summary()}")

    elif args.n_tasks:
        # Task-by-task sampling with fixed agents per task
        pairs = sample_by_task(
            n_tasks=args.n_tasks,
            agents_per_task=args.agents_per_task,
        )
        print(f"Sampled {len(pairs)} trajectories ({args.n_tasks} tasks × {args.agents_per_task} agents)")

        df = await extractor.extract_batch_async(
            pairs,
            output_path=output_path,
            resume=not args.no_resume,
            parallel=args.parallel,
        )
        print(f"\nSaved {len(df)} results to {output_path}")

    elif args.num_trajectories:
        # Sample trajectories for exploration (legacy mode)
        n_per_agent = max(1, args.num_trajectories // len(AGENT_NAMES))
        pairs = sample_trajectories(n_per_agent=n_per_agent)[: args.num_trajectories]
        print(f"Sampled {len(pairs)} trajectories from {len(AGENT_NAMES)} agents")

        df = await extractor.extract_batch_async(
            pairs,
            output_path=output_path,
            resume=not args.no_resume,
            parallel=args.parallel,
        )
        print(f"\nSaved {len(df)} results to {output_path}")

    else:
        # Full extraction for all selected agents
        pairs = []
        for agent in AGENT_NAMES:
            task_ids = extractor.get_task_ids(agent)
            pairs.extend([(agent, tid) for tid in task_ids])

        print(f"Full extraction: {len(pairs)} trajectories")

        df = await extractor.extract_batch_async(
            pairs,
            output_path=output_path,
            resume=not args.no_resume,
            parallel=args.parallel,
        )
        print(f"\nSaved {len(df)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from agent trajectories"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Extract for specific agent only",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
        help="Extract for specific task only",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Number of trajectories to sample (for exploration runs)",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=None,
        help="Number of tasks to sample (task-by-task mode)",
    )
    parser.add_argument(
        "--agents_per_task",
        type=int,
        default=6,
        help="Number of agents per task (task-by-task mode)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel API calls",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="chris_output/trajectory_features/raw_features.csv",
        help="Output path for results",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from existing results",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
