#!/usr/bin/env python3
"""
Scrape Terminal Bench leaderboard data and convert to IRT JSONL format.

This script uses Playwright to:
1. Navigate to the leaderboard
2. Click on each agent row to get the correct detail URL
3. Extract per-task results from each detail page
4. Convert to JSONL format compatible with swebench_irt/train.py
"""

import json
import time
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def parse_task_results(html_content: str) -> dict[str, tuple[int, int]]:
    """
    Parse per-task results from an agent detail page.

    Returns dict mapping task_id -> (successes, trials)
    """
    results = {}
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all table rows with data-slot="table-row"
    rows = soup.find_all('tr', {'data-slot': 'table-row'})

    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 4:
            # First cell: task name in <span class="font-normal">
            task_span = cells[0].find('span', class_='font-normal')
            if task_span:
                task_name = task_span.get_text().strip()

                # Second cell: trials, Third cell: successes
                trials_p = cells[1].find('p', class_='text-right')
                successes_p = cells[2].find('p', class_='text-right')

                if trials_p and successes_p:
                    try:
                        trials = int(trials_p.get_text().strip())
                        successes = int(successes_p.get_text().strip())
                        results[task_name] = (successes, trials)
                    except ValueError:
                        pass  # Skip if can't parse numbers

    return results


def results_to_binary(results: dict[str, tuple[int, int]], threshold: str = "any") -> dict[str, int]:
    """
    Convert (successes, trials) to binary 0/1.

    threshold:
        - "any": 1 if any trial succeeded
        - "majority": 1 if majority (>=50%) succeeded
        - "all": 1 if all trials succeeded
    """
    binary = {}
    for task_id, (successes, trials) in results.items():
        if threshold == "any":
            binary[task_id] = 1 if successes > 0 else 0
        elif threshold == "majority":
            binary[task_id] = 1 if successes >= trials / 2 else 0
        elif threshold == "all":
            binary[task_id] = 1 if successes == trials else 0
        else:
            raise ValueError(f"Unknown threshold: {threshold}")
    return binary


def create_subject_id(agent_name: str, model_name: str) -> str:
    """Create a unique subject ID from agent name and model."""
    # Clean up and combine
    agent_clean = agent_name.lower().replace(" ", "_").replace("-", "_")
    model_clean = model_name.lower().replace("@", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
    return f"{agent_clean}_{model_clean}"


def scrape_all_agents(headless: bool = True, delay: float = 1.0):
    """
    Scrape all agents from the Terminal Bench leaderboard.

    Returns list of dicts with agent info and task results.
    """
    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()

        # Go to leaderboard
        print("Loading leaderboard...")
        page.goto('https://www.tbench.ai/leaderboard/terminal-bench/2.0',
                  wait_until='networkidle', timeout=60000)
        page.wait_for_timeout(2000)

        # Get all data rows
        rows = page.locator('tr[data-slot="table-row"][data-state]').all()
        num_agents = len(rows)
        print(f"Found {num_agents} agents")

        for i in range(num_agents):
            # Need to re-query rows after each navigation
            page.goto('https://www.tbench.ai/leaderboard/terminal-bench/2.0',
                      wait_until='networkidle', timeout=60000)
            page.wait_for_timeout(1000)

            rows = page.locator('tr[data-slot="table-row"][data-state]').all()
            row = rows[i]

            # Get row info
            row_text = row.inner_text()
            parts = row_text.split('\t')

            # Parse row: Rank, Agent, Model, Date, Agent Org, Model Org, Accuracy
            if len(parts) >= 7:
                rank = parts[0].strip()
                agent_name = parts[1].strip()
                model_name = parts[2].strip()
                date = parts[3].strip()
                agent_org = parts[4].strip()
                model_org = parts[5].strip()
                accuracy = parts[6].strip()
            else:
                print(f"  [{i+1}/{num_agents}] Skipping malformed row: {row_text[:50]}")
                continue

            print(f"[{i+1}/{num_agents}] {agent_name} + {model_name} ({agent_org}/{model_org})")

            # Click to navigate to detail page
            row.click()
            page.wait_for_timeout(2000)

            # Get the detail URL
            detail_url = page.url

            # Get page content
            html = page.content()

            # Parse task results
            task_results = parse_task_results(html)

            if len(task_results) > 2:  # More than just header rows
                print(f"  -> {len(task_results)} tasks, URL: ...{detail_url[-60:]}")

                all_results.append({
                    'rank': rank,
                    'agent': agent_name,
                    'model': model_name,
                    'agent_org': agent_org,
                    'model_org': model_org,
                    'date': date,
                    'accuracy': accuracy,
                    'detail_url': detail_url,
                    'task_results': task_results
                })
            else:
                print(f"  -> No task data found")

            if delay > 0:
                time.sleep(delay)

        browser.close()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Scrape Terminal Bench and convert to IRT format")
    parser.add_argument("--output", type=str, default="clean_data/terminal_bench/terminal_bench_2.0.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--threshold", type=str, default="any", choices=["any", "majority", "all"],
                       help="How to convert multi-trial results to binary")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between requests in seconds")
    parser.add_argument("--no-headless", action="store_true",
                       help="Run browser in visible mode (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be done without making requests")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run mode - would scrape all 85 agents from Terminal Bench leaderboard")
        print(f"Output: {output_path}")
        print(f"Threshold: {args.threshold}")
        return

    # Scrape all agents
    print("Starting scraper...")
    all_results = scrape_all_agents(headless=not args.no_headless, delay=args.delay)

    print(f"\nSuccessfully scraped {len(all_results)} agents")

    # Collect all tasks
    all_tasks = set()
    for result in all_results:
        all_tasks.update(result['task_results'].keys())

    print(f"Total unique tasks: {len(all_tasks)}")

    # Convert to IRT JSONL format
    irt_records = []
    for result in all_results:
        subject_id = create_subject_id(result['agent'], result['model'])
        binary_responses = results_to_binary(result['task_results'], args.threshold)

        # Ensure all tasks are present (missing = 0)
        for task in all_tasks:
            if task not in binary_responses:
                binary_responses[task] = 0

        irt_records.append({
            "subject_id": subject_id,
            "responses": binary_responses
        })

    # Write JSONL
    with open(output_path, "w") as f:
        for record in irt_records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(irt_records)} agents x {len(all_tasks)} tasks to {output_path}")

    # Also save full results with metadata
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "agents": len(all_results),
            "tasks": len(all_tasks),
            "task_list": sorted(all_tasks),
            "threshold": args.threshold,
            "results": [
                {
                    **{k: v for k, v in r.items() if k != 'task_results'},
                    "task_results": {k: f"{v[0]}/{v[1]}" for k, v in r['task_results'].items()}
                }
                for r in all_results
            ]
        }, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
