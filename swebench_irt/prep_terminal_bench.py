import argparse
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from prep_utils import matrix_responses, print_matrix_stats, resolve_path, write_jsonl_records

LEADERBOARD_URL = "https://www.tbench.ai/leaderboard/terminal-bench/2.0"


def _normalize_cell_text(s: str) -> str:
    return " ".join(s.replace("\u00a0", " ").split()).strip()


def get_leaderboard_column_indices(page) -> dict[str, int]:
    header_candidates = page.locator('tr[data-slot="table-column-headers"], thead tr')
    col_idx: dict[str, int] = {}

    if header_candidates.count() > 0:
        header_row = header_candidates.first
        header_cells = header_row.locator("th, td").all()
        header_texts = [_normalize_cell_text(c.inner_text()).lower() for c in header_cells]

        want = {
            "rank": "rank",
            "agent": "agent",
            "model": "model",
            "date": "date",
            "agent org": "agent_org",
            "model org": "model_org",
            "accuracy": "accuracy",
        }
        for i, t in enumerate(header_texts):
            key = want.get(t)
            if key:
                col_idx[key] = i

    if "agent" not in col_idx or "model" not in col_idx:
        first_row = page.locator('tr[data-slot="table-row"][data-state]').first
        if first_row.count() > 0:
            n = first_row.locator("th, td").count()
            base = 1 if n >= 8 else 0
            col_idx = {
                "rank": 0 + base,
                "agent": 1 + base,
                "model": 2 + base,
                "date": 3 + base,
                "agent_org": 4 + base,
                "model_org": 5 + base,
                "accuracy": 6 + base,
            }

    return col_idx


def parse_leaderboard_row(row, col_idx: dict[str, int]) -> dict | None:
    cells = row.locator("th, td")

    def _get(key: str) -> str:
        i = col_idx.get(key)
        if i is None:
            return ""
        if i < 0 or i >= cells.count():
            return ""
        return _normalize_cell_text(cells.nth(i).inner_text())

    rank = _get("rank")
    agent_name = _get("agent")
    model_name = _get("model")
    date = _get("date")
    agent_org = _get("agent_org")
    model_org = _get("model_org")
    accuracy = _get("accuracy")

    if agent_name == "" or model_name == "":
        return None

    return {
        "rank": rank,
        "agent": agent_name,
        "model": model_name,
        "date": date,
        "agent_org": agent_org,
        "model_org": model_org,
        "accuracy": accuracy,
    }


def parse_task_results(html_content: str) -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    soup = BeautifulSoup(html_content, "html.parser")
    rows = soup.find_all("tr", {"data-slot": "table-row"})

    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 4:
            task_span = cells[0].find("span", class_="font-normal")
            if task_span:
                task_name = task_span.get_text().strip()
                trials_p = cells[1].find("p", class_="text-right")
                successes_p = cells[2].find("p", class_="text-right")

                if trials_p and successes_p:
                    try:
                        trials = int(trials_p.get_text().strip())
                        successes = int(successes_p.get_text().strip())
                        results[task_name] = (successes, trials)
                    except ValueError:
                        continue

    return results


def results_to_binary(results: dict[str, tuple[int, int]], threshold: str = "majority") -> dict[str, int]:
    binary: dict[str, int] = {}
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
    agent_clean = agent_name.lower().replace(" ", "_").replace("-", "_")
    model_clean = model_name.lower().replace("@", "_").replace(".", "_").replace("-", "_").replace(" ", "_")
    return f"{agent_clean}_{model_clean}"


def scrape_all_agents(headless: bool = True, delay: float = 1.0, limit: int | None = None):
    all_results = []

    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page()

        print("Loading leaderboard...")
        page.goto(LEADERBOARD_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2000)

        col_idx = get_leaderboard_column_indices(page)
        rows = page.locator('tr[data-slot="table-row"][data-state]').all()
        num_agents = len(rows)
        if limit is not None:
            num_agents = min(num_agents, max(0, limit))
        print(f"Found {num_agents} agents")

        for i in range(num_agents):
            page.goto(LEADERBOARD_URL, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1000)

            rows = page.locator('tr[data-slot="table-row"][data-state]').all()
            row = rows[i]

            parsed = parse_leaderboard_row(row, col_idx)
            if not parsed:
                row_text = _normalize_cell_text(row.inner_text())
                print(f"  [{i+1}/{num_agents}] Skipping malformed row: {row_text[:80]}")
                continue

            rank = parsed["rank"]
            agent_name = parsed["agent"]
            model_name = parsed["model"]
            date = parsed["date"]
            agent_org = parsed["agent_org"]
            model_org = parsed["model_org"]
            accuracy = parsed["accuracy"]
            print(f"[{i+1}/{num_agents}] {agent_name} + {model_name} ({agent_org}/{model_org})")
            row.click()
            page.wait_for_timeout(2000)
            detail_url = page.url
            html = page.content()
            task_results = parse_task_results(html)

            if len(task_results) > 2:
                print(f"  -> {len(task_results)} tasks, URL: ...{detail_url[-60:]}")
                all_results.append(
                    {
                        "rank": rank,
                        "agent": agent_name,
                        "model": model_name,
                        "agent_org": agent_org,
                        "model_org": model_org,
                        "date": date,
                        "accuracy": accuracy,
                        "detail_url": detail_url,
                        "task_results": task_results,
                    }
                )
            else:
                print(f"  -> No task data found")

            if delay > 0:
                time.sleep(delay)

        browser.close()

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Terminal Bench and convert to IRT format")
    parser.add_argument("--output", type=str, default="out/chris_irt/terminal_bench.jsonl")
    parser.add_argument("--threshold", type=str, default="majority", choices=["any", "majority", "all"])
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--irt-only", action="store_true")
    args = parser.parse_args()

    output_path = resolve_path(args.output)

    if args.dry_run:
        print("Dry run mode - would scrape all 85 agents from Terminal Bench leaderboard")
        print(f"Output: {output_path}")
        print(f"Threshold: {args.threshold}")
        return

    print("Starting scraper...")
    all_results = scrape_all_agents(
        headless=not args.no_headless,
        delay=args.delay,
        limit=args.limit,
    )

    print(f"\nSuccessfully scraped {len(all_results)} agents")
    all_tasks = set()
    for result in all_results:
        all_tasks.update(result["task_results"].keys())

    print(f"Total unique tasks: {len(all_tasks)}")
    all_tasks_sorted = sorted(all_tasks)
    irt_records = []
    summary: list[tuple[str, int, int]] = []
    for result in all_results:
        subject_id = create_subject_id(result["agent"], result["model"])
        binary_responses = results_to_binary(result["task_results"], args.threshold)
        responses = matrix_responses(binary_responses, all_tasks_sorted, no_complete_matrix=False)
        summary.append((subject_id, len(responses), sum(responses.values())))

        record = {
            "subject_id": subject_id,
            "responses": responses,
        }
        if not args.irt_only:
            record.update(
                {
                    "rank": result.get("rank", ""),
                    "agent": result.get("agent", ""),
                    "model": result.get("model", ""),
                    "agent_org": result.get("agent_org", ""),
                    "model_org": result.get("model_org", ""),
                    "date": result.get("date", ""),
                    "accuracy": result.get("accuracy", ""),
                    "detail_url": result.get("detail_url", ""),
                }
            )

        irt_records.append(record)

    write_jsonl_records(output_path, irt_records)
    print_matrix_stats(
        records=irt_records,
        all_items=all_tasks,
        no_complete_matrix=False,
        subject_label="agents",
        output_path=output_path,
        summary=summary,
    )

    print(f"Saved {len(irt_records)} agents x {len(all_tasks)} tasks to {output_path}")

if __name__ == "__main__":
    main()
