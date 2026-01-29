#!/usr/bin/env python3
"""
Test the SWE-bench Pro API to understand structure.
"""

import json
import httpx
from pathlib import Path

def load_api_key() -> str:
    """Load API key from config file."""
    config_path = Path.home() / ".config" / "docent" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["api_key"]

def test_api():
    api_key = load_api_key()
    collection_id = "032fb63d-4992-4bfc-911d-3b7dafcb931f"

    with httpx.Client(
        base_url="https://docent.transluce.org/api",
        headers={"X-API-Key": api_key},
        timeout=30.0
    ) as client:

        # Test 1: List collections
        print("=== Test 1: List Collections ===")
        try:
            r = client.get("/collections")
            print(f"Status: {r.status_code}")
            collections = r.json()
            print(f"Found {len(collections)} collections")
            for c in collections[:3]:
                print(f"  - {c.get('id')}: {c.get('name')}")
        except Exception as e:
            print(f"Error: {e}")

        # Test 2: Get collection details
        print(f"\n=== Test 2: Get Collection {collection_id} ===")
        try:
            r = client.get(f"/collections/{collection_id}")
            print(f"Status: {r.status_code}")
            data = r.json()
            print(json.dumps(data, indent=2)[:500])
        except Exception as e:
            print(f"Error: {e}")

        # Test 3: List agent runs in collection
        print(f"\n=== Test 3: List Agent Runs ===")
        try:
            r = client.get(f"/collections/{collection_id}/agent_runs")
            print(f"Status: {r.status_code}")
            runs = r.json()
            print(f"Found {len(runs)} agent runs")

            if runs:
                print("\nFirst agent run:")
                print(json.dumps(runs[0], indent=2)[:800])

                # Try to fetch the full trajectory for the first run
                first_id = runs[0]['id']
                print(f"\n=== Test 4: Get Agent Run {first_id} ===")
                r = client.get(f"/collections/{collection_id}/agent_runs/{first_id}")
                print(f"Status: {r.status_code}")
                run_data = r.json()

                # Show keys and sizes
                print("\nAgent run structure:")
                for key, value in run_data.items():
                    if isinstance(value, (list, dict, str)):
                        size = len(str(value))
                        print(f"  {key}: {type(value).__name__} ({size} chars)")
                    else:
                        print(f"  {key}: {type(value).__name__} = {value}")

                # Save sample for inspection
                with open("swebench_pro_sample_trajectory.json", "w") as f:
                    json.dump(run_data, f, indent=2)
                print("\nSaved sample trajectory to swebench_pro_sample_trajectory.json")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_api()
