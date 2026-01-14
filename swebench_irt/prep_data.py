"""Script for processing the original MMLU JSON data on model-answer pairs. In the orgiginal data,
each JSON file corresponds to a given model, and each entry in the JSON file
corresponds to a question-answer pair."""
# %%

import json
from pathlib import Path
import jsonlines

INPUT_DIR = Path("raw_data/mmlu_data")
OUTPUT_DIR = Path("clean_data/mmlu_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_single_model(json_path: Path) -> dict[str, int]:
    """
    Process a single model's JSON data and convert to responses to binary correctness format.
    Args: json_path: Path to the JSON file containing model predictions
    Returns: Dictionary mapping question IDs to binary correctness (1 for correct, 0 for incorrect)
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    subject_id = [r["question_id"] for r in data]
    correct_answer = [r["answer"] for r in data]
    model_answer = [r["pred"] for r in data]
    binary_correct = [int(x == y) for x,y in zip(correct_answer, model_answer)]
    response_pairs = dict(zip(subject_id, binary_correct))

    return response_pairs

# %%
def extract_model_name(file_path: Path) -> str:
    """Extract model name from file path by removing prefix and suffix."""
    return file_path.name.removeprefix("model_outputs_").removesuffix(".json")

def main() -> None:
    """Main function to process all model data and write to output file."""

    output_file = "model_response_correctness.jsonl"
    output_file_path = OUTPUT_DIR / output_file
    
    with jsonlines.open(output_file_path, "w") as writer:
        for json_file in INPUT_DIR.glob("*.json"):
            if json_file.name == "model_outputs_DeepSeek-Coder-V2_5shots.json":
                # This file has some formatting issues
                continue

            model_name = extract_model_name(json_file)
            responses = process_single_model(json_file)
            
            writer.write({
                "subject_id": model_name,
                "responses": responses
            })

if __name__ == "__main__":
    main()
