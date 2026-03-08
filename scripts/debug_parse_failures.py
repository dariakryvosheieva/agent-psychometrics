"""Debug parse failures in auditor agent logs."""
from inspect_ai.log import read_eval_log
from pathlib import Path

FAILED_IDS = [
    "numpy__numpy-1fcda82", "numpy__numpy-330057f", "numpy__numpy-567b57d",
    "numpy__numpy-68eead8", "numpy__numpy-7ff7ec7", "numpy__numpy-8dd6761",
    "numpy__numpy-cb0d7cd", "numpy__numpy-cb461ba", "abetlen__llama-cpp-python-2bc1d97",
]

log_dir = Path("chris_output/auditor_features/gso_v4")
for log_path in sorted(log_dir.rglob("*.eval")):
    log = read_eval_log(str(log_path))
    for sample in log.samples or []:
        sid = str(sample.id)
        if any(fid in sid for fid in FAILED_IDS[:3]):
            print(f"\n=== {sid} ===")
            print(f"Status: {sample.output.status if sample.output else 'no output'}")
            print(f"Stop reason: {getattr(sample.output, 'stop_reason', 'N/A')}")
            print(f"Num messages: {len(sample.messages)}")
            comp = (sample.output.completion or "")[:500] if sample.output else ""
            print(f"Completion: {comp}")
