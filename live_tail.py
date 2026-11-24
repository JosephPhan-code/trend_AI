import time
from pathlib import Path

log_path = Path(r"C:\Users\LAP\.gemini\antigravity\brain\79a769b3-ff1f-4211-9135-f01868892c28\.system_generated\tasks\task-722.log")
pos = 0
print("=== LIVE PROGRESS: merge_data_train.py ===", flush=True)
while True:
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        if chunk:
            print(chunk, end="", flush=True)
            if "MERGE data_train COMPLETE" in chunk or "No new valid unique images" in chunk:
                break
    time.sleep(2)
