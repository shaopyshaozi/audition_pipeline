import json
import sys

full_text = []

for line in sys.stdin:
    line = line.strip()

    if not line.startswith("{"):
        continue

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue

    text = data.get("text", "")

    if text:
        emission_time = data.get("emission_time", None)
        if emission_time is not None:
            print(f"[emitted {emission_time:.2f}s] {text}")
        full_text.append(text)

print("\n========== FULL TRANSCRIPT ==========\n")
print(" ".join(full_text))