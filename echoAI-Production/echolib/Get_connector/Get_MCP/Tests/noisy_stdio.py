import sys
import json
import time

# Simulate startup noise
print("Starting tool v1.0", flush=True)
print("Warning: experimental mode enabled", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        print("ERROR: invalid input", flush=True)
        continue

    # Simulate log noise
    print("Processing request...", flush=True)
    time.sleep(0.2)

    response = {
        "status": "ok",
        "data": {
            "received": payload
        }
    }

    print(json.dumps(response), flush=True)
