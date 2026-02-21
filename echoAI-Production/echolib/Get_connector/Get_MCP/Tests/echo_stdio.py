import sys
import json

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            error = {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}"
            }
            print(json.dumps(error), flush=True)
            continue

        response = {
            "status": "ok",
            "data": {
                "received": payload
            }
        }

        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()
