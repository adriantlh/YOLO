import os
import sys
import urllib.request


def main() -> int:
    # Always check via loopback inside the container
    port = os.environ.get("PORT", "8000")
    url = f"http://127.0.0.1:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 0 if resp.getcode() == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())

