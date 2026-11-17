"""Read-only LUMIERE inventory: list what is on disk and print counts only (no patient or study names).

Usage: python scripts/audit_lumiere.py --root /content/drive/MyDrive/LUMIERE [--workers 16] [--inventory OUT.json]
The optional inventory file keeps patient and study names: write it to private local storage only.
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rhuh_synthesis.lumiere import run_audit  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="LUMIERE folder (read only)")
    parser.add_argument("--workers", type=int, default=16, help="parallel folder listings")
    parser.add_argument("--inventory", help="optional private JSON with the full listing")
    args = parser.parse_args(argv)
    summary = run_audit(args.root, workers=args.workers, inventory_out=args.inventory)
    print("SUMMARY")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
