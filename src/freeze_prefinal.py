"""
freeze_prefinal.py -- snapshot a championship's projection BEFORE it is run.

WHY THIS EXISTS
The Results page compares what the model said beforehand against what happened.
That comparison is only worth anything if the "beforehand" half is genuinely
from beforehand. Once a meeting is run its marks flow into the next refresh, the
live projection quietly absorbs them, and a comparison drawn from it stops being
a forecast and becomes the model describing what it has already seen.

The Diamond League Final already has such a snapshot
(outputs/predictions_prefinal.csv) and it was taken by hand. This does the same
thing for the Ultimate Championship, and refuses in the two situations where a
snapshot would be worthless:

  * the file already exists -- overwriting it is exactly the mistake the freeze
    exists to prevent, so it needs --force and says so;
  * results already exist for the event -- at that point "before" has passed and
    a snapshot taken now is not a forecast.

THE DEADLINE IS THE POINT. Run this before the first session. There is no way to
reconstruct it afterwards: the projection it captures no longer exists once the
data behind it moves.

Usage:
    python src/freeze_prefinal.py            # freeze the Ultimate
    python src/freeze_prefinal.py --status   # say what is frozen, change nothing
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ULTIMATE_DIR = os.path.join(BASE_DIR, "data", "ultimate")
LIVE_PATH = os.path.join(ULTIMATE_DIR, "predictions.json")
FROZEN_PATH = os.path.join(ULTIMATE_DIR, "predictions_prefinal.json")
EVENT_PATH = os.path.join(ULTIMATE_DIR, "event.json")


def results_exist():
    """True once WA is serving results for the event, which is the moment a
    fresh snapshot stops being a forecast."""
    try:
        with open(EVENT_PATH, encoding="utf-8") as f:
            event = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(event.get("results")) or bool(event.get("resultsAvailable"))


def describe(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"events": 0, "athletes": 0, "broken": True}
    projections = data.get("projections") or []
    return {
        "events": len(projections),
        "athletes": sum(len(p.get("athletes") or []) for p in projections),
        "frozenAt": data.get("frozenAt"),
    }


def freeze(force=False):
    if not os.path.exists(LIVE_PATH):
        print(f"  nothing to freeze: {LIVE_PATH} does not exist")
        print("  run python src/ultimate_predictions.py first")
        return 1

    if os.path.exists(FROZEN_PATH) and not force:
        have = describe(FROZEN_PATH)
        print(f"  ALREADY FROZEN: {FROZEN_PATH}")
        print(f"    {have['events']} events, {have['athletes']} athletes"
              f"{', taken ' + have['frozenAt'] if have.get('frozenAt') else ''}")
        print("  Refusing to overwrite. A snapshot taken later is not the one the")
        print("  Results page needs. Pass --force only if you are certain the")
        print("  existing file was taken at the wrong time.")
        return 1

    if results_exist() and not force:
        print("  RESULTS ALREADY EXIST for this event.")
        print("  A projection frozen now has had the chance to see them, so it is")
        print("  not a forecast and the comparison would be meaningless.")
        return 1

    with open(LIVE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Stamped so the page can say WHEN the call was made, and so a snapshot
    # taken at the wrong moment is visible rather than having to be inferred.
    data["frozenAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(FROZEN_PATH), exist_ok=True)
    with open(FROZEN_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    now = describe(FROZEN_PATH)
    print(f"  FROZEN: {now['events']} events, {now['athletes']} athletes -> {FROZEN_PATH}")
    print("  Commit this file. It cannot be reconstructed after the event runs.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true",
                        help="Report what is frozen and change nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing snapshot. Almost never right.")
    args = parser.parse_args()

    print("=== Pre-event projection snapshot ===")
    if args.status:
        live, frozen = describe(LIVE_PATH), describe(FROZEN_PATH)
        print(f"  live    {LIVE_PATH}")
        print(f"          {live if live else 'absent'}")
        print(f"  frozen  {FROZEN_PATH}")
        print(f"          {frozen if frozen else 'ABSENT -- nothing is frozen yet'}")
        print(f"  results published: {results_exist()}")
        sys.exit(0)

    sys.exit(freeze(force=args.force))
