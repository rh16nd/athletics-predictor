"""Put the latest championship results on the live site in one go.

Double-click refresh-results.cmd in the repo root, or run from the repo root:

    venv\\Scripts\\python.exe src\\refresh_results.py          (asks before pushing)
    venv\\Scripts\\python.exe src\\refresh_results.py --yes    (pushes without asking)

It stops at the first step that fails, and pushes nothing unless every step
before the push worked:

  1. src/ultimate_scraper.py fetches the results from World Athletics into
     data/ultimate/event.json. It refuses to save a run that lost data the saved
     file already has, which is what a failed fetch looks like.
  2. src/build_static_api.py --core-only rewrites the page-level JSON the site
     reads (results, the championship, predictions) in seconds, without the long
     athlete-profile pass.
  3. It shows how many events have results and which files changed, then
     commits exactly those files in both repos and pushes main. Vercel and
     Render redeploy on their own.

World Athletics moves its data server now and then. The scrapers look the new
one up by themselves (dl_final_results_scraper.discover_endpoint), so a run that
still fails for every event at once should start there.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND = os.path.abspath(os.path.join(ROOT, "..", "track-insights-main"))
EVENT_PATH = os.path.join(ROOT, "data", "ultimate", "event.json")


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8")


def step(title, cmd, cwd=ROOT):
    print(f"\n== {title}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"\nStopped: {title[0].lower() + title[1:]} failed. Nothing was pushed.")


def events_with_results():
    try:
        with open(EVENT_PATH, encoding="utf-8") as f:
            rows = json.load(f).get("results") or []
    except (OSError, ValueError):
        return []
    return sorted({r.get("discipline") for r in rows if r.get("discipline")})


def changed(repo, path, top_level_only=False):
    files = []
    for line in git(repo, "status", "--porcelain", "--", path).stdout.splitlines():
        name = line[3:].strip()
        # --core-only writes public/data/*.json and never the athlete or country
        # folders, so anything changed down there is not this run's to commit.
        if name and (not top_level_only or name.count("/") == path.count("/") + 1):
            files.append(name)
    return files


def main():
    ask = "--yes" not in sys.argv[1:]
    for repo in (ROOT, FRONTEND):
        name = os.path.basename(repo)
        if git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() != "main":
            sys.exit(f"Stopped: {name} is not on the main branch.")
        git(repo, "fetch", "-q", "origin", "main")
        behind = git(repo, "rev-list", "--count", "HEAD..origin/main").stdout.strip()
        if behind not in ("", "0"):
            sys.exit(f"Stopped: {name} is {behind} commit(s) behind GitHub. Run git pull there first.")

    before = events_with_results()
    step("Fetching results from World Athletics",
         [sys.executable, os.path.join("src", "ultimate_scraper.py")])
    after = events_with_results()
    step("Rebuilding the site's data files",
         [sys.executable, os.path.join("src", "build_static_api.py"), "--core-only"])

    print(f"\nEvents with results: {len(after)} (before this run: {len(before)})")
    new = [e for e in after if e not in before]
    if new:
        print("  new: " + ", ".join(new))
    if len(after) < len(before):
        print("  WARNING: fewer events have results than before this run. Check before pushing.")
        ask = True

    backend = changed(ROOT, "data/ultimate/event.json")
    frontend = changed(FRONTEND, "public/data", top_level_only=True)
    if not backend and not frontend:
        print("\nNothing changed, so there is nothing to push.")
        return
    print("\nFiles that will be committed:")
    for f in backend:
        print(f"  athletics-predictor/{f}")
    for f in frontend:
        print(f"  track-insights-main/{f}")

    if ask:
        try:
            answer = input("\nPush these to the live site? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Not pushed. The files are updated on this computer only.")
            return

    message = f"Add the latest championship results, {len(after)} events in"
    for repo, files in ((ROOT, backend), (FRONTEND, frontend)):
        if not files:
            continue
        name = os.path.basename(repo)
        step(f"Staging {name}", ["git", "add", "--", *files], cwd=repo)
        step(f"Committing {name}", ["git", "commit", "-q", "-m", message], cwd=repo)
        step(f"Pushing {name}", ["git", "push", "-q", "origin", "main"], cwd=repo)
    print("\nDone. The site shows the new results within a couple of minutes.")


if __name__ == "__main__":
    main()
