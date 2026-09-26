"""Put the latest championship results on the live site in one go.

Double-click refresh-results.cmd in the repo root, or run from the repo root:

    venv\\Scripts\\python.exe src\\refresh_results.py          (asks before pushing)
    venv\\Scripts\\python.exe src\\refresh_results.py --yes    (pushes without asking)

It works on the current championship in src/championships.py, stops at the first
step that fails, and pushes nothing unless every step before the push worked:

  1. The championship's scraper fetches the results from World Athletics into its
     event.json. It refuses to save a run that lost data the saved file already
     has, which is what a failed fetch looks like.
  2. src/build_static_api.py --core-only rewrites the page-level JSON the site
     reads -- results, the championship, predictions, and a page for each of
     the 36 events -- in seconds, without the long athlete-profile pass.
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

import championships

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND = os.path.abspath(os.path.join(ROOT, "..", "track-insights-main"))
CHAMP = championships.current()
EVENT_PATH = championships.path(CHAMP, "event.json")


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8")


def step(title, cmd, cwd=ROOT):
    print(f"\n== {title}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"\nStopped: {title[0].lower() + title[1:]} failed. Nothing was pushed.")


def commit_command(message, files):
    """Commit exactly these files, whatever else is staged. A plain commit takes
    the whole index: on 2026-09-21 the frontend's index held two staged
    deletions from unfinished work, which the next refresh would have pushed,
    breaking the live build."""
    return ["git", "commit", "-q", "-m", message, "--", *files]


def events_with_results():
    try:
        with open(EVENT_PATH, encoding="utf-8") as f:
            rows = json.load(f).get("results") or []
    except (OSError, ValueError, TypeError):
        return []
    return sorted({r.get("discipline") for r in rows if r.get("discipline")})


def core_dirs():
    """The folders under public/data that `build_static_api.py --core-only`
    writes, asked of that script instead of assumed here.

    write_core() writes every snapshot_paths() entry, and that is the top-level
    pages *and* one page per discipline: 47 files on 2026-09-24, only 11 of
    them top level. The rule this replaced kept one-segment paths and claimed
    in its comment that core-only never wrote a folder, which was wrong, so
    every refresh held back all 36 event pages and the live site's event pages
    ran a refresh behind the results they were built from.

    Imported inside the function because it pulls in the whole Flask app, a
    second this script has no reason to spend before it knows a refresh even
    changed anything. By the time this runs, the --core-only step has already
    imported the same module in a subprocess and come back clean.
    """
    import build_static_api
    return {os.path.dirname(name).replace("\\", "/")
            for _, name in build_static_api.snapshot_paths()}


def changed(repo, path, core_only=False):
    files = []
    dirs = core_dirs() if core_only else None
    for line in git(repo, "status", "--porcelain", "--", path).stdout.splitlines():
        name = line[3:].strip()
        if not name:
            continue
        if dirs is not None:
            # A --core-only run rewrote the top-level pages and the discipline
            # folder. It does not touch athlete, athlete-status or country --
            # those come from a full build, so a change down there belongs to
            # some other run and is not this one's to commit.
            rel = name[len(path) + 1:] if name.startswith(path + "/") else name
            if os.path.dirname(rel).replace("\\", "/") not in dirs:
                continue
        files.append(name)
    return files


def main():
    ask = "--yes" not in sys.argv[1:]
    if not CHAMP.get("scraper") or not EVENT_PATH:
        sys.exit(f"Stopped: {CHAMP['id']} has no results scraper yet (see src/championships.py).")
    print(f"Championship: {CHAMP['id']}")

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
         [sys.executable, CHAMP["scraper"], *CHAMP.get("scraperArgs", [])])
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

    event_rel = "/".join([*CHAMP["dataDir"].replace("\\", "/").split("/"), "event.json"])
    backend = changed(ROOT, event_rel)
    frontend = changed(FRONTEND, "public/data", core_only=True)
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
        step(f"Committing {name}", commit_command(message, files), cwd=repo)
        step(f"Pushing {name}", ["git", "push", "-q", "origin", "main"], cwd=repo)
    print("\nDone. The site shows the new results within a couple of minutes.")


if __name__ == "__main__":
    main()
