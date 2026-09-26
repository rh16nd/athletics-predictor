"""refresh_results.py pushes to the live site, so what it commits matters.

It must commit the data files it rebuilt and nothing else. Both halves of that
have bitten. On 2026-09-21 the frontend's index held two staged deletions from
unfinished work; a plain `git commit` would have pushed them with the results
and broken the build. And until 2026-09-24 the filter that keeps a full build's
athlete pages out of a results refresh also threw away the 36 event pages that
refresh had just rebuilt, so the live event pages ran a refresh behind.
"""
import subprocess

import refresh_results as r


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout


def test_the_commit_names_its_files():
    assert r.commit_command("msg", ["a.json", "b.json"])[-3:] == ["--", "a.json", "b.json"]


def test_a_staged_change_that_is_not_ours_stays_out_of_the_commit(tmp_path):
    repo = tmp_path
    _git(repo, "init", "-q", "-b", "main")
    (repo / "podium.tsx").write_text("keep me\n")
    (repo / "results.json").write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "start")

    # Someone else's unfinished work, staged.
    _git(repo, "rm", "-q", "podium.tsx")
    # The run's own files: one changed, one new.
    (repo / "results.json").write_text('{"events": 1}\n')
    (repo / "championship.json").write_text("{}\n")
    files = ["results.json", "championship.json"]
    _git(repo, "add", "--", *files)

    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *r.commit_command("results", files)[1:]],
        cwd=repo, check=True,
    )

    committed = set(_git(repo, "show", "--name-only", "--format=", "HEAD").split())
    assert committed == {"results.json", "championship.json"}
    assert "podium.tsx" in _git(repo, "ls-tree", "--name-only", "HEAD")
    # The unrelated deletion is still staged, untouched.
    assert "D  podium.tsx" in _git(repo, "status", "--porcelain")


def test_a_changed_event_page_is_committed_and_an_athlete_page_is_not(tmp_path):
    """A --core-only run rewrites public/data/discipline/<key>.json for every
    event, so a changed one belongs in the commit. The athlete, athlete-status
    and country folders come from a full build and do not."""
    repo = tmp_path
    _git(repo, "init", "-q", "-b", "main")
    rebuilt = [
        "public/data/championship.json",
        "public/data/discipline/men_DT.json",
        "public/data/athlete/men_DT/mykolas-alekna.json",
        "public/data/athlete-status/men_DT/someone-else.json",
        "public/data/country/LTU.json",
    ]
    for rel in rebuilt:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "start")

    # Every one of them changed: the deep folders left over from some earlier
    # full build, the other two written by this run.
    for rel in rebuilt:
        (repo / rel).write_text('{"changed": 1}\n')

    assert sorted(r.changed(repo, "public/data", core_only=True)) == [
        "public/data/championship.json",
        "public/data/discipline/men_DT.json",
    ]
    # Without the flag nothing is held back.
    assert sorted(r.changed(repo, "public/data")) == sorted(rebuilt)


def test_the_core_folders_are_read_from_the_builder():
    """core_dirs() asks build_static_api what --core-only writes instead of
    keeping a second copy of the answer here, which is how the two drifted
    apart in the first place."""
    import build_static_api

    names = [name for _, name in build_static_api.snapshot_paths()]
    assert any(name.startswith("discipline/") for name in names)

    dirs = r.core_dirs()
    assert {"", "discipline"} <= dirs
    assert not dirs & {"athlete", "athlete-status", "country"}
