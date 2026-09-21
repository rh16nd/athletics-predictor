"""refresh_results.py pushes to the live site, so what it commits matters.

It must commit the data files it rebuilt and nothing else. On 2026-09-21 the
frontend's index held two staged deletions from unfinished work; a plain
`git commit` would have pushed them with the results and broken the build.
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
