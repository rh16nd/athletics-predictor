"""The snapshot builder's own logic -- which athletes get a file, and which
deliberately do not.

Everything here runs against a fake Flask test client rather than the real app:
these are decisions about *sets of athletes*, and pinning them should not
depend on what the last scrape happened to leave on disk, nor cost a World
Athletics round trip per athlete.

The bug this file exists to prevent is a quiet one. A missing snapshot is
invisible -- the page still renders, it just renders after a 32-second wait
that only the first visitor of the day sees.
"""
import json
import os

import build_static_api as b


class _Res:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def get_json(self):
        return self._payload


class _Client:
    """Answers the two paths the builder asks for, and records what it was
    asked so a test can assert on the requests themselves."""

    def __init__(self, index=None, per_path=None):
        self.index = index or {"athletes": []}
        self.per_path = per_path or {}
        self.seen = []

    def get(self, path):
        self.seen.append(path)
        if path == "/api/search-index":
            return _Res(self.index)
        if path in self.per_path:
            return self.per_path[path]
        return _Res({"error": "not found"}, 404)


def _index(*rows):
    return {"athletes": [list(r) for r in rows]}


def test_status_pairs_stops_at_the_requested_depth():
    client = _Client(_index(
        ("Ranked FIRST", "men_100m", "9.9", 1),
        ("Ranked FIFTIETH", "men_100m", "10.3", 50),
        ("Ranked FIFTY FIRST", "men_100m", "10.4", 51),
    ))
    names = [n for _k, n in b.status_pairs(client, 50, set())]
    assert names == ["Ranked FIRST", "Ranked FIFTIETH"]


def test_status_pairs_skips_athletes_who_already_have_a_full_profile():
    """A finalist's profile is written by the pass before this one. Writing a
    status page for them too would be a second, lesser answer to the same
    question."""
    client = _Client(_index(
        ("In THE FIELD", "men_100m", "9.9", 1),
        ("Outside IT", "men_100m", "10.0", 2),
    ))
    already = {("men_100m", "In THE FIELD")}
    assert [n for _k, n in b.status_pairs(client, 50, already)] == ["Outside IT"]


def test_status_pairs_skips_the_unranked():
    """No world rank means no place in the depth cut at all -- including it
    would quietly make --profile-depth mean nothing."""
    client = _Client(_index(("No RANK", "men_100m", "10.9", None)))
    assert b.status_pairs(client, 50, set()) == []


def test_status_pairs_asks_for_each_athlete_once():
    """The same name can be ranked in several disciplines, but a repeat within
    ONE discipline would write the same file twice."""
    client = _Client(_index(
        ("Same PERSON", "men_100m", "9.9", 1),
        ("Same PERSON", "men_100m", "9.9", 1),
        ("Same PERSON", "men_200m", "19.9", 2),
    ))
    assert b.status_pairs(client, 50, set()) == [
        ("men_100m", "Same PERSON"),
        ("men_200m", "Same PERSON"),
    ]


def test_status_pairs_survives_a_missing_index():
    """No snapshot to read from is a skip, not a crash: the rest of the build
    is still worth writing."""
    client = _Client()
    client.get = lambda path: _Res({"error": "nope"}, 404)
    assert b.status_pairs(client, 50, set()) == []


def test_write_snapshots_names_files_by_the_frontend_s_slug(tmp_path):
    client = _Client(per_path={
        "/api/athlete-status/men_100m/Yeral%20NU%C3%91EZ": _Res({"name": "Yeral NUÑEZ"}),
    })
    written, skipped, total = b.write_snapshots(
        client, str(tmp_path), "athlete-status", [("men_100m", "Yeral NUÑEZ")], "status",
    )
    dest = tmp_path / "athlete-status" / "men_100m" / "yeral-nunez.json"
    assert (written, skipped) == (1, 0)
    assert total > 0
    assert json.loads(dest.read_text(encoding="utf-8"))["name"] == "Yeral NUÑEZ"


def test_write_snapshots_writes_neither_side_of_a_slug_collision(tmp_path):
    """Two names folding to one file would serve one athlete the other's page.
    Skipping both sends them to the live API, which cannot be wrong."""
    client = _Client(per_path={
        "/api/athlete-status/men_100m/Jose%20GOMEZ": _Res({"name": "Jose GOMEZ"}),
        "/api/athlete-status/men_100m/Jos%C3%A9%20G%C3%B3mez": _Res({"name": "José Gómez"}),
    })
    written, skipped, _total = b.write_snapshots(
        client, str(tmp_path), "athlete-status",
        [("men_100m", "Jose GOMEZ"), ("men_100m", "José Gómez")], "status",
    )
    assert (written, skipped) == (1, 1)
    assert len(os.listdir(tmp_path / "athlete-status" / "men_100m")) == 1


def test_write_snapshots_skips_a_non_200_rather_than_freezing_an_error(tmp_path):
    """A 404 written to disk would be served as the athlete's page forever.
    Absent, the frontend falls through to the live API and can recover."""
    client = _Client()
    written, skipped, _total = b.write_snapshots(
        client, str(tmp_path), "athlete-status", [("men_100m", "Nobody HERE")], "status",
    )
    assert (written, skipped) == (0, 1)
    assert not os.path.exists(tmp_path / "athlete-status")
