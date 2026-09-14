"""freeze_prefinal.py's two refusals, and the freeze itself, for any registered
championship. A snapshot is only worth anything if it was taken before the event
and never overwritten, so the refusals are the point of the script."""
import json

import pytest

import championships
import freeze_prefinal as fp


@pytest.fixture
def champ_dir(tmp_path, monkeypatch):
    """A throwaway championship whose data directory is a temp folder."""
    entry = {"id": "test-games-2026", "labelKey": "results.meet.test", "venue": "Nowhere",
             "startDate": "2026-10-01", "endDate": "2026-10-02", "theme": "ultimate",
             "competitionId": 1, "area": None, "dataDir": str(tmp_path), "scraper": None}
    monkeypatch.setattr(championships, "CHAMPIONSHIPS", championships.CHAMPIONSHIPS + [entry])
    return tmp_path


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_freezes_the_live_call_and_stamps_when_it_was_taken(champ_dir):
    _write(champ_dir / "predictions.json",
           {"projections": [{"discKey": "men_PV", "athletes": [{"name": "Armand DUPLANTIS"}]}]})
    assert fp.freeze("test-games-2026") == 0
    frozen = json.loads((champ_dir / "predictions_prefinal.json").read_text(encoding="utf-8"))
    assert frozen["projections"][0]["discKey"] == "men_PV"
    assert frozen["frozenAt"].endswith("Z")


def test_refuses_to_overwrite_an_existing_snapshot(champ_dir):
    _write(champ_dir / "predictions.json", {"projections": [{"discKey": "taken-later"}]})
    _write(champ_dir / "predictions_prefinal.json",
           {"projections": [{"discKey": "taken-before"}], "frozenAt": "2026-09-22T00:00:00Z"})
    assert fp.freeze("test-games-2026") == 1
    kept = json.loads((champ_dir / "predictions_prefinal.json").read_text(encoding="utf-8"))
    assert kept["projections"][0]["discKey"] == "taken-before"


def test_refuses_once_results_exist(champ_dir):
    _write(champ_dir / "predictions.json", {"projections": []})
    _write(champ_dir / "event.json", {"results": [{"discipline": "men_PV"}]})
    assert fp.freeze("test-games-2026") == 1
    assert not (champ_dir / "predictions_prefinal.json").exists()


def test_a_championship_without_its_own_data_cannot_be_frozen():
    with pytest.raises(SystemExit):
        fp.paths("dl-final-2026")
