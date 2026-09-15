from pathlib import Path

from cypher_extract.paths import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, get_data_root

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_default_data_root_is_repository_data_directory(monkeypatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)

    assert DEFAULT_DATA_ROOT == REPOSITORY_ROOT / "data"
    assert get_data_root() == DEFAULT_DATA_ROOT


def test_data_root_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    assert get_data_root() == tmp_path
