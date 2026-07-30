"""Tests for data/kaggle_paths.py's resolve_kaggle_dataset_path: the real,
recurring bug it exists to work around is that Kaggle's
/kaggle/input/datasets/<username>/... mount path's username segment
changes between sessions for the same attached dataset (confirmed twice,
2026-07-29/30, on this exact project -- sumayarahman30 one session,
sumayarahmanmeherin the next). CPU-only; monkeypatches the module's
hardcoded /kaggle/input/datasets/ prefix to a tmp_path-based fake mount so
this is testable off Kaggle.
"""
from pathlib import Path

import pytest

import data.kaggle_paths as kaggle_paths
from data.kaggle_paths import resolve_kaggle_dataset_path


@pytest.fixture
def fake_datasets_root(tmp_path, monkeypatch):
    """Redirects the module's hardcoded prefix at a tmp directory and
    returns that prefix (as a string, trailing slash included, matching
    the real constant's format)."""
    root = tmp_path / "datasets"
    root.mkdir()
    prefix = str(root) + "/"
    monkeypatch.setattr(kaggle_paths, "_DATASETS_PREFIX", prefix)
    return prefix


def test_existing_path_is_returned_unchanged(fake_datasets_root):
    path = fake_datasets_root + "alice/some-dataset"
    Path(path).mkdir(parents=True)
    assert resolve_kaggle_dataset_path(path) == path


def test_path_outside_datasets_prefix_is_returned_unchanged_even_if_missing():
    path = "/some/totally/unrelated/path/that/does/not/exist"
    assert resolve_kaggle_dataset_path(path) == path


def test_substitutes_the_username_when_exactly_one_other_username_matches(fake_datasets_root):
    Path(fake_datasets_root, "bob", "annoted-sct", "deep", "nested", "path").mkdir(parents=True)

    stale_path = fake_datasets_root + "alice/annoted-sct/deep/nested/path"  # "alice" doesn't exist
    resolved = resolve_kaggle_dataset_path(stale_path)
    assert resolved == fake_datasets_root + "bob/annoted-sct/deep/nested/path"


def test_raises_when_multiple_other_usernames_match(fake_datasets_root):
    for username in ["bob", "carol"]:
        (Path(fake_datasets_root) / username / "annoted-sct" / "rest").mkdir(parents=True)

    stale_path = fake_datasets_root + "alice/annoted-sct/rest"
    with pytest.raises(FileNotFoundError, match="more than one"):
        resolve_kaggle_dataset_path(stale_path)


def test_returns_original_path_unchanged_when_no_substitute_exists_either(fake_datasets_root):
    # some OTHER dataset exists under a different username, but not the one we're looking for
    (Path(fake_datasets_root) / "bob" / "a-totally-different-dataset").mkdir(parents=True)

    stale_path = fake_datasets_root + "alice/annoted-sct/rest"
    assert resolve_kaggle_dataset_path(stale_path) == stale_path


def test_returns_original_path_unchanged_when_datasets_root_itself_is_missing(tmp_path, monkeypatch):
    missing_prefix = str(tmp_path / "does_not_exist_at_all") + "/"
    monkeypatch.setattr(kaggle_paths, "_DATASETS_PREFIX", missing_prefix)
    path = missing_prefix + "alice/annoted-sct"
    assert resolve_kaggle_dataset_path(path) == path


def test_path_too_short_to_contain_a_username_segment_is_returned_unchanged(fake_datasets_root):
    # just the prefix itself, nothing after it that could be split into <username>/<rest>
    path = fake_datasets_root.rstrip("/")
    assert resolve_kaggle_dataset_path(path) == path


def test_does_not_offer_the_stale_username_itself_as_its_own_candidate(fake_datasets_root):
    """The stale username directory might exist but just be missing the
    specific dataset -- must not be re-checked as its own substitute."""
    Path(fake_datasets_root, "alice").mkdir()  # alice exists, but has no "annoted-sct" inside it
    Path(fake_datasets_root, "bob", "annoted-sct").mkdir(parents=True)

    stale_path = fake_datasets_root + "alice/annoted-sct"
    resolved = resolve_kaggle_dataset_path(stale_path)
    assert resolved == fake_datasets_root + "bob/annoted-sct"
