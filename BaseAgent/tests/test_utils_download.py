"""Unit tests for BaseAgent.utils.download module."""

from __future__ import annotations

import os

import pytest

from BaseAgent.utils.download import check_or_create_path


class TestCheckOrCreatePath:
    """Tests for check_or_create_path()."""

    def test_creates_new_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_subdir")
        assert not os.path.exists(new_dir)
        result = check_or_create_path(new_dir)
        assert os.path.exists(new_dir)
        assert result == new_dir

    def test_existing_directory_is_idempotent(self, tmp_path):
        existing = str(tmp_path)
        result = check_or_create_path(existing)
        assert result == existing
        assert os.path.isdir(existing)

    def test_returns_path(self, tmp_path):
        path = str(tmp_path / "output")
        result = check_or_create_path(path)
        assert isinstance(result, str)
        assert result == path

    def test_default_path_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = check_or_create_path(None)
        assert os.path.exists(result)
        assert "tmp_directory" in result
