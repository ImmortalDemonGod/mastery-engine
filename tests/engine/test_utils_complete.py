"""Complete test coverage for engine/utils.py."""

import pytest
from pathlib import Path
import os
from engine.utils import find_project_root


class TestFindProjectRoot:
    """Complete coverage for find_project_root function."""
    
    def test_find_by_pyproject_toml(self, tmp_path):
        """Should find root by pyproject.toml."""
        # Create structure
        (tmp_path / "pyproject.toml").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        # Change to subdir and find root
        old_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            root = find_project_root()
            assert root == tmp_path
        finally:
            os.chdir(old_cwd)
    
    def test_find_by_git_dir(self, tmp_path):
        """Should find root by .git directory."""
        # Create .git dir (no pyproject.toml)
        (tmp_path / ".git").mkdir()
        subdir = tmp_path / "deep" / "nested"
        subdir.mkdir(parents=True)
        
        old_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            root = find_project_root()
            assert root == tmp_path
        finally:
            os.chdir(old_cwd)
    
    def test_find_by_curricula_and_engine(self, tmp_path):
        """Should find root by curricula + engine directories."""
        # Create curricula and engine dirs (no .git or pyproject)
        (tmp_path / "curricula").mkdir()
        (tmp_path / "engine").mkdir()
        subdir = tmp_path / "workspace"
        subdir.mkdir()
        
        old_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            root = find_project_root()
            assert root == tmp_path
        finally:
            os.chdir(old_cwd)
    
    def test_not_found_raises_error(self, tmp_path):
        """Should raise RuntimeError when no markers found."""
        # Empty directory with no markers
        subdir = tmp_path / "nowhere"
        subdir.mkdir()
        
        old_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            with pytest.raises(RuntimeError, match="Could not find project root"):
                find_project_root()
        finally:
            os.chdir(old_cwd)
    
    def test_with_explicit_start_path(self, tmp_path):
        """Should work with explicit start_path parameter."""
        (tmp_path / "pyproject.toml").touch()
        subdir = tmp_path / "src" / "module"
        subdir.mkdir(parents=True)
        
        # Don't change cwd, pass path explicitly
        root = find_project_root(start_path=subdir)
        assert root == tmp_path
