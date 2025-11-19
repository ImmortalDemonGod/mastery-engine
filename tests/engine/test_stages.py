"""
Comprehensive tests for stage modules (harden and justify).

These tests target the uncovered code in:
- engine/stages/harden.py (28% → target 80%)
- engine/stages/justify.py (34% → target 80%)

Coverage targets:
- HardenRunner.present_challenge() - Challenge setup and bug injection
- HardenRunner._select_bug() - Bug selection logic
- JustifyRunner.load_questions() - Question loading and parsing
- JustifyRunner.check_fast_filter() - Keyword-based filtering
"""

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from engine.stages.harden import HardenRunner, HardenChallengeError
from engine.stages.justify import JustifyRunner, JustifyQuestionsError
from engine.schemas import ModuleMetadata, JustifyQuestion, FailureMode


class TestHardenRunner:
    """Tests for HardenRunner challenge setup."""
    
    def test_init_stores_managers(self):
        """Should store curriculum and workspace managers."""
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        assert runner.curriculum_mgr == mock_curr_mgr
        assert runner.workspace_mgr == mock_workspace_mgr
    
    def test_present_challenge_success(self, tmp_path):
        """Should successfully set up harden challenge in shadow worktree using AST mode."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
        source_file = Path("utils.py")
        
        # Create real file system structure for test
        shadow_worktree = tmp_path / ".mastery_engine_worktree"
        shadow_worktree.mkdir()
        
        # Create student source file
        student_file = tmp_path / "utils.py"
        student_file.write_text("def softmax(x):\\n    return x\\n", encoding='utf-8')
        
        # Mock bug selection - use JSON (AST mode) to avoid needing developer file
        mock_bugs_dir = MagicMock()
        mock_curr_mgr.get_bugs_dir.return_value = mock_bugs_dir
        
        mock_bug_file = MagicMock()
        mock_bug_file.name = "bug1.json"
        mock_bug_file.suffix = ".json"
        mock_symptom_file = MagicMock()
        mock_symptom_file.read_text.return_value = "Function returns incorrect value"
        
        runner._select_bug = MagicMock(return_value=(mock_bug_file, mock_symptom_file))
        
        # Mock AST injector to succeed (patch where it's imported from)
        with patch('engine.ast_harden.generic_injector.GenericBugInjector') as mock_injector_cls:
            mock_injector = MagicMock()
            mock_injector_cls.return_value = mock_injector
            mock_injector.inject.return_value = ("def softmax(x):\\n    # BUGGY\\n    return x\\n", True)
            
            # Change to temp directory for Path.cwd() calls
            import os
            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                # Execute
                harden_file, symptom = runner.present_challenge("test_curriculum", module, source_file)
                
                # Verify
                assert symptom == "Function returns incorrect value"
                mock_curr_mgr.get_bugs_dir.assert_called_once_with("test_curriculum", module)
                runner._select_bug.assert_called_once_with(mock_bugs_dir)
                # In AST mode, no patch application
                mock_workspace_mgr.apply_patch.assert_not_called()
            finally:
                os.chdir(old_cwd)
    
    @patch('engine.stages.harden.Path')
    def test_present_challenge_no_shadow_worktree(self, mock_path_cls):
        """Should raise error if shadow worktree doesn't exist."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
        source_file = Path("utils.py")
        
        # Mock shadow worktree does NOT exist
        mock_shadow_worktree = MagicMock()
        mock_shadow_worktree.exists.return_value = False
        mock_path_cls.return_value = mock_shadow_worktree
        
        # Execute & Verify
        with pytest.raises(HardenChallengeError) as exc_info:
            runner.present_challenge("test_curriculum", module, source_file)
        
        assert "Shadow worktree not found" in str(exc_info.value)
    
    def test_select_bug_success(self):
        """Should successfully select a bug from bugs directory."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        mock_bugs_dir = MagicMock()
        mock_bugs_dir.exists.return_value = True
        
        # Mock patch files
        mock_patch = MagicMock()
        mock_patch.stem = "bug1"
        mock_patch.name = "bug1.patch"
        mock_patch.suffix = ".patch"
        # glob is called twice - once for .patch, once for .json
        def glob_side_effect(pattern):
            if pattern == "*.patch":
                return [mock_patch]
            elif pattern == "*.json":
                return []
            return []
        mock_bugs_dir.glob.side_effect = glob_side_effect
        
        # Mock symptom file
        mock_symptom = MagicMock()
        mock_symptom.exists.return_value = True
        mock_bugs_dir.__truediv__ = lambda self, name: mock_symptom if "symptom" in name else MagicMock()
        
        # Execute
        with patch('engine.stages.harden.random.choice', return_value=mock_patch):
            patch_file, symptom_file = runner._select_bug(mock_bugs_dir)
        
        # Verify
        assert patch_file == mock_patch
        assert symptom_file == mock_symptom
    
    def test_select_bug_no_bugs_dir(self):
        """Should raise error if bugs directory doesn't exist."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        mock_bugs_dir = MagicMock()
        mock_bugs_dir.exists.return_value = False
        
        # Execute & Verify
        with pytest.raises(HardenChallengeError) as exc_info:
            runner._select_bug(mock_bugs_dir)
        
        assert "Bugs directory not found" in str(exc_info.value)
    
    def test_select_bug_no_patches(self):
        """Should raise error if no patch files found."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        mock_bugs_dir = MagicMock()
        mock_bugs_dir.exists.return_value = True
        # glob is called twice - once for .patch, once for .json
        def glob_side_effect(pattern):
            return []  # No files for either pattern
        mock_bugs_dir.glob.side_effect = glob_side_effect
        
        # Execute & Verify
        with pytest.raises(HardenChallengeError) as exc_info:
            runner._select_bug(mock_bugs_dir)
        
        assert "No bug files found" in str(exc_info.value)
    
    def test_select_bug_missing_symptom(self):
        """Should raise error if symptom file is missing."""
        # Setup
        mock_curr_mgr = MagicMock()
        mock_workspace_mgr = MagicMock()
        runner = HardenRunner(mock_curr_mgr, mock_workspace_mgr)
        
        mock_bugs_dir = MagicMock()
        mock_bugs_dir.exists.return_value = True
        
        mock_patch = MagicMock()
        mock_patch.stem = "bug1"
        mock_patch.name = "bug1.patch"
        mock_bugs_dir.glob.return_value = [mock_patch]
        
        # Mock symptom file does NOT exist
        mock_symptom = MagicMock()
        mock_symptom.exists.return_value = False
        mock_bugs_dir.__truediv__ = lambda self, name: mock_symptom
        
        # Execute & Verify
        with patch('engine.stages.harden.random.choice', return_value=mock_patch):
            with pytest.raises(HardenChallengeError) as exc_info:
                runner._select_bug(mock_bugs_dir)
        
        assert "Symptom file missing" in str(exc_info.value)


class TestJustifyRunner:
    """Tests for JustifyRunner question loading and filtering."""
    
    def test_init_stores_curriculum_manager(self):
        """Should store curriculum manager."""
        mock_curr_mgr = MagicMock()
        
        runner = JustifyRunner(mock_curr_mgr)
        
        assert runner.curriculum_mgr == mock_curr_mgr
    
    def test_load_questions_success(self):
        """Should successfully load and parse questions from JSON."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_curr_mgr.get_justify_questions_path.return_value = mock_path
        
        questions_data = [
            {
                "id": "q1",
                "question": "Why softmax?",
                "model_answer": "Because X, Y, Z",
                "failure_modes": [],
                "required_concepts": ["normalization"]
            }
        ]
        
        # Mock file read
        with patch('builtins.open', mock_open(read_data=json.dumps(questions_data))):
            questions = runner.load_questions("test_curriculum", module)
        
        # Verify
        assert len(questions) == 1
        assert questions[0].id == "q1"
        assert questions[0].question == "Why softmax?"
        mock_curr_mgr.get_justify_questions_path.assert_called_once_with("test_curriculum", module)
    
    def test_load_questions_file_not_found(self):
        """Should raise error if questions file doesn't exist."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
        
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_curr_mgr.get_justify_questions_path.return_value = mock_path
        
        # Execute & Verify
        with pytest.raises(JustifyQuestionsError) as exc_info:
            runner.load_questions("test_curriculum", module)
        
        assert "Justify questions not found" in str(exc_info.value)
    
    def test_load_questions_malformed_json(self):
        """Should raise error if JSON is malformed."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        module = ModuleMetadata(id="softmax", name="Softmax", path="modules/softmax")
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_curr_mgr.get_justify_questions_path.return_value = mock_path
        
        # Mock malformed JSON
        with patch('builtins.open', mock_open(read_data="{ invalid json")):
            with pytest.raises(JustifyQuestionsError) as exc_info:
                runner.load_questions("test_curriculum", module)
        
        assert "Malformed JSON" in str(exc_info.value)
    
    @patch.dict('os.environ', {'MASTERY_DISABLE_FAST_FILTER': ''}, clear=False)
    def test_check_fast_filter_matches_keyword(self):
        """Should detect failure mode when keyword matches."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        question = JustifyQuestion(
            id="q1",
            question="Why softmax?",
            model_answer="Because X, Y, Z",
            failure_modes=[
                FailureMode(
                    category="Hand-Waver",
                    keywords=["I don't know", "not sure"],
                    feedback="Let's dig deeper..."
                )
            ],
            required_concepts=["normalization"]
        )
        
        user_answer = "I don't know why we use softmax"
        
        # Execute
        matched, feedback = runner.check_fast_filter(question, user_answer)
        
        # Verify
        assert matched is True
        assert feedback == "Let's dig deeper..."
    
    @patch.dict('os.environ', {'MASTERY_DISABLE_FAST_FILTER': ''}, clear=False)
    def test_check_fast_filter_no_match(self):
        """Should not match when no keywords present."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        question = JustifyQuestion(
            id="q1",
            question="Why softmax?",
            model_answer="Because X, Y, Z",
            failure_modes=[
                FailureMode(
                    category="Hand-Waver",
                    keywords=["I don't know", "not sure"],
                    feedback="Let's dig deeper..."
                )
            ],
            required_concepts=["normalization"]
        )
        
        user_answer = "Softmax applies exponential function and normalizes the output"
        
        # Execute
        matched, feedback = runner.check_fast_filter(question, user_answer)
        
        # Verify
        assert matched is False
        assert feedback is None
    
    @patch.dict('os.environ', {'MASTERY_DISABLE_FAST_FILTER': ''}, clear=False)
    def test_check_fast_filter_case_insensitive(self):
        """Should match keywords case-insensitively."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        question = JustifyQuestion(
            id="q1",
            question="Why softmax?",
            model_answer="Because X, Y, Z",
            failure_modes=[
                FailureMode(
                    category="Hand-Waver",
                    keywords=["vague"],
                    feedback="Be more specific..."
                )
            ],
            required_concepts=["normalization"]
        )
        
        user_answer = "It's VAGUE to me"
        
        # Execute
        matched, feedback = runner.check_fast_filter(question, user_answer)
        
        # Verify
        assert matched is True
        assert feedback == "Be more specific..."
    
    @patch.dict('os.environ', {'MASTERY_DISABLE_FAST_FILTER': 'true'})
    def test_check_fast_filter_disabled_via_env(self):
        """Should skip filtering when disabled via environment variable."""
        # Setup
        mock_curr_mgr = MagicMock()
        runner = JustifyRunner(mock_curr_mgr)
        
        question = JustifyQuestion(
            id="q1",
            question="Why softmax?",
            model_answer="Because X, Y, Z",
            failure_modes=[
                FailureMode(
                    category="Hand-Waver",
                    keywords=["I don't know"],
                    feedback="Let's dig deeper..."
                )
            ],
            required_concepts=["normalization"]
        )
        
        user_answer = "I don't know why"
        
        # Execute
        matched, feedback = runner.check_fast_filter(question, user_answer)
        
        # Verify - should NOT match because filter is disabled
        assert matched is False
        assert feedback is None
