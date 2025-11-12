"""
End-to-end test for the complete softmax module BJH loop.

This test validates the entire user journey through Build → Justify → Harden,
ensuring all components (State, Curriculum, Validation, LLM) work together correctly.

Key validations:
- Build stage: Implementation submission and validation
- Justify (shallow): Fast filter catches vague answers (LLM not called)
- Justify (deep): LLM evaluation on detailed answers (LLM called once)
- Harden stage: Bug injection, debugging, and fix validation
- State transitions: Progress advances correctly through all stages
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from engine.main import app
from engine.schemas import UserProgress, LLMEvaluationResponse


runner = CliRunner()


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """
    Create isolated workspace and state file for E2E testing.
    
    This fixture ensures tests don't interfere with each other or the user's
    actual workspace/state files.
    """
    # Set up temporary workspace
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    
    # Set up temporary state file location
    state_file = tmp_path / ".mastery_progress.json"
    
    # Patch the workspace and state file paths
    monkeypatch.setattr("engine.workspace.WorkspaceManager.WORKSPACE_DIR", workspace_dir)
    monkeypatch.setattr("engine.state.StateManager.STATE_FILE", state_file)
    
    return {
        "workspace": workspace_dir,
        "state_file": state_file,
        "tmp_path": tmp_path
    }


def load_state(state_file: Path) -> UserProgress:
    """Helper to load and parse state file."""
    with open(state_file, 'r') as f:
        data = json.load(f)
    return UserProgress.model_validate(data)


class TestFullSoftmaxLoop:
    """
    End-to-end test for the complete softmax module journey.
    
    Tests the full Build → Justify → Harden loop with real curriculum,
    mocked LLM, and comprehensive state assertions.
    """
    
    # Correct softmax implementation (for build stage)
    CORRECT_SOFTMAX = """import torch
from jaxtyping import Float


def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    \"\"\"Numerically-stable softmax using subtract-max trick.\"\"\"
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    max_vals = x32.max(dim=dim, keepdim=True).values
    shifted = x32 - max_vals
    exps = torch.exp(shifted)
    sums = exps.sum(dim=dim, keepdim=True)
    out = exps / sums
    return out.to(orig_dtype)
"""
    
    def test_complete_softmax_bjh_loop(self, isolated_workspace, mocker):
        """
        Test the complete softmax Build-Justify-Harden loop.
        
        This is the fortress test that validates all components working together.
        """
        workspace = isolated_workspace["workspace"]
        state_file = isolated_workspace["state_file"]
        
        # ========================================
        # PHASE 1: BUILD STAGE
        # ========================================
        
        # Initialize state with cs336_a1 curriculum
        initial_state = UserProgress(
            curriculum_id="cs336_a1",
            current_module_index=0,
            current_stage="build"
        )
        with open(state_file, 'w') as f:
            f.write(initial_state.model_dump_json(indent=2))
        
        # View build prompt
        result = runner.invoke(app, ["next"])
        assert result.exit_code == 0
        assert "Numerically Stable Softmax" in result.stdout
        assert "subtract-max trick" in result.stdout
        
        # Write correct implementation
        softmax_file = workspace / "cs336_basics" / "utils.py"
        softmax_file.parent.mkdir(parents=True, exist_ok=True)
        softmax_file.write_text(self.CORRECT_SOFTMAX)
        
        # Submit build (this will run the validator)
        # Note: We need to mock the validator to avoid running actual pytest
        with patch('engine.validator.ValidationSubsystem.execute') as mock_validator:
            mock_validator.return_value = MagicMock(
                exit_code=0,
                stdout="All tests passed\nPERFORMANCE_SECONDS: 0.15",
                stderr="",
                performance_seconds=0.15
            )
            
            result = runner.invoke(app, ["submit-build"])
            assert result.exit_code == 0
            assert "Validation Passed" in result.stdout
            
            # Verify state advanced to justify
            state = load_state(state_file)
            assert state.current_stage == "justify"
            assert state.current_module_index == 0
        
        # ========================================
        # PHASE 2: JUSTIFY STAGE (SHALLOW PATH)
        # ========================================
        
        # View justify question
        result = runner.invoke(app, ["next"])
        assert result.exit_code == 0
        assert "subtract-max trick" in result.stdout
        assert "stub mode" not in result.stdout.lower()  # No longer stub
        
        # Submit shallow answer that should trigger fast filter
        # The softmax question has a failure mode with keyword "stability"
        shallow_answer = "It makes the code more stable and better"
        
        # Mock LLMService to verify it's NOT called
        mock_llm = MagicMock()
        
        with patch('engine.main.LLMService', return_value=mock_llm):
            result = runner.invoke(app, ["submit-justification", shallow_answer])
            assert result.exit_code == 0
            
            # Verify fast filter feedback is shown
            assert "Needs More Depth" in result.stdout or "more specific" in result.stdout.lower()
            
            # CRITICAL: Verify LLM was NOT called
            mock_llm.evaluate_justification.assert_not_called()
            
            # Verify state did NOT advance
            state = load_state(state_file)
            assert state.current_stage == "justify"
        
        # ========================================
        # PHASE 3: JUSTIFY STAGE (DEEP PATH)
        # ========================================
        
        # Submit deep, correct answer that passes fast filter
        deep_answer = (
            "The subtract-max trick prevents overflow by shifting the input range to (-inf, 0]. "
            "It works because softmax(x) = softmax(x - c) for any constant c, as exp(c) cancels "
            "out in the numerator and denominator. By choosing c = max(x), the largest value "
            "becomes 0, so exp(0) = 1.0, preventing overflow while maintaining mathematical correctness."
        )
        
        # Mock LLM to return success
        mock_llm = MagicMock()
        mock_evaluation = LLMEvaluationResponse(
            is_correct=True,
            feedback="Excellent! You've demonstrated deep understanding of the numerical stability mechanism."
        )
        mock_llm.evaluate_justification.return_value = mock_evaluation
        
        with patch('engine.main.LLMService', return_value=mock_llm):
            result = runner.invoke(app, ["submit-justification", deep_answer])
            assert result.exit_code == 0
            
            # Verify success feedback
            assert "Correct Understanding" in result.stdout
            assert "Excellent" in result.stdout
            
            # CRITICAL: Verify LLM WAS called once
            mock_llm.evaluate_justification.assert_called_once()
            
            # Verify state advanced to harden
            state = load_state(state_file)
            assert state.current_stage == "harden"
        
        # ========================================
        # PHASE 4: HARDEN STAGE
        # ========================================
        
        # View harden challenge (creates harden workspace with bug)
        with patch('engine.workspace.WorkspaceManager.create_harden_workspace') as mock_create_harden:
            with patch('engine.workspace.WorkspaceManager.apply_patch') as mock_apply_patch:
                # Mock harden workspace creation
                harden_file = workspace / "harden" / "cs336_basics" / "utils.py"
                mock_create_harden.return_value = harden_file
                
                result = runner.invoke(app, ["next"])
                assert result.exit_code == 0
                assert "Debug Challenge" in result.stdout
                assert "NaN" in result.stdout  # Bug symptom
                
                # Verify harden workspace was created
                mock_create_harden.assert_called_once()
                mock_apply_patch.assert_called_once()
        
        # Fix the bug (restore the subtract-max trick)
        harden_dir = workspace / "harden"
        harden_dir.mkdir(parents=True, exist_ok=True)
        harden_softmax = harden_dir / "cs336_basics" / "utils.py"
        harden_softmax.parent.mkdir(parents=True, exist_ok=True)
        harden_softmax.write_text(self.CORRECT_SOFTMAX)
        
        # Submit fix
        with patch('engine.validator.ValidationSubsystem.execute') as mock_validator:
            mock_validator.return_value = MagicMock(
                exit_code=0,
                stdout="All tests passed",
                stderr="",
                performance_seconds=None
            )
            
            result = runner.invoke(app, ["submit-fix"])
            assert result.exit_code == 0
            assert "Bug Fixed" in result.stdout
            
            # Verify state advanced to complete (or next module)
            state = load_state(state_file)
            # Since softmax is the only module, current_module_index should advance
            assert state.current_module_index == 1
            assert state.current_stage == "build"  # Ready for next module
    
    def test_validation_chain_branches_correctly(self, isolated_workspace, mocker):
        """
        Isolated test to verify the validation chain branching logic.
        
        This test specifically validates that:
        1. Fast filter blocks shallow answers without calling LLM
        2. Deep answers bypass fast filter and trigger LLM
        """
        state_file = isolated_workspace["state_file"]
        
        # Set up state in justify stage
        justify_state = UserProgress(
            curriculum_id="cs336_a1",
            current_module_index=0,
            current_stage="justify"
        )
        with open(state_file, 'w') as f:
            f.write(justify_state.model_dump_json(indent=2))
        
        # Test 1: Keyword "stability" should trigger fast filter
        mock_llm = MagicMock()
        with patch('engine.main.LLMService', return_value=mock_llm):
            result = runner.invoke(app, ["submit-justification", "It improves stability"])
            mock_llm.evaluate_justification.assert_not_called()
        
        # Test 2: Keyword "overflow" alone should trigger fast filter
        mock_llm = MagicMock()
        with patch('engine.main.LLMService', return_value=mock_llm):
            result = runner.invoke(app, ["submit-justification", "It prevents overflow"])
            mock_llm.evaluate_justification.assert_not_called()
        
        # Test 3: No matching keywords should call LLM
        mock_llm = MagicMock()
        mock_llm.evaluate_justification.return_value = LLMEvaluationResponse(
            is_correct=False,
            feedback="Try explaining the mathematical equivalence."
        )
        with patch('engine.main.LLMService', return_value=mock_llm):
            result = runner.invoke(app, ["submit-justification", "The technique adjusts values"])
            mock_llm.evaluate_justification.assert_called_once()
