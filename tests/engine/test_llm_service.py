"""
Unit tests for engine.services.llm_service.LLMService.

The Justify grader runs on a real `claude -p` subprocess (Cycle-2). These tests
mock `subprocess.run` so no live grader call is made, and cover the JSON parsing,
the prose/fence tolerance, the fail-closed behavior, and the explicit demo mock.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from engine.services.llm_service import (
    LLMService,
    LLMResponseError,
    LLMAPIError,
    _extract_json,
)
from engine.schemas import JustifyQuestion, FailureMode


@pytest.fixture
def sample_question():
    """Create a sample justify question for testing."""
    return JustifyQuestion(
        id="test_question",
        question="Why is the subtract-max trick used in softmax?",
        model_answer="It prevents overflow by shifting the range to (-inf, 0].",
        failure_modes=[
            FailureMode(
                category="Vague",
                keywords=["stability", "better"],
                feedback="Be more specific about the mechanism."
            )
        ],
        required_concepts=["overflow prevention", "range shift", "mathematical equivalence"]
    )


def _graded_service():
    """A service wired for graded mode with a stand-in claude binary (subprocess is mocked)."""
    service = LLMService(api_key="test-key")
    service.claude_bin = "/dummy/claude"
    service.use_mock = False  # hermetic: ignore any MASTERY_GRADER_MOCK in the outer env
    return service


def _proc(stdout="", returncode=0, stderr=""):
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = stderr
    return m


class TestLLMServiceInit:
    """Test cases for LLMService initialization."""

    def test_init_with_api_key(self):
        """Default model + (new) 120s timeout; OpenAI client retained for generate_completion."""
        service = LLMService(api_key="test-key-123")
        assert service.model == "gpt-4o-mini"
        assert service.timeout == 120
        assert service.client is not None

    def test_init_with_env_var(self, monkeypatch):
        """Should load the OpenAI key from env for the generate_completion client."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-456")
        service = LLMService()
        assert service.client is not None

    def test_init_no_openai_key_is_failclosed_not_mock(self, monkeypatch):
        """No OpenAI key => client None, but grader is NOT auto-mock (fail-closed by default)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MASTERY_GRADER_MOCK", raising=False)
        service = LLMService()
        assert service.client is None
        assert service.use_mock is False

    def test_init_grader_mock_env_enables_mock(self, monkeypatch):
        """Mock mode is now EXPLICIT via MASTERY_GRADER_MOCK=1."""
        monkeypatch.setenv("MASTERY_GRADER_MOCK", "1")
        service = LLMService(api_key="test-key")
        assert service.use_mock is True

    def test_init_custom_model_and_timeout(self):
        service = LLMService(api_key="test-key", model="gpt-4", timeout=60)
        assert service.model == "gpt-4"
        assert service.timeout == 60


class TestEvaluateJustification:
    """Test cases for LLMService.evaluate_justification() over the claude -p seam."""

    def test_evaluate_correct_answer(self, sample_question):
        service = _graded_service()
        out = json.dumps({"is_correct": True, "feedback": "Excellent! Key mechanism identified."})
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout=out)) as run:
            result = service.evaluate_justification(
                sample_question,
                "The subtract-max trick prevents overflow by shifting values to (-inf, 0]"
            )
        assert result.is_correct is True
        assert "Excellent" in result.feedback
        # Invoked the resolved claude binary with -p, prompt on stdin.
        assert run.call_args.args[0] == ["/dummy/claude", "-p"]
        assert "Why is the subtract-max" in run.call_args.kwargs["input"]

    def test_evaluate_incorrect_answer(self, sample_question):
        service = _graded_service()
        out = json.dumps({"is_correct": False, "feedback": "Explain the specific numerical problem."})
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout=out)):
            result = service.evaluate_justification(sample_question, "It makes the code more stable.")
        assert result.is_correct is False
        assert "numerical problem" in result.feedback

    def test_evaluate_tolerates_prose_preamble(self, sample_question):
        """claude often adds prose; _extract_json must still recover the object."""
        service = _graded_service()
        out = 'Here is my evaluation:\n{"is_correct": true, "feedback": "ok"} \nDone.'
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout=out)):
            result = service.evaluate_justification(sample_question, "answer")
        assert result.is_correct is True

    def test_evaluate_tolerates_code_fence(self, sample_question):
        service = _graded_service()
        out = '```json\n{"is_correct": false, "feedback": "hint"}\n```'
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout=out)):
            result = service.evaluate_justification(sample_question, "answer")
        assert result.is_correct is False

    def test_evaluate_empty_output_raises(self, sample_question):
        service = _graded_service()
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout="")):
            with pytest.raises(LLMResponseError, match="empty output"):
                service.evaluate_justification(sample_question, "answer")

    def test_evaluate_no_json_raises(self, sample_question):
        service = _graded_service()
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout="I cannot.")):
            with pytest.raises(LLMResponseError, match="no parseable JSON"):
                service.evaluate_justification(sample_question, "answer")

    def test_evaluate_invalid_schema_raises(self, sample_question):
        service = _graded_service()
        out = json.dumps({"is_correct": True})  # missing required 'feedback'
        with patch("engine.services.llm_service.subprocess.run", return_value=_proc(stdout=out)):
            with pytest.raises(LLMResponseError, match="does not match expected schema"):
                service.evaluate_justification(sample_question, "answer")

    def test_evaluate_nonzero_exit_raises(self, sample_question):
        service = _graded_service()
        with patch("engine.services.llm_service.subprocess.run",
                   return_value=_proc(returncode=1, stderr="boom")):
            with pytest.raises(LLMAPIError, match="exited 1"):
                service.evaluate_justification(sample_question, "answer")

    def test_evaluate_timeout_raises(self, sample_question):
        service = _graded_service()
        with patch("engine.services.llm_service.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=120)):
            with pytest.raises(LLMAPIError, match="timed out"):
                service.evaluate_justification(sample_question, "answer")

    def test_evaluate_fail_closed_when_no_binary(self, sample_question):
        """No grader engine => FAIL CLOSED (raise), never silently pass."""
        service = LLMService(api_key="test-key")
        service.claude_bin = None
        with pytest.raises(LLMAPIError, match="grader unavailable"):
            service.evaluate_justification(sample_question, "answer")

    def test_evaluate_mock_mode_autopasses(self, sample_question, monkeypatch):
        """Explicit demo mock (MASTERY_GRADER_MOCK=1) auto-passes with a clear marker."""
        monkeypatch.setenv("MASTERY_GRADER_MOCK", "1")
        service = LLMService(api_key="test-key")
        result = service.evaluate_justification(sample_question, "anything")
        assert result.is_correct is True
        assert "MOCK MODE" in result.feedback


class TestExtractJson:
    """Test cases for the prose/fence-tolerant JSON extractor."""

    def test_plain_object(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_prose_wrapped(self):
        assert _extract_json('text {"a": 1} more') == '{"a": 1}'

    def test_fenced(self):
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_json_returns_none(self):
        assert _extract_json("no object here") is None


class TestBuildCOTPrompt:
    """Test cases for Chain-of-Thought prompt construction."""

    def test_build_cot_prompt_includes_all_elements(self, sample_question):
        service = LLMService(api_key="test-key")
        prompt = service._build_cot_prompt(sample_question, "User's answer here")
        assert sample_question.question in prompt
        assert "User's answer here" in prompt
        assert sample_question.model_answer in prompt
        assert "overflow prevention" in prompt
        assert "range shift" in prompt
        assert "mathematical equivalence" in prompt
        assert "chain-of-thought" in prompt.lower()
        assert "JSON" in prompt

    def test_build_cot_prompt_formats_required_concepts(self, sample_question):
        service = LLMService(api_key="test-key")
        prompt = service._build_cot_prompt(sample_question, "Answer")
        assert "- overflow prevention" in prompt
        assert "- range shift" in prompt
