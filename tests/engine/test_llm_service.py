"""
Unit tests for engine.services.llm_service.LLMService.

These tests achieve 100% coverage on the LLMService using pytest-mock
to simulate OpenAI API responses without making live API calls.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError

from engine.services.llm_service import (
    LLMService,
    ConfigurationError,
    LLMResponseError,
    LLMAPIError
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


class TestLLMServiceInit:
    """Test cases for LLMService initialization."""
    
    def test_init_with_api_key(self):
        """Should initialize successfully with provided API key."""
        service = LLMService(api_key="test-key-123")
        assert service.model == "gpt-4o-mini"
        assert service.timeout == 30
    
    def test_init_with_env_var(self, monkeypatch):
        """Should load API key from environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-456")
        service = LLMService()
        assert service.client is not None
    
    def test_init_missing_api_key_raises_error(self, monkeypatch):
        """Should raise ConfigurationError if API key is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ConfigurationError, match="API key not found"):
            LLMService()
    
    def test_init_custom_model_and_timeout(self):
        """Should accept custom model and timeout parameters."""
        service = LLMService(api_key="test-key", model="gpt-4", timeout=60)
        assert service.model == "gpt-4"
        assert service.timeout == 60


class TestEvaluateJustification:
    """Test cases for LLMService.evaluate_justification()."""
    
    def test_evaluate_correct_answer(self, sample_question):
        """Should correctly evaluate a correct answer."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "is_correct": True,
                        "feedback": "Excellent! You've identified the key mechanism."
                    })
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Patch OpenAI constructor to return our mock
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            result = service.evaluate_justification(
                sample_question,
                "The subtract-max trick prevents overflow by shifting values to (-inf, 0]"
            )
        
        assert result.is_correct is True
        assert "Excellent" in result.feedback
        
        # Verify API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs['model'] == 'gpt-4o-mini'
        assert call_kwargs['response_format'] == {'type': 'json_object'}
        assert call_kwargs['temperature'] == 0.3
    
    def test_evaluate_incorrect_answer(self, sample_question):
        """Should correctly evaluate an incorrect answer."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "is_correct": False,
                        "feedback": "Can you explain the specific numerical problem this solves?"
                    })
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            result = service.evaluate_justification(
                sample_question,
                "It makes the code more stable."
            )
        
        assert result.is_correct is False
        assert "numerical problem" in result.feedback
    
    def test_evaluate_empty_response_raises_error(self, sample_question):
        """Should raise LLMResponseError if LLM returns empty content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMResponseError, match="empty response"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_malformed_json_raises_error(self, sample_question):
        """Should raise LLMResponseError if response is malformed JSON."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Not valid JSON"))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMResponseError):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_invalid_schema_raises_error(self, sample_question):
        """Should raise LLMResponseError if response doesn't match schema."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Missing required 'feedback' field
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({"is_correct": True})
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMResponseError, match="does not match expected schema"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_authentication_error(self, sample_question):
        """Should raise LLMAPIError on authentication failure."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=MagicMock(status_code=401),
            body=None
        )
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMAPIError, match="Authentication failed"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_rate_limit_error(self, sample_question):
        """Should raise LLMAPIError on rate limit exceeded."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None
        )
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMAPIError, match="Rate limit exceeded"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_connection_error(self, sample_question):
        """Should raise LLMAPIError on network connection failure."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMAPIError, match="Network error"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_api_error(self, sample_question):
        """Should raise LLMAPIError on generic API error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            "Internal server error",
            request=MagicMock(),
            body=None
        )
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMAPIError, match="OpenAI API error"):
                service.evaluate_justification(sample_question, "Some answer")
    
    def test_evaluate_unexpected_error(self, sample_question):
        """Should raise LLMAPIError on unexpected exceptions."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Unexpected!")
        
        with patch('engine.services.llm_service.OpenAI', return_value=mock_client):
            service = LLMService(api_key="test-key")
            
            with pytest.raises(LLMAPIError, match="Unexpected error"):
                service.evaluate_justification(sample_question, "Some answer")


class TestBuildCOTPrompt:
    """Test cases for Chain-of-Thought prompt construction."""
    
    def test_build_cot_prompt_includes_all_elements(self, sample_question):
        """Should include question, model answer, and required concepts."""
        service = LLMService(api_key="test-key")
        prompt = service._build_cot_prompt(sample_question, "User's answer here")
        
        # Verify all key elements are present
        assert sample_question.question in prompt
        assert "User's answer here" in prompt
        assert sample_question.model_answer in prompt
        assert "overflow prevention" in prompt
        assert "range shift" in prompt
        assert "mathematical equivalence" in prompt
        assert "chain-of-thought" in prompt.lower()
        assert "JSON" in prompt
    
    def test_build_cot_prompt_formats_required_concepts(self, sample_question):
        """Should format required concepts as bullet list."""
        service = LLMService(api_key="test-key")
        prompt = service._build_cot_prompt(sample_question, "Answer")
        
        # Check that concepts are formatted with bullet points
        assert "- overflow prevention" in prompt
        assert "- range shift" in prompt
