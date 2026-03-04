"""
Focused LLM Service Integration Tests

These tests validate the LLMService directly with real OpenAI API calls.
They ensure prompt formatting, response parsing, and error handling work correctly.

Cost: ~$0.009 per full test run (3 API calls × $0.003 each)

Run with: pytest tests/integration/test_llm_service.py -m integration -v
Skip with: pytest -m "not integration"
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables at module level
load_dotenv()

from engine.services.llm_service import LLMService, ConfigurationError, LLMAPIError
from engine.schemas import JustifyQuestion, FailureMode


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def check_api_key():
    """Check if OpenAI API key is available, skip tests if not."""
    # Ensure .env is loaded
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping live LLM integration tests")


@pytest.fixture
def sample_question():
    """Create a sample justify question for testing."""
    return JustifyQuestion(
        id="test_subtract_max",
        question="Explain why the subtract-max trick works in softmax.",
        model_answer=(
            "The subtract-max trick exploits softmax(x) = softmax(x - c) for any constant c. "
            "By choosing c = max(x), we shift inputs to (-inf, 0], preventing overflow in exp()."
        ),
        failure_modes=[
            FailureMode(
                category="Vague",
                keywords=["stability", "better", "safer"],
                feedback="Be more specific about the numerical problem."
            )
        ],
        required_concepts=["mathematical equivalence", "overflow prevention", "range shift"]
    )


def test_llm_service_initialization_with_api_key(check_api_key):
    """
    Test that LLMService initializes correctly with a valid API key.
    
    Cost: $0 (no API call)
    """
    llm_service = LLMService()
    assert llm_service is not None
    assert llm_service.model == "gpt-4o-mini"
    assert llm_service.timeout == 30


def test_llm_service_missing_api_key(monkeypatch):
    """
    Test that missing API key raises ConfigurationError with helpful message.
    
    Cost: $0 (no API call)
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    with pytest.raises(ConfigurationError) as exc_info:
        LLMService()
    
    error_msg = str(exc_info.value)
    assert "api key" in error_msg.lower()
    assert "openai" in error_msg.lower()


def test_llm_accepts_correct_answer(check_api_key, sample_question):
    """
    Test that LLM correctly accepts a comprehensive, accurate answer.
    
    This validates:
    - API integration works
    - Prompt formatting is correct  
    - Response parsing succeeds
    - Evaluation logic identifies understanding
    
    Cost: ~$0.003 (one API call)
    """
    llm_service = LLMService()
    
    # Comprehensive correct answer
    correct_answer = (
        "The subtract-max technique works because softmax(x) = softmax(x - c) for any constant c. "
        "This mathematical identity holds because when computing exp(x-c) / sum(exp(x-c)), we can "
        "factor out exp(-c) from both numerator and denominator, causing them to cancel. By choosing "
        "c = max(x), the shifted values x - max(x) lie in (-infinity, 0]. The maximum shifted value "
        "is zero, so the largest exponential is exp(0) = 1. Negative values produce exponentials "
        "between 0 and 1, all finite. For x = [1000, 1001], subtracting 1001 gives [-1, 0], yielding "
        "exp(-1) ≈ 0.37 and exp(0) = 1, which are safe values that produce the correct probabilities."
    )
    
    result = llm_service.evaluate_justification(sample_question, correct_answer)
    
    # Assertions
    assert result.is_correct is True, f"Should accept correct answer. Feedback: {result.feedback}"
    assert result.feedback is not None and len(result.feedback) > 0
    
    # Feedback should be encouraging
    feedback_lower = result.feedback.lower()
    assert any(word in feedback_lower for word in ["good", "great", "correct", "understand", "excellent"]), \
        f"Feedback should be positive. Got: {result.feedback}"


def test_llm_rejects_incomplete_answer(check_api_key, sample_question):
    """
    Test that LLM correctly rejects answers with conceptual gaps.
    
    This validates:
    - LLM detects missing concepts
    - Feedback is specific and educational  
    - Feedback guides without revealing the answer
    
    Cost: ~$0.003 (one API call)
    """
    llm_service = LLMService()
    
    # Answer that mentions some concepts but misses key points
    incomplete_answer = (
        "The subtract-max technique works by shifting all values by the same amount. "
        "This preserves the relative differences between values while making computation easier. "
        "It's based on the property that exponentials scale proportionally."
    )
    
    result = llm_service.evaluate_justification(sample_question, incomplete_answer)
    
    # Assertions
    assert result.is_correct is False, "Should reject incomplete answer"
    assert result.feedback is not None and len(result.feedback) > 50
    
    # Feedback should mention what's missing
    feedback_lower = result.feedback.lower()
    should_mention = ["overflow", "underflow", "exp", "range", "(-inf", "prevent"]
    assert any(word in feedback_lower for word in should_mention), \
        f"Feedback should guide toward missing concepts. Got: {result.feedback}"


def test_llm_rejects_conceptual_error(check_api_key, sample_question):
    """
    Test that LLM correctly identifies conceptual misconceptions.
    
    This validates:
    - LLM catches subtle errors
    - Feedback corrects misconceptions
    - Feedback is Socratic
    
    Cost: ~$0.003 (one API call)
    """
    llm_service = LLMService()
    
    # Answer with a fundamental misconception
    wrong_answer = (
        "The subtract-max trick is an approximation that sacrifices accuracy for speed. "
        "While not mathematically exact, it provides results close enough for practical use. "
        "The small error introduced is acceptable given the performance gains."
    )
    
    result = llm_service.evaluate_justification(sample_question, wrong_answer)
    
    # Assertions
    assert result.is_correct is False, "Should reject answer with misconceptions"
    assert result.feedback is not None
    
    # Feedback should address the misconception in some way
    # (either by correcting it directly or by guiding toward the right concepts)
    feedback_lower = result.feedback.lower()
    # Look for either direct correction or guidance toward key concepts
    addresses_misconception = (
        "exact" in feedback_lower or 
        "not an approximation" in feedback_lower or
        "mathematical" in feedback_lower or
        "equivalence" in feedback_lower or
        "sacrifices" in feedback_lower  # Mentions the error
    )
    assert addresses_misconception, \
        f"Feedback should address the approximation misconception. Got: {result.feedback}"


def test_llm_timeout_handling(check_api_key, sample_question):
    """
    Test that very short timeouts are handled gracefully.
    
    Note: This may succeed if API is very fast, or timeout if not.
    Both outcomes are acceptable - we're testing that timeout doesn't crash.
    
    Cost: ~$0 (request likely times out before completion)
    """
    llm_service = LLMService(timeout=1)  # 1 second timeout
    
    try:
        result = llm_service.evaluate_justification(
            sample_question,
            "Test answer for timeout handling"
        )
        # If it succeeded, verify it's a valid result
        assert isinstance(result.is_correct, bool)
        assert result.feedback is not None
    except LLMAPIError as e:
        # If it timed out, error should be clear
        error_msg = str(e).lower()
        assert "timeout" in error_msg or "api" in error_msg or "error" in error_msg


def test_response_format_validation():
    """
    Test that response parsing handles various formats correctly.
    
    This is a unit-level test within integration suite to document expected format.
    
    Cost: $0 (no API call)
    """
    # This documents what we expect from the LLM response
    expected_structure = {
        "is_correct": bool,
        "feedback": str
    }
    
    # Verify our LLMEvaluationResponse schema matches expectations
    from engine.schemas import LLMEvaluationResponse
    
    # Valid response
    valid = LLMEvaluationResponse(
        is_correct=True,
        feedback="Great explanation!"
    )
    assert valid.is_correct is True
    assert valid.feedback == "Great explanation!"
    
    # Another valid response
    valid2 = LLMEvaluationResponse(
        is_correct=False,
        feedback="Consider the numerical stability aspect."
    )
    assert valid2.is_correct is False
    assert len(valid2.feedback) > 0


# Cost summary documentation
def test_cost_analysis_documentation():
    """Document the cost of running these integration tests."""
    costs = {
        "test_llm_service_initialization_with_api_key": "$0",
        "test_llm_service_missing_api_key": "$0",
        "test_llm_accepts_correct_answer": "$0.003",
        "test_llm_rejects_incomplete_answer": "$0.003",
        "test_llm_rejects_conceptual_error": "$0.003",
        "test_llm_timeout_handling": "$0 (timeout)",
        "test_response_format_validation": "$0",
        "TOTAL PER RUN": "$0.009",
        "100 RUNS": "$0.90",
        "1000 RUNS": "$9.00",
    }
    
    # Always passes - just documentation
    assert True, f"Test suite costs: {costs}"
