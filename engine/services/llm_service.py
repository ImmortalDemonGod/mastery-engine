"""
LLM Service for the Mastery Engine.

The LLMService provides a robust, abstracted interface for interacting with
the OpenAI API to perform Chain-of-Thought evaluations of user justifications.

Key design principles:
- Fail fast with clear error messages for configuration issues
- Use JSON mode to guarantee parsable responses
- Comprehensive error handling for network/API failures
- Cost-effective model selection (gpt-4o-mini default)
"""

import logging
import os
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from pydantic import ValidationError

from engine.schemas import JustifyQuestion, LLMEvaluationResponse


logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for evaluating user justifications using OpenAI's API.
    
    Uses Chain-of-Thought prompting to analyze user answers, compare them
    to model answers, and generate Socratic feedback.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TIMEOUT = 30  # seconds
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key. If None, loads from OPENAI_API_KEY env var.
            model: Model to use for evaluation (default: gpt-4o-mini for cost efficiency)
            timeout: API request timeout in seconds
            
        Raises:
            ConfigurationError: If API key is missing or invalid
        """
        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        # Enable mock mode if no API key (for demo/portfolio viewing)
        if not api_key:
            self.use_mock = True
            self.client = None
            self.model = model
            self.timeout = timeout
            logger.warning(
                "⚠️  No OpenAI API key found. LLMService operating in MOCK mode.\n"
                "   Justify stage will auto-pass with simulated feedback.\n"
                "   Set OPENAI_API_KEY environment variable for production use.\n"
                "   Get a key from: https://platform.openai.com/api-keys"
            )
            return
        
        self.use_mock = False
        self.model = model
        self.timeout = timeout
        
        try:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
            logger.info(f"LLMService initialized with model={model}, timeout={timeout}s")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}") from e
    
    def evaluate_justification(
        self,
        question: JustifyQuestion,
        user_answer: str
    ) -> LLMEvaluationResponse:
        """
        Evaluate a user's answer to a justify question using Chain-of-Thought reasoning.
        
        This method constructs a multi-step prompt that instructs the LLM to:
        1. Analyze the user's answer for key concepts
        2. Compare to the model answer
        3. Classify as correct/incorrect
        4. Generate appropriate Socratic feedback
        
        Args:
            question: The justify question with model answer and rubric
            user_answer: The user's response
            
        Returns:
            LLMEvaluationResponse with is_correct flag and feedback
            
        Raises:
            LLMResponseError: If response is malformed or unparsable
            LLMAPIError: If API call fails (network, auth, rate limit, etc.)
        """
        # Mock mode: auto-pass for demo/portfolio viewing
        if self.use_mock:
            logger.info(f"[MOCK MODE] Auto-passing justify question '{question.id}'")
            return LLMEvaluationResponse(
                is_correct=True,
                feedback=(
                    "🎭 MOCK MODE: No OpenAI API key detected.\n\n"
                    "In production, GPT-4o would evaluate your response against:\n"
                    f"- Model Answer: {question.model_answer[:100]}...\n"
                    f"- Required Concepts: {', '.join(question.required_concepts[:3])}\n"
                    f"- Failure Modes: {', '.join(question.failure_modes[:3])}\n\n"
                    "This step is auto-passed for demonstration purposes.\n"
                    "Set OPENAI_API_KEY to enable real LLM evaluation."
                )
            )
        
        try:
            # Construct Chain-of-Thought prompt
            prompt = self._build_cot_prompt(question, user_answer)
            
            logger.debug(f"Sending evaluation request for question '{question.id}'")
            
            # Make API call with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert technical educator evaluating student understanding. "
                            "Analyze the student's answer using chain-of-thought reasoning, then provide "
                            "a structured evaluation in JSON format."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},  # Enforce JSON mode
                temperature=0.3,  # Lower temperature for consistent evaluation
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            if not content:
                raise LLMResponseError("LLM returned empty response")
            
            logger.debug(f"Received LLM response: {content[:100]}...")
            
            # Parse and validate response
            try:
                evaluation = LLMEvaluationResponse.model_validate_json(content)
                logger.info(f"Evaluation complete for question '{question.id}': "
                           f"is_correct={evaluation.is_correct}")
                return evaluation
            except ValidationError as e:
                raise LLMResponseError(f"LLM response does not match expected schema: {e}") from e
        
        except LLMResponseError:
            # Re-raise response errors without wrapping
            raise
        except AuthenticationError as e:
            raise LLMAPIError(
                "Authentication failed. Please check your OpenAI API key."
            ) from e
        except RateLimitError as e:
            raise LLMAPIError(
                "Rate limit exceeded. Please try again later or upgrade your OpenAI plan."
            ) from e
        except APIConnectionError as e:
            raise LLMAPIError(
                f"Network error connecting to OpenAI API: {e}"
            ) from e
        except APIError as e:
            raise LLMAPIError(
                f"OpenAI API error: {e}"
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            logger.exception(f"Unexpected error during LLM evaluation: {e}")
            raise LLMAPIError(f"Unexpected error during evaluation: {e}") from e
    
    def generate_completion(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format = None
    ) -> str:
        """
        Generate a completion for a given prompt.
        
        Generic method for LLM generation tasks (e.g., bug authoring).
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
            
        Raises:
            LLMAPIError: If API call fails
        """
        # Mock mode: return placeholder
        if self.use_mock:
            logger.warning("[MOCK MODE] generate_completion called without API key - returning placeholder")
            return "MOCK_RESPONSE: LLM generation not available in mock mode"
        
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            # Use Structured Outputs if response_format is a Pydantic model
            if response_format and hasattr(response_format, '__mro__'):
                # This is a Pydantic model - use .parse() for Structured Outputs
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
                # Return the JSON string from the parsed response
                if response.choices[0].message.parsed:
                    return response.choices[0].message.parsed.model_dump_json(indent=2)
                else:
                    # Handle refusal
                    if response.choices[0].message.refusal:
                        raise LLMAPIError(f"Model refused: {response.choices[0].message.refusal}")
                    return response.choices[0].message.content
            else:
                # Regular generation without structured outputs
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
        except AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise LLMAPIError(f"Authentication error: {e}") from e
        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise LLMAPIError(f"Rate limit error: {e}") from e
        except APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise LLMAPIError(f"Connection error: {e}") from e
        except APIError as e:
            logger.error(f"API error: {e}")
            raise LLMAPIError(f"API error: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during LLM generation")
            raise LLMAPIError(f"Unexpected API error: {e}") from e
    
    def _build_cot_prompt(self, question: JustifyQuestion, user_answer: str) -> str:
        """
        Build Chain-of-Thought prompt for justification evaluation.
        
        The prompt instructs the LLM to:
        1. Identify key concepts in user's answer
        2. Compare to required concepts from rubric
        3. Check for conceptual errors or misconceptions
        4. Make a binary decision (correct/incorrect)
        5. Generate Socratic feedback (acceptance or hint)
        
        Args:
            question: Justify question with model answer and required concepts
            user_answer: User's response
            
        Returns:
            Formatted prompt string
        """
        # Format required concepts as bullet list
        required_concepts_str = "\n".join(f"- {concept}" for concept in question.required_concepts)
        
        prompt = f"""You are evaluating a student's answer to a technical question. Use chain-of-thought reasoning to determine if they demonstrate sufficient understanding.

**Question:**
{question.question}

**Student's Answer:**
{user_answer}

**Model Answer (for reference):**
{question.model_answer}

**Required Concepts:**
The student's answer must demonstrate understanding of these concepts:
{required_concepts_str}

**Evaluation Instructions:**
1. **Identify Concepts**: What key concepts did the student mention in their answer?
2. **Compare to Required**: Do they demonstrate understanding of the required concepts?
3. **Check for Errors**: Are there any conceptual errors or misconceptions?
4. **Decision**: Based on the above, is the answer correct (demonstrates sufficient understanding)?
5. **Feedback**: 
   - If CORRECT: Write a brief acceptance message acknowledging their understanding
   - If INCORRECT: Write a Socratic hint that guides them toward the missing concept WITHOUT giving away the answer

**Output Format:**
You MUST respond with valid JSON in this exact format:
{{
    "is_correct": true or false,
    "feedback": "Your feedback message here"
}}

Think step-by-step, then provide your final evaluation in JSON format."""
        
        return prompt


class ConfigurationError(Exception):
    """Raised when LLM service configuration is invalid (e.g., missing API key)."""
    pass


class LLMResponseError(Exception):
    """Raised when LLM response is malformed or doesn't match expected schema."""
    pass


class LLMAPIError(Exception):
    """Raised when OpenAI API call fails (auth, network, rate limit, etc.)."""
    pass
