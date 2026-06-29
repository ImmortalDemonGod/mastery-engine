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

import json
import logging
import os
import re
import subprocess
from typing import Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError
from pydantic import ValidationError

from engine.schemas import JustifyQuestion, LLMEvaluationResponse


logger = logging.getLogger(__name__)


# Justify grader provider hooks (Cycle-2: real `claude -p`, NOT the OpenRouter free shim).
GRADER_MOCK_ENV = "MASTERY_GRADER_MOCK"   # "1" => explicit demo auto-pass
CLAUDE_BIN_ENV = "MASTERY_CLAUDE_BIN"     # override path to the real claude CLI


def _resolve_claude_bin() -> Optional[str]:
    """Locate the REAL Anthropic `claude` CLI from TRUSTED locations only.

    Order: $MASTERY_CLAUDE_BIN -> ~/.local/bin/claude. We deliberately do NOT
    fall back to `shutil.which("claude")`: that could resolve a wrapper/shim on
    PATH (e.g. an OpenRouter free-model shim) and silently send grading to the
    wrong backend, violating the "real claude -p only" contract.
    """
    env_bin = os.getenv(CLAUDE_BIN_ENV)
    if env_bin and os.path.exists(env_bin):
        return env_bin
    default = os.path.expanduser("~/.local/bin/claude")
    if os.path.exists(default):
        return default
    return None


def _extract_json(text: str) -> Optional[str]:
    """Extract the FIRST valid JSON object from an LLM text response (tolerates prose/fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Scan each '{' and try to decode a full object there; return the first that parses.
    # (A first-brace-to-last-brace slice breaks when prose contains extra braces.)
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            _, end = decoder.raw_decode(text[idx:])
            return text[idx:idx + end]
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
    return None


class LLMService:
    """
    Service for evaluating user justifications using OpenAI's API.
    
    Uses Chain-of-Thought prompting to analyze user answers, compare them
    to model answers, and generate Socratic feedback.
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TIMEOUT = 120  # seconds (a `claude -p` subprocess can be slow)
    
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
        self.model = model
        self.timeout = timeout

        # Justify grader runs on the REAL `claude -p` (judgment-critical; Cycle-2 lock).
        self.claude_bin = _resolve_claude_bin()
        # Demo auto-pass is now EXPLICIT-only (was: any missing OpenAI key auto-passed → fail-open).
        self.use_mock = os.getenv(GRADER_MOCK_ENV) == "1"

        # OpenAI client retained ONLY for generate_completion() (create-bug / Harden authoring).
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key, timeout=timeout)
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize OpenAI client: {e}") from e
        else:
            self.client = None

        if self.use_mock:
            logger.warning(f"⚠️  {GRADER_MOCK_ENV}=1 — Justify grader auto-passes (demo mode).")
        elif not self.claude_bin:
            logger.warning(
                "No `claude` binary found — Justify grader will FAIL CLOSED. "
                f"Set {CLAUDE_BIN_ENV}, or {GRADER_MOCK_ENV}=1 for demo."
            )
        else:
            logger.info(f"Justify grader: claude -p at {self.claude_bin} (timeout={timeout}s)")
    
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
        # Explicit demo mode only (MASTERY_GRADER_MOCK=1): auto-pass with a clear marker.
        if self.use_mock:
            logger.info(f"[GRADER MOCK] Auto-passing justify question '{question.id}'")
            concepts = ", ".join(question.required_concepts[:3])
            return LLMEvaluationResponse(
                is_correct=True,
                feedback=(
                    f"MOCK MODE ({GRADER_MOCK_ENV}=1): auto-passed for demo. "
                    f"Real grading would assess concepts: {concepts}."
                ),
            )

        # Graded mode: FAIL CLOSED if the grader engine is unavailable (never silently pass).
        if not self.claude_bin:
            raise LLMAPIError(
                "Justify grader unavailable: no `claude` binary found. "
                f"Set {CLAUDE_BIN_ENV} to the real claude CLI, or {GRADER_MOCK_ENV}=1 for demo."
            )
        
        # Construct the (provider-agnostic) Chain-of-Thought prompt and grade via claude -p.
        prompt = self._build_cot_prompt(question, user_answer)
        logger.debug(f"Grading question '{question.id}' via claude -p")
        raw = self._run_claude(prompt)

        payload = _extract_json(raw)
        if payload is None:
            raise LLMResponseError(f"Grader returned no parseable JSON. Output head: {raw[:160]!r}")
        try:
            evaluation = LLMEvaluationResponse.model_validate_json(payload)
        except ValidationError as e:
            raise LLMResponseError(f"Grader JSON does not match expected schema: {e}") from e
        logger.info(f"Evaluation complete for question '{question.id}': "
                    f"is_correct={evaluation.is_correct}")
        return evaluation

    def _run_claude(self, prompt: str) -> str:
        """Invoke the real `claude -p` with the prompt on stdin; return stdout.

        Fails CLOSED: any non-zero exit, timeout, or empty output raises (never auto-passes).
        """
        system = (
            "You are an expert technical educator grading a student's conceptual answer. "
            "Reason step by step, then output ONLY a JSON object: "
            '{"is_correct": true|false, "feedback": "..."} - no prose, no code fences.'
        )
        full_prompt = f"{system}\n\n{prompt}"
        try:
            proc = subprocess.run(
                [self.claude_bin, "-p"],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise LLMAPIError(f"Grader timed out after {self.timeout}s") from e
        except Exception as e:
            raise LLMAPIError(f"Grader subprocess failed to launch: {e}") from e
        if proc.returncode != 0:
            raise LLMAPIError(f"Grader exited {proc.returncode}: {(proc.stderr or '')[:200]}")
        out = (proc.stdout or "").strip()
        if not out:
            raise LLMResponseError("Grader returned empty output")
        return out
    
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
        # Placeholder only when no OpenAI client (this path uses OpenAI, not the grader).
        # NOT gated on self.use_mock: the grader demo flag must not disable authoring flows.
        if self.client is None:
            logger.warning("[NO OPENAI CLIENT] generate_completion has no key - returning placeholder")
            return "MOCK_RESPONSE: LLM generation not available without OPENAI_API_KEY"
        
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
