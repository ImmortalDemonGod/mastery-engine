"""
Justify stage runner for the Mastery Engine (STUB).

The JustifyRunner implements the "Justify" stage of the BJH pedagogy loop.
It challenges users to explain their implementation decisions through
Socratic questioning.

This is currently a STUB implementation. The real implementation will integrate
LLM-powered evaluation of user responses. For now, it accepts any non-empty answer.

Key responsibilities:
- Load and present justify questions from curriculum
- Collect user responses (stub: accept any non-empty answer)
- Evaluate responses (stub: always accept)
- Future: LLM evaluation with Chain-of-Thought reasoning
"""

import logging
import json
import os
from pathlib import Path

from engine.schemas import JustifyQuestion
from engine.curriculum import CurriculumManager


logger = logging.getLogger(__name__)


class JustifyRunner:
    """
    Manages the Justify stage workflow (STUB).
    
    The Justify stage tests conceptual understanding by asking users
    to explain their implementation choices. This stub version simply
    ensures the state machine is complete.
    """
    
    def __init__(self, curriculum_manager: CurriculumManager):
        """
        Initialize justify runner.
        
        Args:
            curriculum_manager: For accessing justify questions
        """
        self.curriculum_mgr = curriculum_manager
    
    def load_questions(self, curriculum_id: str, module) -> list[JustifyQuestion]:
        """
        Load justify questions for a module.
        
        Args:
            curriculum_id: ID of the current curriculum
            module: Module metadata
            
        Returns:
            List of JustifyQuestion objects
            
        Raises:
            JustifyQuestionsError: If questions file is missing or malformed
        """
        questions_path = self.curriculum_mgr.get_justify_questions_path(curriculum_id, module)
        
        if not questions_path.exists():
            raise JustifyQuestionsError(
                f"Justify questions not found: {questions_path}. "
                "This is a curriculum configuration error."
            )
        
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = [JustifyQuestion.model_validate(q) for q in data]
            logger.info(f"Loaded {len(questions)} justify questions for module '{module.id}'")
            return questions
            
        except json.JSONDecodeError as e:
            raise JustifyQuestionsError(f"Malformed JSON in questions file: {e}") from e
        except Exception as e:
            raise JustifyQuestionsError(f"Failed to load justify questions: {e}") from e
    
    def check_fast_filter(self, question: JustifyQuestion, user_answer: str) -> tuple[bool, str | None]:
        """
        Fast keyword-based filter for common failure modes.
        
        This is the first layer of the validation chain. It performs simple,
        local keyword matching against pre-defined failure modes to catch
        shallow/vague answers without calling the LLM.
        
        Can be disabled via environment variable: MASTERY_DISABLE_FAST_FILTER=true
        
        Args:
            question: The justify question with failure modes
            user_answer: User's response text (case-insensitive matching)
            
        Returns:
            Tuple of (matched, feedback):
            - matched: True if a failure mode was detected
            - feedback: Pre-written feedback for the matched failure mode, or None
        """
        # Check if fast filter is disabled via environment variable (for debugging)
        if os.getenv('MASTERY_DISABLE_FAST_FILTER', '').lower() in ('true', '1', 'yes'):
            logger.info("Fast filter DISABLED via MASTERY_DISABLE_FAST_FILTER environment variable")
            return False, None
        
        user_answer_lower = user_answer.lower()
        
        for failure_mode in question.failure_modes:
            # Check if any keyword from this failure mode appears in the answer
            for keyword in failure_mode.keywords:
                if keyword.lower() in user_answer_lower:
                    logger.info(f"Fast filter matched: category='{failure_mode.category}', "
                               f"keyword='{keyword}'")
                    return True, failure_mode.feedback
        
        # No match found
        logger.info("No fast filter match, proceeding to LLM evaluation")
        return False, None


class JustifyQuestionsError(Exception):
    """Raised when justify questions cannot be loaded."""
    pass
