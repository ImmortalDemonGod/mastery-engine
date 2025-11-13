"""
Pydantic data schemas for the Mastery Engine.

This module defines all data contracts between the engine core and curriculum content.
Using Pydantic provides:
- Runtime validation of curriculum and state files
- Type safety and IDE autocompletion
- Clear API contracts for content authors
- Automatic JSON serialization/deserialization
"""

from typing import Optional
from pydantic import BaseModel, Field


class ModuleMetadata(BaseModel):
    """
    Metadata for a single curriculum module.
    
    Attributes:
        id: Unique identifier for the module (e.g., "rmsnorm", "multihead_attention")
        name: Human-readable display name (e.g., "RMS Normalization")
        path: Relative path to module directory within curriculum pack
        baseline_perf_seconds: Optional performance baseline from Phase 0 CI for novelty detection
    """
    id: str
    name: str
    path: str
    baseline_perf_seconds: Optional[float] = None


class CurriculumManifest(BaseModel):
    """
    Root schema for a curriculum pack's manifest.json file.
    
    Attributes:
        curriculum_name: Unique identifier for the curriculum pack (e.g., "cs336_a1")
        author: Creator/maintainer of the curriculum
        version: Semantic version string (e.g., "1.0.0")
        modules: Ordered list of modules in the curriculum
    """
    curriculum_name: str
    author: str
    version: str
    modules: list[ModuleMetadata]


class UserProgress(BaseModel):
    """
    Schema for .mastery_progress.json tracking user's learning state.
    
    Attributes:
        curriculum_id: ID of the active curriculum pack
        current_module_index: Index into curriculum's modules list (0-based)
        current_stage: Current stage within the BJH loop
        completed_modules: List of module IDs the user has fully completed
    """
    curriculum_id: str
    current_module_index: int = 0
    current_stage: str = "build"  # "build", "justify", "harden", or "complete"
    completed_modules: list[str] = Field(default_factory=list)
    
    def mark_stage_complete(self, stage: str) -> None:
        """Advance to next stage in BJH loop or next module."""
        if stage == "build":
            self.current_stage = "justify"
        elif stage == "justify":
            self.current_stage = "harden"
        elif stage == "harden":
            # Module complete, advance to next
            module_id = f"module_{self.current_module_index}"  # Will be replaced with actual ID
            if module_id not in self.completed_modules:
                self.completed_modules.append(module_id)
            self.current_module_index += 1
            self.current_stage = "build"


class FailureMode(BaseModel):
    """
    Schema for a single failure mode pattern in Justify stage evaluation.
    
    Attributes:
        category: Name of the failure pattern (e.g., "Hand-Waver", "Conceptual Mismatch")
        keywords: List of keywords that indicate this pattern
        feedback: Socratic hint to guide user toward correct understanding
    """
    category: str
    keywords: list[str]
    feedback: str


class JustifyQuestion(BaseModel):
    """
    Schema for a single Justify stage question in justify_questions.json.
    
    Attributes:
        id: Unique identifier for the question
        question: The actual question text presented to the user
        model_answer: Canonical correct answer for LLM comparison
        failure_modes: List of common failure patterns for keyword filtering
        required_concepts: List of key concepts that must be present in answer
    """
    id: str
    question: str
    model_answer: str
    failure_modes: list[FailureMode]
    required_concepts: list[str]


class LLMEvaluationResponse(BaseModel):
    """
    Schema for structured LLM response from Justify stage evaluation.
    
    This enforces the JSON format we require from the LLM API.
    
    Attributes:
        is_correct: Boolean indicating if the user's answer demonstrates understanding
        feedback: Either acceptance message or Socratic hint for next iteration
    """
    is_correct: bool
    feedback: str


class ValidationResult(BaseModel):
    """
    Schema for results from validator.sh execution.
    
    Attributes:
        exit_code: Shell exit code (0 = success)
        stdout: Captured standard output
        stderr: Captured standard error
        performance_seconds: Parsed performance metric if present
    """
    exit_code: int
    stdout: str
    stderr: str
    performance_seconds: Optional[float] = None


# Bug Definition Schemas for LLM Authoring Tool

from typing import List, Dict, Any, Union, Literal


class ContextReference(BaseModel):
    """Reference to a tracked context variable"""
    from_context: str
    
    class Config:
        extra = 'forbid'


class NameNode(BaseModel):
    """Name node in AST pattern"""
    node_type: Literal["Name"]
    id: Optional[Union[str, ContextReference]] = None
    
    class Config:
        extra = 'forbid'


class NestedPattern(BaseModel):
    """Nested pattern for value, left, right, func, etc."""
    node_type: str
    op: Optional[str] = None
    attr: Optional[str] = None
    
    class Config:
        extra = 'forbid'


class KeywordArg(BaseModel):
    """Keyword argument pattern"""
    arg: Optional[str] = None
    value: Optional[NestedPattern] = None
    
    class Config:
        extra = 'forbid'


class Pattern(BaseModel):
    """AST pattern for matching nodes"""
    node_type: str
    targets: Optional[List[NameNode]] = None
    value: Optional[NestedPattern] = None
    attr: Optional[str] = None
    op: Optional[str] = None
    func: Optional[NestedPattern] = None
    left: Optional[NestedPattern] = None
    right: Optional[NestedPattern] = None
    args: Optional[List[NestedPattern]] = None
    keywords: Optional[List[KeywordArg]] = None
    
    class Config:
        extra = 'forbid'


class Condition(BaseModel):
    """Condition for pattern matching"""
    check: str
    value: Optional[Union[int, str]] = None
    index: Optional[int] = None
    name: Optional[str] = None
    
    class Config:
        extra = 'forbid'


class Replacement(BaseModel):
    """Replacement specification"""
    type: str
    source: Optional[Union[str, ContextReference]] = None
    name: Optional[str] = None
    
    class Config:
        extra = 'forbid'


class PassDefinition(BaseModel):
    """Single pass in bug injection logic"""
    pass_: int = Field(..., alias="pass")
    type: str
    description: str
    pattern: Optional[Pattern] = None
    conditions: Optional[List[Condition]] = None
    track_as: Optional[Dict[str, str]] = None
    replacement: Optional[Replacement] = None
    
    class Config:
        populate_by_name = True
        extra = 'forbid'


class BugMetadata(BaseModel):
    """Bug metadata"""
    created: str
    version: str
    author: str
    tier: str
    
    class Config:
        extra = 'forbid'


class BugDefinition(BaseModel):
    """Complete bug definition matching v2.1 schema"""
    id: str
    description: str
    injection_type: str
    engine_version: str
    target_function: str
    logic: List[PassDefinition]
    metadata: BugMetadata
    
    class Config:
        extra = 'forbid'
