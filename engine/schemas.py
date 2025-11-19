"""
Pydantic data schemas for the Mastery Engine.

This module defines all data contracts between the engine core and curriculum content.
Using Pydantic provides:
- Runtime validation of curriculum and state files
- Type safety and IDE autocompletion
- Clear API contracts for content authors
- Automatic JSON serialization/deserialization
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class CurriculumType(str, Enum):
    """Type of curriculum structure."""
    LINEAR = "linear"    # Sequential modules (e.g., cs336_a1)
    LIBRARY = "library"  # Hierarchical patterns+problems (e.g., cp_accelerator)


# --- Legacy / Linear Components ---

class ModuleMetadata(BaseModel):
    """
    Metadata for a single curriculum module (Linear mode).
    
    Used by LINEAR curricula like cs336_a1 where modules must be completed sequentially.
    
    Attributes:
        id: Unique identifier for the module (e.g., "rmsnorm", "multihead_attention")
        name: Human-readable display name (e.g., "RMS Normalization")
        path: Relative path to module directory within curriculum pack
        baseline_perf_seconds: Optional performance baseline from Phase 0 CI for novelty detection
        dependencies: Optional list of module IDs that must be completed first
        module_type: Type classification (default: "standard")
        metadata: Optional dict for curriculum-specific extensions (e.g., rating_bracket, priority)
    """
    id: str
    name: str
    path: str
    baseline_perf_seconds: Optional[float] = None
    dependencies: list[str] = Field(default_factory=list)
    module_type: str = "standard"
    metadata: dict = Field(default_factory=dict)


# --- New / Hierarchical Components ---

class ProblemMetadata(BaseModel):
    """
    Metadata for a specific practice problem (Library mode).
    
    Used by LIBRARY curricula like cp_accelerator where users can select problems freely.
    
    Attributes:
        id: Unique identifier for the problem (e.g., "lc_912")
        title: Human-readable problem name (e.g., "Sort an Array")
        path: Relative path to problem directory (e.g., "patterns/sorting/problems/lc_912")
        difficulty: Problem difficulty level (e.g., "Medium")
        baseline_perf_seconds: Optional performance baseline
        metadata: Optional dict for problem-specific data (e.g., leetcode_id, tags)
    """
    id: str
    title: str
    path: str
    difficulty: str
    baseline_perf_seconds: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class PatternMetadata(BaseModel):
    """
    Metadata for a pattern container (Library mode).
    
    A pattern groups related problems and shared theory (e.g., "Sorting Algorithms").
    
    Attributes:
        id: Unique identifier for the pattern (e.g., "sorting")
        title: Human-readable pattern name (e.g., "Sorting Algorithms")
        theory_path: Path to shared theory resources (e.g., "patterns/sorting/theory")
        problems: List of problems in this pattern
        metadata: Optional dict for pattern-specific data (e.g., estimated_hours, priority)
    """
    id: str
    title: str
    theory_path: str
    problems: List[ProblemMetadata]
    metadata: dict = Field(default_factory=dict)


# --- Root Manifest ---

class CurriculumManifest(BaseModel):
    """
    Root schema for a curriculum pack's manifest.json file.
    
    Supports both LINEAR (sequential modules) and LIBRARY (hierarchical patterns) modes.
    
    Attributes:
        curriculum_name: Unique identifier for the curriculum pack (e.g., "cs336_a1")
        author: Creator/maintainer of the curriculum
        version: Semantic version string (e.g., "1.0.0")
        type: Curriculum structure type (LINEAR or LIBRARY, defaults to LINEAR for backward compat)
        modules: Ordered list of modules (LINEAR mode only)
        patterns: Hierarchical patterns with problems (LIBRARY mode only)
    """
    curriculum_name: str
    author: str
    version: str
    type: CurriculumType = CurriculumType.LINEAR  # Default maintains backward compatibility
    
    # Mutually exclusive based on type
    modules: Optional[list[ModuleMetadata]] = None      # For LINEAR curricula
    patterns: Optional[List[PatternMetadata]] = None    # For LIBRARY curricula


class UserProgress(BaseModel):
    """
    Schema for .mastery_progress.json tracking user's learning state.
    
    Supports both LINEAR and LIBRARY curriculum modes.
    
    Attributes:
        curriculum_id: ID of the active curriculum pack
        
        Linear Mode State:
        current_module_index: Index into curriculum's modules list (0-based)
        
        Library Mode State:
        active_pattern_id: Currently selected pattern ID (e.g., "sorting")
        active_problem_id: Currently selected problem ID (e.g., "lc_912")
        
        Shared State:
        current_stage: Current stage within the BJH loop
        completed_modules: List of module IDs fully completed (LINEAR mode)
        completed_patterns: List of pattern IDs with theory completed (LIBRARY mode)
        completed_problems: List of problem IDs fully completed (LIBRARY mode, format: "pattern_id/problem_id")
    """
    curriculum_id: str
    
    # Linear Mode State
    current_module_index: int = 0
    
    # Library Mode State (New in v2.0)
    active_pattern_id: Optional[str] = None
    active_problem_id: Optional[str] = None
    
    # Shared State
    current_stage: str = "build"  # "build", "justify", "harden", or "complete"
    completed_modules: list[str] = Field(default_factory=list)     # Linear mode tracking
    completed_patterns: list[str] = Field(default_factory=list)    # Library mode pattern theory
    completed_problems: list[str] = Field(default_factory=list)    # Library mode problems
    
    def mark_stage_complete(self, stage: str) -> None:
        """Advance to next stage in BJH loop or next module.
        
        NOTE: This method maintains legacy LINEAR behavior for backward compatibility.
        LIBRARY mode logic will be implemented in Phase 2 (CLI refactor).
        """
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
