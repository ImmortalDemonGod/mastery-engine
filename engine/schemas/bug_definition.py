"""
Pydantic schemas for bug definitions - used with OpenAI Structured Outputs
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class Pattern(BaseModel):
    """AST pattern for matching nodes"""
    node_type: str
    targets: Optional[List[Dict[str, Any]]] = None
    value: Optional[Dict[str, Any]] = None
    attr: Optional[str] = None
    op: Optional[str] = None
    func: Optional[Dict[str, Any]] = None
    left: Optional[Dict[str, Any]] = None
    right: Optional[Dict[str, Any]] = None
    args: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[Dict[str, Any]]] = None


class Condition(BaseModel):
    """Condition for pattern matching"""
    check: str
    value: Optional[Union[int, str]] = None
    index: Optional[int] = None
    name: Optional[str] = None


class Replacement(BaseModel):
    """Replacement specification"""
    type: str
    source: Optional[Union[str, Dict[str, Any]]] = None
    name: Optional[str] = None


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


class Metadata(BaseModel):
    """Bug metadata"""
    created: str
    version: str
    author: str
    tier: str


class BugDefinition(BaseModel):
    """Complete bug definition matching v2.1 schema"""
    id: str
    description: str
    injection_type: str
    engine_version: str
    target_function: str
    logic: List[PassDefinition]
    metadata: Metadata
