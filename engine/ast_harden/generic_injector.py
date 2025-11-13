"""
Generic bug injector that interprets declarative JSON bug definitions.

This module provides the main GenericBugInjector class that orchestrates
the multi-pass bug injection process based on JSON logic definitions.
"""

from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
from typing import Optional

from engine.services.ast_service import Canonicalizer
from engine.ast_harden.pattern_matcher import FindAndTrackVisitor, FindAndReplaceTransformer


class GenericBugInjector:
    """
    Generic bug injector that works from JSON definitions.
    
    This replaces hardcoded bug injectors with a data-driven approach
    that interprets multi-pass logic from JSON files.
    """
    
    def __init__(self, bug_definition: dict | Path):
        """
        Initialize generic bug injector.
        
        Args:
            bug_definition: Either a dict with bug definition or path to JSON file
        """
        if isinstance(bug_definition, Path):
            with open(bug_definition, 'r') as f:
                self.bug_def = json.load(f)
        else:
            self.bug_def = bug_definition
        
        self.validate_definition()
    
    def validate_definition(self):
        """Validate that bug definition has required fields."""
        required = ['id', 'injection_type', 'logic']
        for field in required:
            if field not in self.bug_def:
                raise ValueError(f"Bug definition missing required field: {field}")
        
        if self.bug_def['injection_type'] != 'ast':
            raise ValueError(f"Unsupported injection type: {self.bug_def['injection_type']}")
    
    def inject(self, source_code: str) -> tuple[str, bool]:
        """
        Inject bug into source code based on JSON definition.
        
        Args:
            source_code: Student's correct implementation
            
        Returns:
            (buggy_code, success) where success indicates if bug was injected
        """
        # Step 1: Parse student's code to ORIGINAL AST (preserve this!)
        try:
            original_ast = ast.parse(source_code)
        except SyntaxError:
            return source_code, False
        
        # Step 2: Check if target function exists
        target_function = self.bug_def.get('target_function')
        if target_function and not self._has_function(original_ast, target_function):
            return source_code, False
        
        # Step 3: Create canonical AST for pattern matching
        temp_ast = copy.deepcopy(original_ast)
        canonicalizer = Canonicalizer(target_function=target_function if target_function else 'softmax')
        canonical_ast = canonicalizer.visit(temp_ast)
        ast.fix_missing_locations(canonical_ast)
        
        # Step 4: Execute multi-pass logic
        context = {}  # Shared context across passes
        
        for pass_def in self.bug_def['logic']:
            pass_type = pass_def['type']
            
            if pass_type == 'find_and_track':
                # Pass 1: Find patterns and extract context
                visitor = FindAndTrackVisitor(pass_def, context)
                visitor.visit(canonical_ast)
            
            elif pass_type == 'find_and_replace':
                # Pass 2: Transform original AST
                transformer = FindAndReplaceTransformer(pass_def, context, original_ast)
                original_ast = transformer.visit(original_ast)
                ast.fix_missing_locations(original_ast)
                
                if transformer.replacements_made == 0:
                    # Pattern not found
                    return source_code, False
        
        # Step 5: Unparse modified AST
        try:
            buggy_code = ast.unparse(original_ast)
            return buggy_code, True
        except Exception:
            return source_code, False
    
    def _has_function(self, tree: ast.AST, function_name: str) -> bool:
        """Check if AST contains a function with given name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True
        return False
