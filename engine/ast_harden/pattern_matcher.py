"""
Generic AST pattern matching system for declarative bug definitions.

This module provides a flexible pattern matching system that can match
AST nodes against JSON-defined patterns, with support for context variables
and cross-pass references.
"""

from __future__ import annotations
import ast
from typing import Any, Optional


class PatternMatcher:
    """
    Matches AST nodes against JSON-defined patterns.
    
    Supports:
    - Node type matching
    - Attribute matching
    - Nested pattern matching
    - Context variable references (from_context)
    - Conditional checks
    """
    
    def __init__(self, pattern: dict, context: dict[str, Any] | None = None):
        """
        Initialize pattern matcher.
        
        Args:
            pattern: JSON pattern definition
            context: Shared context dictionary for cross-pass references
        """
        self.pattern = pattern
        self.context = context or {}
    
    def matches(self, node: ast.AST) -> bool:
        """Check if a node matches the pattern."""
        return self._match_node(node, self.pattern)
    
    def _match_node(self, node: ast.AST, pattern: dict) -> bool:
        """Recursively match a node against a pattern."""
        # Check node type
        if 'node_type' in pattern:
            expected_type = pattern['node_type']
            if node.__class__.__name__ != expected_type:
                return False
        
        # Match specific attributes
        for key, value in pattern.items():
            if key in ['node_type', 'from_context']:
                continue  # Already handled or special
            
            if not hasattr(node, key):
                return False
            
            node_value = getattr(node, key)
            
            # Special handling for 'op' attribute (ast operator nodes)
            if key == 'op' and isinstance(value, str):
                # Convert AST op node to string for comparison
                op_name = node_value.__class__.__name__
                if op_name != value:
                    return False
                continue
            
            if isinstance(value, dict):
                # Check for context reference
                if 'from_context' in value:
                    context_key = value['from_context']
                    if context_key not in self.context:
                        return False
                    expected = self.context[context_key]
                    if node_value != expected:
                        return False
                else:
                    # Nested pattern match
                    if not self._match_node(node_value, value):
                        return False
            elif isinstance(value, str):
                # Direct string comparison
                if node_value != value:
                    return False
            elif isinstance(value, list):
                # Match list of patterns
                if not isinstance(node_value, list):
                    return False
                if len(node_value) != len(value):
                    return False
                for node_item, pattern_item in zip(node_value, value):
                    if isinstance(pattern_item, dict):
                        if not self._match_node(node_item, pattern_item):
                            return False
            else:
                # Direct value comparison
                if node_value != value:
                    return False
        
        return True
    
    def check_conditions(self, node: ast.AST, conditions: list[dict]) -> bool:
        """Check additional conditions on a matched node."""
        for condition in conditions:
            check_type = condition['check']
            
            if check_type == 'targets_length_equals':
                if not hasattr(node, 'targets'):
                    return False
                if len(node.targets) != condition['value']:
                    return False
            
            elif check_type == 'target_is_name':
                if not hasattr(node, 'targets'):
                    return False
                index = condition.get('index', 0)
                if index >= len(node.targets):
                    return False
                if not isinstance(node.targets[index], ast.Name):
                    return False
            
            elif check_type == 'has_keyword_arg':
                if not isinstance(node, ast.Call):
                    return False
                arg_name = condition['name']
                if not any(kw.arg == arg_name for kw in node.keywords):
                    return False
            
            # Add more condition types as needed
        
        return True
    
    def extract_context(self, node: ast.AST, track_as: dict[str, str]) -> dict[str, Any]:
        """Extract context variables from a matched node."""
        result = {}
        
        for context_key, path in track_as.items():
            value = self._evaluate_path(node, path)
            if value is not None:
                result[context_key] = value
        
        return result
    
    def _evaluate_path(self, node: ast.AST, path: str) -> Any:
        """
        Evaluate a path like 'node.targets[0].id' on an AST node.
        
        Args:
            node: AST node to evaluate on
            path: Dot-notation path string
            
        Returns:
            Value at the path, or None if path invalid
        """
        parts = path.split('.')
        current = node
        
        for part in parts[1:]:  # Skip 'node' prefix
            # Handle array indexing
            if '[' in part:
                attr_name = part[:part.index('[')]
                index_str = part[part.index('[')+1:part.index(']')]
                index = int(index_str)
                
                if not hasattr(current, attr_name):
                    return None
                current = getattr(current, attr_name)
                if not isinstance(current, (list, tuple)):
                    return None
                if index >= len(current):
                    return None
                current = current[index]
            else:
                # Simple attribute access
                if not hasattr(current, part):
                    return None
                current = getattr(current, part)
        
        return current


class FindAndTrackVisitor(ast.NodeVisitor):
    """
    Visitor for 'find_and_track' pass.
    
    Traverses the canonical AST to find patterns and extract context variables
    from the ORIGINAL AST (to preserve student variable names).
    """
    
    def __init__(self, pass_def: dict, context: dict[str, Any], original_ast: ast.AST):
        self.pass_def = pass_def
        self.context = context
        self.original_ast = original_ast
        self.pattern_matcher = PatternMatcher(pass_def['pattern'], context)
    
    def visit(self, node: ast.AST):
        """Visit a node and check for pattern match."""
        if self.pattern_matcher.matches(node):
            # Check additional conditions if present
            if 'conditions' in self.pass_def:
                if not self.pattern_matcher.check_conditions(node, self.pass_def['conditions']):
                    self.generic_visit(node)
                    return
            
            # Find corresponding node in original AST
            original_node = self._find_node_at_location(
                self.original_ast,
                node.lineno,
                node.col_offset
            )
            
            # Extract context variables from ORIGINAL AST
            if 'track_as' in self.pass_def and original_node:
                extracted = self.pattern_matcher.extract_context(original_node, self.pass_def['track_as'])
                self.context.update(extracted)
        
        self.generic_visit(node)
    
    def _find_node_at_location(self, tree: ast.AST, lineno: int, col_offset: int) -> Optional[ast.AST]:
        """Find a node in the tree at the given source location."""
        for node in ast.walk(tree):
            if (hasattr(node, 'lineno') and hasattr(node, 'col_offset') and
                node.lineno == lineno and node.col_offset == col_offset):
                return node
        return None


class FindAndReplaceTransformer(ast.NodeTransformer):
    """
    Transformer for 'find_and_replace' pass.
    
    Finds patterns in the original AST and replaces them according to
    replacement rules.
    """
    
    def __init__(self, pass_def: dict, context: dict[str, Any], original_ast: ast.AST):
        self.pass_def = pass_def
        self.context = context
        self.original_ast = original_ast
        self.pattern_matcher = PatternMatcher(pass_def['pattern'], context)
        self.replacements_made = 0
    
    def visit(self, node: ast.AST):
        """Visit a node and check for pattern match."""
        if self.pattern_matcher.matches(node):
            # Found a match! Apply replacement
            replacement = self._create_replacement(node)
            if replacement is not None:
                self.replacements_made += 1
                return replacement
        
        return self.generic_visit(node)
    
    def _create_replacement(self, node: ast.AST) -> Optional[ast.AST]:
        """Create replacement node according to replacement rules."""
        replacement_def = self.pass_def.get('replacement', {})
        replacement_type = replacement_def.get('type')
        
        if replacement_type == 'replace_value_with':
            # Replace node's value with value from another part of the node
            source_path = replacement_def['source']
            matcher = PatternMatcher({}, self.context)
            new_value = matcher._evaluate_path(node, source_path)
            
            if new_value is not None and hasattr(node, 'value'):
                # Create a copy of the node with new value
                new_node = ast.copy_location(
                    ast.Assign(targets=node.targets, value=new_value),
                    node
                )
                return new_node
        
        elif replacement_type == 'replace_with':
            # Replace entire node with another part of itself
            source_path = replacement_def['source']
            matcher = PatternMatcher({}, self.context)
            replacement_node = matcher._evaluate_path(node, source_path)
            
            if replacement_node is not None:
                return replacement_node
        
        elif replacement_type == 'remove_keyword_arg':
            # Remove a keyword argument from a Call node
            arg_name = replacement_def['name']
            if isinstance(node, ast.Call):
                new_node = ast.copy_location(
                    ast.Call(
                        func=node.func,
                        args=node.args,
                        keywords=[kw for kw in node.keywords if kw.arg != arg_name]
                    ),
                    node
                )
                return new_node
        
        return None
