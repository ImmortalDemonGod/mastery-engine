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
    
    def inject(self, source_code: str, debug: bool = False) -> tuple[str, bool]:
        """
        Inject bug into source code based on JSON definition.
        
        Args:
            source_code: Student's correct implementation
            debug: Enable debug logging
            
        Returns:
            (buggy_code, success) where success indicates if bug was injected
        """
        if debug:
            print(f"\n{'='*60}")
            print("INJECTION DEBUG TRACE")
            print(f"{'='*60}")
        
        # Step 1: Parse student's code to ORIGINAL AST (preserve this!)
        # Note: Dedent code first if it has leading whitespace
        import textwrap
        dedented_code = textwrap.dedent(source_code)
        
        if debug:
            print(f"\n[Step 1] Dedent & Parse")
            print(f"  Original length: {len(source_code)} chars")
            print(f"  Dedented length: {len(dedented_code)} chars")
            print(f"  First 50 chars: {repr(dedented_code[:50])}")
        
        try:
            original_ast = ast.parse(dedented_code)
            if debug:
                print(f"  ✅ Parse successful")
        except SyntaxError as e:
            if debug:
                print(f"  ❌ Parse failed: {e}")
            return dedented_code, False
        
        # Step 2: Check if target function exists (skip for code snippets)
        target_function = self.bug_def.get('target_function')
        # Note: For code snippets from patches, the function won't exist in AST
        # Only check if we have actual function definitions in the code
        has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(original_ast))
        
        if debug:
            print(f"\n[Step 2] Target Function Check")
            print(f"  Target function: {target_function}")
            print(f"  Has functions: {has_functions}")
        
        if target_function and has_functions and not self._has_function(original_ast, target_function):
            if debug:
                print(f"  ❌ Target function not found - aborting")
            return source_code, False
        
        if debug:
            print(f"  ✅ Proceeding (snippet or function found)")
        
        # Step 3: Create canonical AST for pattern matching
        temp_ast = copy.deepcopy(original_ast)
        canonicalizer = Canonicalizer(target_function=target_function if target_function else 'softmax')
        canonical_ast = canonicalizer.visit(temp_ast)
        ast.fix_missing_locations(canonical_ast)
        
        if debug:
            print(f"\n[Step 3] Canonicalization")
            print(f"  Canonical AST created")
        
        # Step 4: Execute multi-pass logic
        context = {}  # Shared context across passes
        
        if debug:
            print(f"\n[Step 4] Multi-Pass Execution")
            print(f"  Total passes: {len(self.bug_def['logic'])}")
        
        for i, pass_def in enumerate(self.bug_def['logic'], 1):
            pass_type = pass_def['type']
            
            if debug:
                print(f"\n  Pass {i}/{len(self.bug_def['logic'])}: {pass_type}")
                if 'pattern' in pass_def:
                    pattern = pass_def['pattern']
                    if 'targets' in pattern and pattern['targets']:
                        target_id = pattern['targets'][0].get('id', 'N/A')
                        print(f"    Target: {target_id}")
                    if 'value' in pattern:
                        print(f"    Value type: {pattern['value'].get('node_type', 'N/A')}")
            
            if pass_type == 'find_and_track':
                # Pass 1: Find patterns and extract context from original AST
                visitor = FindAndTrackVisitor(pass_def, context, original_ast)
                visitor.visit(canonical_ast)
                
                if debug:
                    print(f"    Context extracted: {list(context.keys())}")
            
            elif pass_type == 'find_and_replace':
                # Pass 2: Transform original AST
                # NOTE: Patterns with specific variable names (id fields) must match
                # against original AST, not canonical (which renames variables to _var0, etc.)
                transformer = FindAndReplaceTransformer(pass_def, context, original_ast, debug=debug)
                original_ast = transformer.visit(original_ast)
                ast.fix_missing_locations(original_ast)
                
                if debug:
                    print(f"    Replacements made: {transformer.replacements_made}")
                
                # Update canonical AST to match (for multi-pass scenarios)
                temp_ast = copy.deepcopy(original_ast)
                canonical_ast = canonicalizer.visit(temp_ast)
                ast.fix_missing_locations(canonical_ast)
                
                if transformer.replacements_made == 0:
                    # Pattern not found
                    if debug:
                        print(f"    ❌ No matches found - aborting")
                    return source_code, False
        
        # Step 5: Unparse modified AST
        if debug:
            print(f"\n[Step 5] Unparse & Return")
        
        try:
            buggy_code = ast.unparse(original_ast)
            if debug:
                print(f"  ✅ Unparse successful")
                print(f"  Original length: {len(dedented_code)}")
                print(f"  Buggy length: {len(buggy_code)}")
                print(f"  Changed: {buggy_code != dedented_code}")
                print(f"\n{'='*60}")
                print("INJECTION SUCCESSFUL")
                print(f"{'='*60}\n")
            return buggy_code, True
        except Exception as e:
            if debug:
                print(f"  ❌ Unparse failed: {e}")
            return source_code, False
    
    def _has_function(self, tree: ast.AST, function_name: str) -> bool:
        """Check if AST contains a function with given name."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True
        return False
