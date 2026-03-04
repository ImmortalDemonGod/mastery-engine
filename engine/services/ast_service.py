"""
AST-based bug injection service for HARDEN stage.

This service implements semantic bug injection using a mapping-based approach:
1. Canonicalize student code for robust pattern matching
2. Identify semantic patterns in canonical AST
3. Create transformation plan with original variable names
4. Apply transformation to original AST (preserves student's names)

This ensures both robustness and pedagogical fidelity.
"""

from __future__ import annotations
import ast
import copy
from typing import Optional
from dataclasses import dataclass


@dataclass
class NodeLocation:
    """Identifies a specific node in an AST by its source location."""
    lineno: int
    col_offset: int
    
    def matches(self, node: ast.AST) -> bool:
        """Check if a node is at this location."""
        return (hasattr(node, 'lineno') and 
                hasattr(node, 'col_offset') and
                node.lineno == self.lineno and 
                node.col_offset == self.col_offset)


@dataclass
class TransformationPlan:
    """
    A plan for transforming the original AST.
    
    Created by analyzing the canonical AST, contains the location
    and the student's original variable name to use in replacement.
    """
    target_location: NodeLocation
    original_var_name: str


class Canonicalizer(ast.NodeTransformer):
    """
    Transforms student code into a canonical form with standardized variable names.
    
    This creates a predictable AST structure that is immune to student naming
    choices, enabling robust semantic pattern matching.
    
    Currently hardcoded for softmax function, but designed to be generalizable.
    """
    
    def __init__(self, target_function: str = 'softmax'):
        self.target_function = target_function
        self.var_counter = 0
        self.var_mapping = {}  # Maps original name -> canonical name
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Only process if this is the target function."""
        if node.name != self.target_function:
            return node
            
        # Reset state for this function
        self.var_counter = 0
        self.var_mapping = {}
        
        # Rename function arguments
        for i, arg in enumerate(node.args.args):
            canonical_name = f'_arg{i}'
            self.var_mapping[arg.arg] = canonical_name
            arg.arg = canonical_name
        
        # Process function body
        node.body = [self.visit(stmt) for stmt in node.body]
        
        return node
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Rename variables at assignment and visit the value."""
        # Visit the right-hand side first (it uses old names)
        node.value = self.visit(node.value)
        
        # Then rename the target(s)
        for target in node.targets:
            if isinstance(target, ast.Name):
                canonical_name = f'_var{self.var_counter}'
                self.var_mapping[target.id] = canonical_name
                target.id = canonical_name
                self.var_counter += 1
        
        return node
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Replace variable references with canonical names."""
        if node.id in self.var_mapping:
            node.id = self.var_mapping[node.id]
        return node


class SoftmaxBugInjector:
    """
    Hardcoded bug injector for softmax no-subtract-max bug.
    
    Uses a two-pass reconnaissance & strike strategy:
    1. Analyze canonical AST to find semantic pattern
    2. Transform original AST to preserve student variable names
    """
    
    def __init__(self):
        self.logger_prefix = "[SoftmaxBugInjector]"
    
    def inject(self, source_code: str) -> tuple[str, bool]:
        """
        Inject the no-subtract-max bug into student's softmax implementation.
        
        Args:
            source_code: Student's correct softmax implementation
            
        Returns:
            Tuple of (buggy_code, success)
            - buggy_code: Modified code with bug injected and original names preserved
            - success: True if bug was successfully injected
        """
        # Step 1: Parse student's code to ORIGINAL AST (preserve this!)
        try:
            original_ast = ast.parse(source_code)
        except SyntaxError as e:
            return source_code, False
        
        # Verify softmax function exists
        if not self._has_softmax_function(original_ast):
            return source_code, False
        
        # Step 2: Create a COPY for canonicalization (disposable)
        temp_ast = copy.deepcopy(original_ast)
        
        # Step 3: Canonicalize the COPY
        canonicalizer = Canonicalizer(target_function='softmax')
        canonical_ast = canonicalizer.visit(temp_ast)
        ast.fix_missing_locations(canonical_ast)
        
        # Step 4: Analyze canonical AST to create transformation plan
        matcher = CanonicalPatternMatcher(original_ast)
        matcher.visit(canonical_ast)
        
        if not matcher.transformation_plan:
            return source_code, False
        
        # Dispose of canonical AST (it was just a map)
        del canonical_ast
        del temp_ast
        
        # Step 5: Transform the ORIGINAL AST using the plan
        transformer = OriginalASTTransformer(matcher.transformation_plan)
        modified_original_ast = transformer.visit(original_ast)
        ast.fix_missing_locations(modified_original_ast)
        
        if not transformer.bug_injected:
            return source_code, False
        
        # Step 6: Unparse the modified ORIGINAL AST
        buggy_code = ast.unparse(modified_original_ast)
        
        return buggy_code, True
    
    def _has_softmax_function(self, tree: ast.AST) -> bool:
        """Check if the AST contains a softmax function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'softmax':
                return True
        return False


class CanonicalPatternMatcher(ast.NodeVisitor):
    """
    Analyzes the canonical AST to find the subtract-max pattern.
    
    Returns a TransformationPlan that describes WHERE and HOW to transform
    the ORIGINAL AST while preserving student variable names.
    """
    
    def __init__(self, original_ast: ast.AST):
        self.original_ast = original_ast
        self.max_vars = {}  # Maps var name -> the variable it's the max of
        self.transformation_plan: Optional[TransformationPlan] = None
        
    def _is_max_call(self, node: ast.AST) -> tuple[bool, Optional[str]]:
        """Check if node is a .max() call, return (is_max, variable_name)."""
        # Handle .values attribute: tensor.max(...).values
        if isinstance(node, ast.Attribute):
            node = node.value
        
        if not isinstance(node, ast.Call):
            return False, None
        
        if not isinstance(node.func, ast.Attribute):
            return False, None
        
        if node.func.attr != 'max':
            return False, None
        
        if not isinstance(node.func.value, ast.Name):
            return False, None
        
        return True, node.func.value.id
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Track max assignments and find subtract-max pattern."""
        # First, check if this assignment computes a max
        is_max, source_var = self._is_max_call(node.value)
        if is_max and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            max_var_name = node.targets[0].id
            self.max_vars[max_var_name] = source_var
            self.generic_visit(node)
            return
        
        # Second, check if this is a subtract-max pattern
        if not isinstance(node.value, ast.BinOp):
            self.generic_visit(node)
            return
        
        if not isinstance(node.value.op, ast.Sub):
            self.generic_visit(node)
            return
        
        binop = node.value
        
        # Both sides should be Name nodes
        if not isinstance(binop.left, ast.Name) or not isinstance(binop.right, ast.Name):
            self.generic_visit(node)
            return
        
        left_var = binop.left.id
        right_var = binop.right.id
        
        # Check if right_var is the max of left_var
        if right_var in self.max_vars and self.max_vars[right_var] == left_var:
            # FOUND IT in the canonical AST!
            # Now find the corresponding node in the ORIGINAL AST
            original_node = self._find_node_at_location(
                self.original_ast, 
                node.lineno, 
                node.col_offset
            )
            
            if original_node and isinstance(original_node, ast.Assign):
                # Extract the student's ORIGINAL variable name from the left side of BinOp
                if isinstance(original_node.value, ast.BinOp):
                    if isinstance(original_node.value.left, ast.Name):
                        original_var_name = original_node.value.left.id
                        
                        # Create the transformation plan
                        self.transformation_plan = TransformationPlan(
                            target_location=NodeLocation(node.lineno, node.col_offset),
                            original_var_name=original_var_name
                        )
        
        self.generic_visit(node)
    
    def _find_node_at_location(self, tree: ast.AST, lineno: int, col_offset: int) -> Optional[ast.AST]:
        """Find a node in the tree at the given source location."""
        for node in ast.walk(tree):
            if (hasattr(node, 'lineno') and hasattr(node, 'col_offset') and
                node.lineno == lineno and node.col_offset == col_offset):
                return node
        return None


class OriginalASTTransformer(ast.NodeTransformer):
    """
    Transforms the ORIGINAL AST based on the TransformationPlan.
    
    This preserves the student's variable names while making the surgical change.
    """
    
    def __init__(self, plan: TransformationPlan):
        self.plan = plan
        self.bug_injected = False
        
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Check if this is the target node, and if so, transform it."""
        if self.plan.target_location.matches(node):
            # Replace the BinOp with just the Name node using the ORIGINAL variable name
            node.value = ast.Name(id=self.plan.original_var_name, ctx=ast.Load())
            self.bug_injected = True
        
        return node
