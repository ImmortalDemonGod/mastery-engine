"""
Phase 1: Proof of Concept v2.1 - Mapping-Based AST Transformation

This implements the hybrid architecture that:
1. Uses canonical AST for robust pattern matching (find the location)
2. Transforms the ORIGINAL AST (preserves student's variable names)

Key Innovation: The canonical AST is a disposable "map" that guides
a surgical transformation on the student's original code.

Result: Bug is injected reliably, but student sees their own variable names.
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
    A plan for transforming the original AST, created by analyzing the canonical AST.
    
    Contains:
    - target_location: Where to make the change (lineno, col_offset)
    - original_var_name: The student's original variable name to use in replacement
    """
    target_location: NodeLocation
    original_var_name: str
    

class SoftmaxCanonicalizer(ast.NodeTransformer):
    """
    Hardcoded canonicalizer for softmax function only.
    
    Renames variables to _arg0, _arg1, _var0, _var1, etc. to create
    a standardized AST that is immune to student naming choices.
    """
    
    def __init__(self):
        self.var_counter = 0
        self.var_mapping = {}  # Maps original name -> canonical name
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Only process if this is the softmax function."""
        if node.name != 'softmax':
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


class CanonicalPatternMatcher(ast.NodeVisitor):
    """
    Analyzes the canonical AST to find the subtract-max pattern.
    
    Returns a TransformationPlan that describes WHERE and HOW to transform
    the ORIGINAL AST.
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
            print(f"    [Canonical] Tracked: {max_var_name} = max of {source_var}")
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
            print(f"    [Canonical] Found pattern: {left_var} - {right_var}")
            print(f"      Location: line {node.lineno}, col {node.col_offset}")
            
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
                        print(f"      [Original] Student's variable: '{original_var_name}'")
                        
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
            # This is the node we need to transform!
            print(f"  [Transform] Applying transformation at line {node.lineno}")
            print(f"    Original: {ast.unparse(node)}")
            
            # Replace the BinOp with just the Name node using the ORIGINAL variable name
            node.value = ast.Name(id=self.plan.original_var_name, ctx=ast.Load())
            self.bug_injected = True
            
            print(f"    Buggy:    {ast.unparse(node)}")
        
        return node


def inject_softmax_bug_v2_1(source_code: str) -> tuple[str, bool]:
    """
    Complete v2.1 pipeline: 
    Parse → Canonicalize (for matching) → Create Plan → Transform Original → Unparse
    
    Args:
        source_code: Student's correct softmax implementation
        
    Returns:
        Tuple of (buggy_code, success)
        - buggy_code: Modified code with bug injected AND original variable names preserved
        - success: True if bug was successfully injected
    """
    # Step 1: Parse student's code to ORIGINAL AST (preserve this!)
    try:
        original_ast = ast.parse(source_code)
    except SyntaxError as e:
        raise SyntaxError(f"Student code is not valid Python: {e}")
    
    # Verify softmax function exists
    softmax_found = False
    for node in ast.walk(original_ast):
        if isinstance(node, ast.FunctionDef) and node.name == 'softmax':
            softmax_found = True
            break
    
    if not softmax_found:
        raise ValueError("No 'softmax' function found in source code")
    
    print("=" * 70)
    print("PHASE 1: ANALYSIS (Using Canonical AST)")
    print("=" * 70)
    
    # Step 2: Create a COPY for canonicalization (disposable)
    temp_ast = copy.deepcopy(original_ast)
    
    # Step 3: Canonicalize the COPY
    print("\n  [Canonical] Creating canonical AST...")
    canonicalizer = SoftmaxCanonicalizer()
    canonical_ast = canonicalizer.visit(temp_ast)
    ast.fix_missing_locations(canonical_ast)
    print(f"  [Canonical] Renamed {len(canonicalizer.var_mapping)} variables")
    
    # Step 4: Analyze canonical AST to create transformation plan
    print("\n  [Canonical] Searching for subtract-max pattern...")
    matcher = CanonicalPatternMatcher(original_ast)
    matcher.visit(canonical_ast)
    
    if not matcher.transformation_plan:
        print("\n  ⚠️  Pattern not found in canonical AST")
        return source_code, False
    
    print(f"\n  ✅ Analysis complete! Transformation plan created.")
    
    # IMPORTANT: Dispose of the canonical AST (it was just a map)
    del canonical_ast
    del temp_ast
    
    print("\n" + "=" * 70)
    print("PHASE 2: TRANSFORMATION (Using Original AST)")
    print("=" * 70)
    
    # Step 5: Transform the ORIGINAL AST using the plan
    print(f"\n  [Transform] Applying plan to original AST...")
    transformer = OriginalASTTransformer(matcher.transformation_plan)
    modified_original_ast = transformer.visit(original_ast)
    ast.fix_missing_locations(modified_original_ast)
    
    if not transformer.bug_injected:
        print("\n  ⚠️  Transformation failed")
        return source_code, False
    
    # Step 6: Unparse the modified ORIGINAL AST
    buggy_code = ast.unparse(modified_original_ast)
    
    print(f"\n  ✅ Bug successfully injected with ORIGINAL variable names preserved!")
    
    return buggy_code, True


# Test cases
if __name__ == '__main__':
    # Test case 1: Standard implementation
    test_code_1 = """
import torch

def softmax(in_features, dim):
    x = in_features
    orig_dtype = x.dtype
    x32 = x.float()
    max_vals = x32.max(dim=dim, keepdim=True).values
    shifted = x32 - max_vals
    exps = torch.exp(shifted)
    sums = exps.sum(dim=dim, keepdim=True)
    out = exps / sums
    return out.to(orig_dtype)
"""
    
    print("=" * 70)
    print("TEST 1: Standard Implementation")
    print("=" * 70)
    buggy, success = inject_softmax_bug_v2_1(test_code_1)
    print("\n" + "=" * 70)
    print("FINAL BUGGY CODE (with original variable names):")
    print("=" * 70)
    print(buggy)
    print(f"\nSuccess: {success}")
    
    # Test case 2: Different variable names
    test_code_2 = """
import torch

def softmax(input_tensor, dimension):
    tensor_float = input_tensor.float()
    maximum_value = tensor_float.max(dim=dimension, keepdim=True).values
    normalized = tensor_float - maximum_value
    exponentials = torch.exp(normalized)
    return exponentials / exponentials.sum(dim=dimension, keepdim=True)
"""
    
    print("\n\n" + "=" * 70)
    print("TEST 2: Different Variable Names")
    print("=" * 70)
    buggy, success = inject_softmax_bug_v2_1(test_code_2)
    print("\n" + "=" * 70)
    print("FINAL BUGGY CODE (with original variable names):")
    print("=" * 70)
    print(buggy)
    print(f"\nSuccess: {success}")
