"""
Phase 1: Proof of Concept - Hardcoded Softmax Bug Injection

This module implements the complete Parse → Canonicalize → Transform → Unparse
pipeline for a single bug (softmax no-subtract-max), without any generic infrastructure.

Goal: Prove the architecture is viable before building generic tooling.
"""

from __future__ import annotations
import ast
from typing import Optional


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


class SoftmaxBugInjector(ast.NodeTransformer):
    """
    Hardcoded bug injector that finds and removes the subtract-max trick.
    
    Looks for TWO patterns in sequence:
    1. _varN = _varM.max(...)  (computing the max)
    2. _varP = _varM - _varN   (subtract-max trick)
    
    Replaces pattern 2 with: _varP = _varM
    """
    
    def __init__(self):
        self.bug_injected = False
        self.max_vars = {}  # Maps var name -> the variable it's the max of
        
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
    
    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Track max assignments and find subtract-max pattern."""
        print(f"  Checking: {ast.unparse(node)[:60]}")
        
        # First, check if this assignment computes a max
        # Pattern: _varN = _varM.max(...)
        is_max, source_var = self._is_max_call(node.value)
        if is_max and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            max_var_name = node.targets[0].id
            self.max_vars[max_var_name] = source_var
            print(f"    Tracked: {max_var_name} = max of {source_var}")
            return node
        
        # Second, check if this is a subtract-max pattern
        # Pattern: _varP = _varM - _varN (where _varN is max of _varM)
        if not isinstance(node.value, ast.BinOp):
            return node
        
        if not isinstance(node.value.op, ast.Sub):
            return node
        
        binop = node.value
        
        # Both sides should be Name nodes
        if not isinstance(binop.left, ast.Name) or not isinstance(binop.right, ast.Name):
            return node
        
        left_var = binop.left.id
        right_var = binop.right.id
        
        # Check if right_var is the max of left_var
        if right_var in self.max_vars and self.max_vars[right_var] == left_var:
            # FOUND IT! _varP = _varM - max(_varM)
            print(f"🐛 BUG INJECTED: Removing subtract-max trick")
            print(f"   Pattern: {left_var} - {right_var} (where {right_var} = max of {left_var})")
            print(f"   Becomes: {left_var}")
            
            node.value = ast.Name(id=left_var, ctx=ast.Load())
            self.bug_injected = True
        
        return node


def inject_softmax_bug(source_code: str) -> tuple[str, bool]:
    """
    Complete pipeline: Parse → Canonicalize → Inject Bug → Unparse.
    
    Args:
        source_code: Student's correct softmax implementation
        
    Returns:
        Tuple of (buggy_code, success)
        - buggy_code: Modified code with bug injected
        - success: True if bug was successfully injected
        
    Raises:
        SyntaxError: If source code is not valid Python
        ValueError: If softmax function not found
    """
    # Step 1: Parse student's code to AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise SyntaxError(f"Student code is not valid Python: {e}")
    
    # Step 2: Find the softmax function
    softmax_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'softmax':
            softmax_found = True
            break
    
    if not softmax_found:
        raise ValueError("No 'softmax' function found in source code")
    
    # Step 3: Canonicalize (standardize variable names)
    print("📝 Canonicalizing student code...")
    canonicalizer = SoftmaxCanonicalizer()
    canonical_tree = canonicalizer.visit(tree)
    ast.fix_missing_locations(canonical_tree)
    
    # Debug: Show canonical form
    canonical_code = ast.unparse(canonical_tree)
    print(f"✅ Canonical form created ({len(canonicalizer.var_mapping)} variables renamed)")
    
    # Step 4: Inject bug
    print("💉 Injecting bug...")
    injector = SoftmaxBugInjector()
    buggy_tree = injector.visit(canonical_tree)
    ast.fix_missing_locations(buggy_tree)
    
    if not injector.bug_injected:
        print("⚠️  WARNING: Bug pattern not found in canonical code!")
        print("   The student's implementation may not use the subtract-max trick,")
        print("   or may use a different coding style.")
        return canonical_code, False
    
    # Step 5: Unparse back to code
    buggy_code = ast.unparse(buggy_tree)
    print("✅ Bug successfully injected")
    
    return buggy_code, True


# Example usage and testing
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
    
    print("=" * 60)
    print("TEST 1: Standard implementation")
    print("=" * 60)
    buggy, success = inject_softmax_bug(test_code_1)
    print("\nBUGGY CODE:")
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
    
    print("\n" + "=" * 60)
    print("TEST 2: Different variable names")
    print("=" * 60)
    buggy, success = inject_softmax_bug(test_code_2)
    print("\nBUGGY CODE:")
    print(buggy)
    print(f"\nSuccess: {success}")
