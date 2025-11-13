#!/usr/bin/env python3
"""
Validate that student mode files contain stubs, not complete implementations.

This script checks all Python files in modes/student/ to ensure they have
NotImplementedError or TODO comments, preventing accidental commits of
complete implementations that would defeat the learning system.

Usage:
    python scripts/validate_student_stubs.py
    
Exit codes:
    0: All student files are properly stubbed
    1: Found complete implementations in student mode
"""

from __future__ import annotations
import sys
from pathlib import Path
import ast
import re


class StubValidator(ast.NodeVisitor):
    """Check if a Python file contains proper stubs."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.functions_checked = 0
        self.functions_with_stubs = 0
        self.issues = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check each function for NotImplementedError or TODO."""
        # Skip private/magic methods
        if node.name.startswith('_'):
            self.generic_visit(node)
            return
            
        self.functions_checked += 1
        
        # Check for NotImplementedError
        has_not_implemented = False
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if stmt.exc.func.id == 'NotImplementedError':
                            has_not_implemented = True
                            break
        
        if has_not_implemented:
            self.functions_with_stubs += 1
        else:
            # Check for TODO comment in the function
            # We'll check this separately in the source
            pass
            
        self.generic_visit(node)


def check_file(filepath: Path) -> tuple[bool, list[str]]:
    """
    Check if a file contains proper stubs.
    
    Returns:
        (is_valid, issues)
    """
    try:
        source = filepath.read_text()
    except Exception as e:
        return False, [f"Could not read {filepath}: {e}"]
    
    # Parse the AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f"Syntax error in {filepath}: {e}"]
    
    # Run the validator
    validator = StubValidator(filepath)
    validator.visit(tree)
    
    # If no functions, that's okay (might be constants/classes only)
    if validator.functions_checked == 0:
        return True, []
    
    # Check for TODO comments
    has_todo = 'TODO' in source or 'TODO:' in source
    
    # If file has NotImplementedError or TODO, it's a stub
    if validator.functions_with_stubs > 0 or has_todo:
        return True, []
    
    # Check for suspicious patterns that indicate complete implementations
    suspicious_patterns = [
        (r'\breturn\s+\w+\.\w+\(', 'Contains method calls in return statements'),
        (r'for\s+\w+\s+in\s+', 'Contains for loops'),
        (r'while\s+', 'Contains while loops'),
        (r'if\s+.*:\s*\n\s+\w+\s*=', 'Contains conditional assignments'),
    ]
    
    issues = []
    for pattern, description in suspicious_patterns:
        if re.search(pattern, source):
            issues.append(f"{filepath.name}: {description} (likely not a stub)")
    
    if issues:
        return False, issues
    
    return True, []


def main():
    """Main validation logic."""
    repo_root = Path(__file__).parent.parent
    student_dir = repo_root / "modes" / "student"
    
    if not student_dir.exists():
        print("✅ No student mode directory found (okay)")
        return 0
    
    # Find all Python files in student mode
    python_files = list(student_dir.rglob("*.py"))
    
    # Exclude __init__.py, __pycache__, and example files
    python_files = [
        f for f in python_files 
        if (f.name != "__init__.py" and 
            "__pycache__" not in str(f) and
            "example" not in f.name.lower())
    ]
    
    if not python_files:
        print("✅ No Python files in student mode to validate")
        return 0
    
    print(f"Validating {len(python_files)} student mode files...")
    print()
    
    all_valid = True
    all_issues = []
    
    for filepath in sorted(python_files):
        rel_path = filepath.relative_to(repo_root)
        is_valid, issues = check_file(filepath)
        
        if is_valid:
            print(f"✅ {rel_path}")
        else:
            print(f"❌ {rel_path}")
            all_valid = False
            all_issues.extend(issues)
            for issue in issues:
                print(f"   {issue}")
    
    print()
    
    if all_valid:
        print("✅ All student mode files are properly stubbed!")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print()
        print("Student mode files contain complete implementations!")
        print("This would allow students to submit without writing code.")
        print()
        print("Please reset these files to stubs with NotImplementedError")
        print("or TODO comments before committing.")
        print()
        print("Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
