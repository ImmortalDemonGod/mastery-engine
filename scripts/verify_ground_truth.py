#!/usr/bin/env python3
"""
Systematically verify all golden patterns work correctly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.ast_harden.generic_injector import GenericBugInjector


def get_all_golden_patterns() -> List[Dict]:
    """Get all modules with golden patterns."""
    base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")
    
    modules = []
    for module_dir in sorted(base_path.iterdir()):
        if not module_dir.is_dir():
            continue
        
        bugs_dir = module_dir / "bugs"
        if not bugs_dir.exists():
            continue
        
        # Find JSON files with logic
        for json_file in bugs_dir.glob("*.json"):
            if "justify" in json_file.name:
                continue
            
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if "logic" in data and data["logic"]:
                        # Find corresponding patch
                        patch_file = json_file.with_suffix('.patch')
                        if patch_file.exists():
                            modules.append({
                                "name": module_dir.name,
                                "json": json_file,
                                "patch": patch_file,
                                "pattern": data
                            })
            except:
                pass
    
    return modules


def extract_patch_code(patch_file: Path) -> tuple[str, str]:
    """Extract BEFORE and AFTER code from patch file."""
    with open(patch_file) as f:
        lines = f.readlines()
    
    before_lines = []
    after_lines = []
    in_diff = False
    
    for line in lines:
        if line.startswith('@@'):
            in_diff = True
            continue
        
        if not in_diff:
            continue
        
        if line.startswith('-') and not line.startswith('---'):
            before_lines.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            after_lines.append(line[1:])
        elif line.startswith(' '):
            # Context line
            before_lines.append(line[1:])
            after_lines.append(line[1:])
    
    return ''.join(before_lines), ''.join(after_lines)


def test_pattern(module_info: Dict) -> tuple[bool, str]:
    """Test if pattern works correctly."""
    try:
        # Get pattern
        bug_def = module_info['pattern']
        
        # Extract patch code
        before_code, expected_after = extract_patch_code(module_info['patch'])
        
        # Test injection
        injector = GenericBugInjector(bug_def)
        actual_after, success = injector.inject(before_code)
        
        if not success:
            return False, "❌ Pattern matching failed"
        
        # Compare (normalize whitespace)
        expected_norm = expected_after.strip()
        actual_norm = actual_after.strip()
        
        if expected_norm == actual_norm:
            return True, "✅ Perfect match"
        else:
            # Show difference
            return False, f"❌ Output mismatch:\n  Expected:\n{expected_norm[:200]}\n  Got:\n{actual_norm[:200]}"
    
    except Exception as e:
        return False, f"❌ Error: {e}"


def main():
    """Verify all golden patterns systematically."""
    
    print("="*70)
    print("SYSTEMATIC GROUND TRUTH VERIFICATION")
    print("="*70)
    
    # Get all modules with golden patterns
    modules = get_all_golden_patterns()
    
    print(f"\nFound {len(modules)} modules with golden patterns\n")
    
    results = {
        'passed': [],
        'failed': [],
        'total': len(modules)
    }
    
    # Test each module
    for i, module_info in enumerate(modules, 1):
        print(f"{i:2d}. {module_info['name']:25}", end=" ")
        
        success, message = test_pattern(module_info)
        
        if success:
            print(message)
            results['passed'].append(module_info['name'])
        else:
            print(message)
            results['failed'].append(module_info['name'])
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n✅ Passed: {len(results['passed'])}/{results['total']}")
    for m in results['passed']:
        print(f"   {m}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}/{results['total']}")
        for m in results['failed']:
            print(f"   {m}")
    
    print(f"\n📊 SUCCESS RATE: {len(results['passed'])/results['total']*100:.0f}%")
    
    return len(results['failed']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
