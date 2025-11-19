#!/usr/bin/env python3
"""
Test script to verify LIBRARY curriculum loading.

This validates that the CurriculumManager can:
1. Load a LIBRARY type curriculum
2. Build lookup caches correctly
3. Resolve problem paths
4. Resolve pattern theory paths
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.curriculum import CurriculumManager
from engine.schemas import CurriculumType


def test_library_loading():
    """Test loading and accessing CP Accelerator LIBRARY curriculum."""
    print("="*60)
    print("LIBRARY CURRICULUM LOADING TEST")
    print("="*60)
    
    # Initialize manager
    print("\n1. Initializing CurriculumManager...")
    manager = CurriculumManager()
    print("   ✓ Manager initialized")
    
    # Load manifest
    print("\n2. Loading cp_accelerator manifest...")
    manifest = manager.load_manifest('cp_accelerator')
    print(f"   ✓ Manifest loaded: {manifest.curriculum_name}")
    
    # Verify type
    print("\n3. Verifying curriculum type...")
    assert manifest.type == CurriculumType.LIBRARY, f"Expected LIBRARY, got {manifest.type}"
    print(f"   ✓ Type: {manifest.type}")
    
    # Verify patterns loaded
    print("\n4. Verifying patterns...")
    assert manifest.patterns is not None, "Patterns should not be None"
    pattern_count = len(manifest.patterns)
    print(f"   ✓ Loaded {pattern_count} patterns")
    
    # Count total problems
    total_problems = sum(len(p.problems) for p in manifest.patterns)
    print(f"   ✓ Total problems: {total_problems}")
    
    # Test problem lookup
    print("\n5. Testing problem path resolution...")
    
    # Note: Some problems appear in multiple patterns (e.g., lc_912 in both sorting and divide_conquer)
    # Cache stores the last pattern encountered during iteration
    test_problems = ['lc_912', 'lc_1']
    
    for problem_id in test_problems:
        path = manager.get_problem_path('cp_accelerator', problem_id)
        assert path is not None, f"Problem {problem_id} not found"
        assert 'problems/' in str(path), f"Path should contain 'problems/', got {path}"
        assert problem_id in str(path), f"Path should contain problem ID {problem_id}, got {path}"
        
        # Get which pattern it resolved to
        pattern_id, _ = manager.get_problem_metadata(problem_id)
        print(f"   ✓ {problem_id}: {path} (pattern: {pattern_id})")
    
    # Test pattern theory path
    print("\n6. Testing pattern theory path resolution...")
    
    test_patterns = [
        ('sorting', 'patterns/sorting/theory'),
        ('hash_table', 'patterns/hash_table/theory'),
    ]
    
    for pattern_id, expected_suffix in test_patterns:
        path = manager.get_pattern_theory_path('cp_accelerator', pattern_id)
        assert path is not None, f"Pattern {pattern_id} not found"
        assert str(path).endswith(expected_suffix), f"Expected path to end with {expected_suffix}, got {path}"
        print(f"   ✓ {pattern_id}: {path}")
    
    # Test problem metadata lookup
    print("\n7. Testing problem metadata lookup...")
    pattern_id, problem_meta = manager.get_problem_metadata('lc_912')
    print(f"   ✓ lc_912 belongs to pattern: {pattern_id}")
    print(f"   ✓ Problem title: {problem_meta.title}")
    print(f"   ✓ Problem difficulty: {problem_meta.difficulty}")
    
    # Test pattern metadata lookup
    print("\n8. Testing pattern metadata lookup...")
    pattern_meta = manager.get_pattern_metadata('sorting')
    print(f"   ✓ Pattern title: {pattern_meta.title}")
    print(f"   ✓ Pattern has {len(pattern_meta.problems)} problems")
    
    # Test file existence
    print("\n9. Verifying actual file paths exist...")
    
    # Use lc_1 which only exists in hash_table pattern with our migrated content
    lc_1_path = manager.get_problem_path('cp_accelerator', 'lc_1')
    assert lc_1_path.exists(), f"Problem directory doesn't exist: {lc_1_path}"
    print(f"   ✓ Problem directory exists: {lc_1_path}")
    
    validator_path = lc_1_path / "validator.sh"
    assert validator_path.exists(), f"Validator doesn't exist: {validator_path}"
    print(f"   ✓ Validator exists: {validator_path}")
    
    # Check theory path for hash_table
    theory_path = manager.get_pattern_theory_path('cp_accelerator', 'hash_table')
    assert theory_path.exists(), f"Theory directory doesn't exist: {theory_path}"
    print(f"   ✓ Theory directory exists: {theory_path}")
    
    justify_path = theory_path / "justify_questions.json"
    assert justify_path.exists(), f"Justify questions don't exist: {justify_path}"
    print(f"   ✓ Justify questions exist: {justify_path}")
    
    # Success summary
    print("\n" + "="*60)
    print("✅ SUCCESS: All tests passed!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   - Curriculum: {manifest.curriculum_name}")
    print(f"   - Type: {manifest.type}")
    print(f"   - Patterns: {pattern_count}")
    print(f"   - Total problems: {total_problems}")
    print(f"   - Cache size: {len(manager._problem_cache)} problems indexed")
    print(f"\n✅ CurriculumManager successfully handles LIBRARY curricula")


if __name__ == '__main__':
    try:
        test_library_loading()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
