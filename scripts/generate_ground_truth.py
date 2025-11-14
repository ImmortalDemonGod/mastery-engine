#!/usr/bin/env python3
"""
Generate golden patterns for all curriculum modules systematically.
Uses gpt-4o to create patterns, then manually verify each one works.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.dev_tools.bug_author import BugAuthor
from engine.services.llm_service import LLMService
from engine.ast_harden.generic_injector import GenericBugInjector


def get_all_modules() -> list[Dict]:
    """Get all curriculum modules with their patch files."""
    base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")
    
    modules = []
    for module_dir in sorted(base_path.iterdir()):
        if not module_dir.is_dir():
            continue
        
        bugs_dir = module_dir / "bugs"
        if not bugs_dir.exists():
            continue
        
        # Find patch file
        patch_files = list(bugs_dir.glob("*.patch"))
        if not patch_files:
            continue
        
        # Find JSON file
        json_files = list(bugs_dir.glob("*.json"))
        if not json_files:
            print(f"⚠️  {module_dir.name}: No JSON file found")
            continue
        
        json_file = json_files[0]
        
        # Check if has golden pattern
        has_golden = False
        try:
            with open(json_file) as f:
                data = json.load(f)
                has_golden = "logic" in data and data["logic"]
        except:
            pass
        
        modules.append({
            "name": module_dir.name,
            "patch": patch_files[0],
            "json": json_file,
            "has_golden": has_golden
        })
    
    return modules


def test_golden_pattern(bug_def: dict, correct_code: str, expected_buggy: str) -> tuple[bool, str]:
    """Test if golden pattern works correctly."""
    try:
        injector = GenericBugInjector(bug_def)
        buggy_code, success = injector.inject(correct_code)
        
        if not success:
            return False, "Pattern matching failed - pattern not found in code"
        
        # Compare results (simple text comparison for now)
        buggy_normalized = buggy_code.strip()
        expected_normalized = expected_buggy.strip()
        
        if buggy_normalized == expected_normalized:
            return True, "✅ Transformation correct"
        else:
            # Show difference
            diff = f"Generated:\n{buggy_code}\n\nExpected:\n{expected_buggy}"
            return False, f"Output mismatch:\n{diff}"
    
    except Exception as e:
        return False, f"Error during injection: {e}"


def generate_golden_for_module(module_info: Dict, author: BugAuthor) -> Optional[dict]:
    """Generate golden pattern for a single module."""
    print(f"\n{'='*70}")
    print(f"MODULE: {module_info['name']}")
    print(f"{'='*70}")
    
    # Extract patch info
    try:
        patch_info = author._extract_patch_info(module_info['patch'])
        print(f"\n✅ Patch extracted successfully")
        print(f"   Before code: {len(patch_info['before'])} chars")
        print(f"   After code: {len(patch_info['after'])} chars")
        print(f"   Source file: {patch_info['source_file']}")
    except Exception as e:
        print(f"\n❌ Failed to extract patch: {e}")
        return None
    
    # Show the transformation
    print(f"\n📝 TRANSFORMATION:")
    print(f"\nBEFORE:\n{patch_info['before'][:200]}...")
    print(f"\nAFTER:\n{patch_info['after'][:200]}...")
    
    # Generate pattern with LLM
    print(f"\n🧠 Generating pattern with gpt-4o...")
    
    try:
        bug_def = author._generate_bug_definition_llm(
            patch_info=patch_info,
            module_name=module_info['name'],
            symptom=f"Bug in {module_info['name']}",
            feedback=None
        )
        
        if not bug_def:
            print(f"❌ LLM failed to generate pattern")
            return None
        
        print(f"✅ Pattern generated")
        
        # Show pattern summary
        logic = bug_def.get('logic', [])
        print(f"\n📊 PATTERN SUMMARY:")
        print(f"   Passes: {len(logic)}")
        for i, pass_def in enumerate(logic, 1):
            pattern = pass_def.get('pattern', {})
            repl = pass_def.get('replacement', {})
            print(f"     Pass {i}: {pass_def.get('type', '?')}")
            print(f"       Pattern: {pattern.get('node_type', '?')}")
            print(f"       Replacement: {repl.get('type', '?')}")
        
        # Test the pattern
        print(f"\n🧪 Testing pattern with GenericBugInjector...")
        success, message = test_golden_pattern(
            bug_def,
            patch_info['before'],
            patch_info['after']
        )
        
        if success:
            print(f"✅ Pattern works correctly!")
            return bug_def
        else:
            print(f"❌ Pattern failed: {message}")
            return None
    
    except Exception as e:
        print(f"❌ Error generating/testing pattern: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Generate golden patterns for all modules."""
    
    print("="*70)
    print("SYSTEMATIC GROUND TRUTH GENERATION")
    print("="*70)
    print("\nGoal: Create golden patterns for ALL 21 curriculum modules")
    print("Method: Use gpt-4o to generate, then verify each one\n")
    
    # Get all modules
    modules = get_all_modules()
    
    print(f"Found {len(modules)} modules with patches")
    
    # Separate by status
    has_golden = [m for m in modules if m['has_golden']]
    needs_golden = [m for m in modules if not m['has_golden']]
    
    print(f"\n✅ Already have golden patterns: {len(has_golden)}")
    for m in has_golden:
        print(f"   {m['name']}")
    
    print(f"\n⚠️  Need golden patterns: {len(needs_golden)}")
    for m in needs_golden:
        print(f"   {m['name']}")
    
    # Initialize bug author with gpt-4o
    print(f"\n🧠 Initializing with gpt-4o (smarter model)...")
    llm_service = LLMService(model="gpt-4o")
    author = BugAuthor(llm_service=llm_service)
    
    # Process each module that needs a golden pattern
    results = {
        'success': [],
        'failed': [],
        'total': len(needs_golden)
    }
    
    for i, module_info in enumerate(needs_golden, 1):
        print(f"\n\n{'='*70}")
        print(f"PROGRESS: {i}/{len(needs_golden)}")
        print(f"{'='*70}")
        
        golden_pattern = generate_golden_for_module(module_info, author)
        
        if golden_pattern:
            results['success'].append(module_info['name'])
            
            # Ask if user wants to save
            save = input(f"\n💾 Save this golden pattern to {module_info['json'].name}? (y/n): ").strip().lower()
            
            if save == 'y':
                # Save to JSON file
                with open(module_info['json'], 'w') as f:
                    json.dump(golden_pattern, f, indent=2)
                print(f"✅ Saved to {module_info['json']}")
            else:
                print(f"⏭️  Skipped saving")
        else:
            results['failed'].append(module_info['name'])
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("GROUND TRUTH GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Successfully generated: {len(results['success'])}/{results['total']}")
    for m in results['success']:
        print(f"   {m}")
    
    if results['failed']:
        print(f"\n❌ Failed to generate: {len(results['failed'])}/{results['total']}")
        for m in results['failed']:
            print(f"   {m}")
    
    total_golden = len(has_golden) + len(results['success'])
    print(f"\n📊 TOTAL GOLDEN PATTERNS: {total_golden}/21")
    print(f"   Coverage: {total_golden/21*100:.0f}%")


if __name__ == "__main__":
    main()
