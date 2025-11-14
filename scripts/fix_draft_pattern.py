#!/usr/bin/env python3
"""
Systematically fix draft patterns by showing:
1. What the patch shows (before/after)
2. What the LLM generated
3. Interactive fixing with LLM assistance
"""

import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.ast_harden.generic_injector import GenericBugInjector


def extract_patch_transformation(patch_file: Path) -> tuple[str, str]:
    """Extract before and after code from patch."""
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
            before_lines.append(line[1:])
            after_lines.append(line[1:])
    
    return ''.join(before_lines).strip(), ''.join(after_lines).strip()


def test_pattern(pattern_file: Path, before_code: str, expected_after: str) -> tuple[bool, str]:
    """Test if pattern works."""
    try:
        with open(pattern_file) as f:
            bug_def = json.load(f)
        
        injector = GenericBugInjector(bug_def)
        actual_after, success = injector.inject(before_code)
        
        if not success:
            return False, "Pattern matching failed - pattern not found in code"
        
        if actual_after.strip() == expected_after.strip():
            return True, "✅ Perfect match!"
        else:
            return False, f"Output mismatch:\nExpected:\n{expected_after[:200]}\n\nGot:\n{actual_after[:200]}"
    
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Interactive draft fixer."""
    
    base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")
    
    # Find all draft files
    draft_files = sorted(base_path.glob("*/bugs/*_draft.json"))
    
    if not draft_files:
        print("No draft files found!")
        return
    
    print("="*70)
    print(f"SYSTEMATIC DRAFT FIXING: {len(draft_files)} drafts to fix")
    print("="*70)
    
    for i, draft_file in enumerate(draft_files, 1):
        module_name = draft_file.parent.parent.name
        patch_file = draft_file.with_name(draft_file.stem.replace('_draft', '') + '.patch')
        final_file = draft_file.with_name(draft_file.stem.replace('_draft', '') + '.json')
        
        print(f"\n{'='*70}")
        print(f"{i}/{len(draft_files)}: {module_name}")
        print(f"{'='*70}")
        
        # Extract patch transformation
        try:
            before, after = extract_patch_transformation(patch_file)
        except Exception as e:
            print(f"❌ Could not extract patch: {e}")
            continue
        
        print(f"\n📝 PATCH TRANSFORMATION:")
        print(f"\nBEFORE ({len(before)} chars):")
        print(before[:300] if len(before) > 300 else before)
        print(f"\nAFTER ({len(after)} chars):")
        print(after[:300] if len(after) > 300 else after)
        
        # Show current draft
        with open(draft_file) as f:
            draft = json.load(f)
        
        print(f"\n📊 DRAFT PATTERN:")
        print(f"   Passes: {len(draft.get('logic', []))}")
        for j, pass_def in enumerate(draft.get('logic', []), 1):
            print(f"\n   Pass {j}: {pass_def.get('type', '?')}")
            print(f"     Pattern: {pass_def.get('pattern', {}).get('node_type', '?')}")
            if 'replacement' in pass_def:
                repl = pass_def['replacement']
                print(f"     Replacement: {repl.get('type', '?')}")
                if 'source' in repl:
                    print(f"       Source: {repl.get('source', 'N/A')[:100]}")
        
        # Test current draft
        success, message = test_pattern(draft_file, before, after)
        print(f"\n🧪 TEST RESULT: {message}")
        
        if success:
            print(f"\n✅ Draft already works! Copying to final location...")
            with open(final_file, 'w') as f:
                json.dump(draft, f, indent=2)
            print(f"   Saved to: {final_file.name}")
        else:
            print(f"\n⚠️  Draft needs fixing")
            print(f"   Draft saved at: {draft_file}")
            print(f"   Patch at: {patch_file}")
            print(f"\n💡 Hints:")
            print(f"   - Check pattern node_type matches actual code structure")
            print(f"   - Verify replacement source is correct")
            print(f"   - Compare to working examples: adamw, attention, rmsnorm")
            
            action = input(f"\n   Fix now? (y/n/skip all): ").strip().lower()
            
            if action == 'skip all':
                print("\n⏭️  Skipping remaining drafts")
                break
            elif action == 'y':
                print("\n📝 Open draft file to fix manually, then press Enter...")
                input("   Press Enter when done...")
                
                # Retest
                success, message = test_pattern(draft_file, before, after)
                if success:
                    print(f"\n✅ Fixed! Copying to final location...")
                    with open(final_file, 'w') as f:
                        with open(draft_file) as df:
                            draft = json.load(df)
                        json.dump(draft, f, indent=2)
                    print(f"   Saved to: {final_file.name}")
                else:
                    print(f"\n❌ Still failing: {message}")
    
    print(f"\n{'='*70}")
    print("DRAFT FIXING COMPLETE")
    print(f"{'='*70}")
    
    # Count fixed
    fixed = list(base_path.glob("*/bugs/*.json"))
    fixed = [f for f in fixed if "justify" not in f.name and "draft" not in f.name and f.stat().st_size > 100]
    
    print(f"\n✅ Total golden patterns: {len(fixed)}/21")
    for f in sorted(fixed):
        module = f.parent.parent.name
        print(f"   {module}")


if __name__ == "__main__":
    main()
