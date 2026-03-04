#!/usr/bin/env python3
"""
Add successfully generated bug definitions to golden dataset.
Only adds if manual verification confirms they work correctly.
"""

import json
import sys
from pathlib import Path

def main():
    results_path = Path("/tmp/llm_evaluation_results.json")
    
    if not results_path.exists():
        print("❌ No results file found. Run evaluation first.")
        sys.exit(1)
    
    with open(results_path) as f:
        data = json.load(f)
    
    # Get successful bugs (including false negatives that actually work)
    successful_bugs = []
    for bug in data['results']:
        module = bug['module']
        
        # Check if any attempt succeeded OR is a false negative
        for attempt in bug['attempts']:
            if attempt['success'] or attempt.get('is_false_negative', False):
                successful_bugs.append({
                    'module': module,
                    'attempt_num': attempt['attempt_num'],
                    'response': attempt['response_text'],
                    'is_false_negative': attempt.get('is_false_negative', False)
                })
                break  # Only take first success
    
    print("="*70)
    print("SUCCESSFUL BUG DEFINITIONS TO ADD TO GOLDEN DATASET")
    print("="*70)
    print(f"\nFound {len(successful_bugs)} successful patterns\n")
    
    for i, bug_info in enumerate(successful_bugs, 1):
        module = bug_info['module']
        attempt = bug_info['attempt_num']
        is_fn = bug_info['is_false_negative']
        
        print(f"\n{i}. {module} (attempt {attempt})")
        if is_fn:
            print("   ⚠️  FALSE NEGATIVE - injection works, comparison issue")
        
        # Parse the response
        response = bug_info['response'].strip()
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            bug_def = json.loads(response)
            
            # Show summary
            logic = bug_def.get('logic', [])
            print(f"   Passes: {len(logic)}")
            for j, pass_def in enumerate(logic, 1):
                pattern = pass_def.get('pattern', {})
                repl = pass_def.get('replacement', {})
                print(f"     Pass {j}: {pattern.get('node_type', '?')} → {repl.get('type', '?')}")
            
            # Ask for confirmation
            add = input(f"\n   Add {module} to golden dataset? (y/n): ").strip().lower()
            
            if add == 'y':
                # Save to golden examples directory
                golden_dir = Path(f"/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules/{module}/bugs")
                
                # Find the bug JSON file
                bug_files = list(golden_dir.glob("*.json"))
                if bug_files:
                    bug_file = bug_files[0]
                    
                    # Save as golden example
                    with open(bug_file, 'w') as f:
                        json.dump(bug_def, f, indent=2)
                    
                    print(f"   ✅ Saved to {bug_file}")
                else:
                    print(f"   ⚠️  No JSON file found in {golden_dir}")
            else:
                print(f"   ⏭️  Skipped {module}")
                
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse JSON: {e}")
    
    print("\n" + "="*70)
    print("GOLDEN DATASET UPDATE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
