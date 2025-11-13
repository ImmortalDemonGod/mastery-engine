#!/usr/bin/env python3
"""
Batch Migration Script: Convert .patch files to .json using LLM

This script scans all curriculum modules and converts legacy .patch bug files
to the new v2.1 JSON format using the LLM bug authoring tool.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.dev_tools.bug_author import BugAuthor


def find_patch_files(curricula_path: Path) -> list[dict]:
    """Find all .patch files that need migration."""
    patch_files = []
    
    for module_dir in curricula_path.glob("*/modules/*"):
        bugs_dir = module_dir / "bugs"
        if not bugs_dir.exists():
            continue
        
        for patch_file in bugs_dir.glob("*.patch"):
            # Check if JSON already exists
            json_file = patch_file.with_suffix('.json')
            if json_file.exists():
                print(f"⏭️  Skipping {patch_file.name} (JSON exists)")
                continue
            
            # Look for symptom file
            symptom_file = bugs_dir.parent / "symptom.txt"
            if not symptom_file.exists():
                symptom_file = None
            
            patch_files.append({
                "module": module_dir.name,
                "patch_path": patch_file,
                "json_path": json_file,
                "symptom_path": symptom_file
            })
    
    return patch_files


def main():
    """Run batch migration."""
    print("="*60)
    print("BATCH BUG MIGRATION: .patch → .json")
    print("="*60)
    
    # Find curriculum root
    curricula_path = Path("curricula/cs336_a1")
    if not curricula_path.exists():
        print(f"❌ Curriculum path not found: {curricula_path}")
        sys.exit(1)
    
    # Find all patch files
    print(f"\n📁 Scanning {curricula_path}...")
    patch_files = find_patch_files(curricula_path)
    
    print(f"\n✨ Found {len(patch_files)} patch files to migrate\n")
    
    if not patch_files:
        print("✅ No patches need migration!")
        return
    
    # Initialize bug author
    print("🤖 Initializing LLM Bug Author...")
    author = BugAuthor()
    print(f"   Loaded {len(author.golden_dataset)} golden examples\n")
    
    # Track results
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    # Migrate each patch
    for i, patch_info in enumerate(patch_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(patch_files)}] {patch_info['module']}: {patch_info['patch_path'].name}")
        print(f"{'='*60}")
        
        # Load symptom
        symptom = ""
        if patch_info['symptom_path'] and patch_info['symptom_path'].exists():
            symptom = patch_info['symptom_path'].read_text()
            print(f"📝 Loaded symptom from {patch_info['symptom_path'].name}")
        else:
            symptom = f"Bug in {patch_info['module']} module"
            print(f"ℹ️  No symptom file, using default")
        
        # Generate bug definition
        bug_def, success = author.generate_bug_definition(
            module_name=patch_info['module'],
            patch_path=patch_info['patch_path'],
            symptom=symptom,
            max_retries=3
        )
        
        if success:
            # Write JSON file
            with open(patch_info['json_path'], 'w') as f:
                import json
                json.dump(bug_def, f, indent=2)
            
            print(f"\n✅ Success! Wrote {patch_info['json_path']}")
            results["success"].append(patch_info)
        else:
            print(f"\n❌ Failed to generate valid definition")
            results["failed"].append(patch_info)
    
    # Print summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    
    if results["failed"]:
        print("\n❌ Failed migrations:")
        for info in results["failed"]:
            print(f"   - {info['module']}: {info['patch_path'].name}")
        print("\nReview these manually or retry with adjusted prompts.")
    
    if results["success"]:
        print("\n✅ Successfully migrated:")
        for info in results["success"]:
            print(f"   - {info['json_path']}")
    
    print(f"\n{'='*60}")
    print(f"Total migrated: {len(results['success'])}/{len(patch_files)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
