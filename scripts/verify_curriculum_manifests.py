#!/usr/bin/env python3
"""
Static verification script for curriculum manifests.
Validates that all declared modules have their required files present.
"""

import json
import sys
from pathlib import Path

def verify_curriculum_manifest(curriculum_path: Path) -> tuple[bool, list[str]]:
    """
    Verify curriculum manifest integrity.
    
    Returns:
        (success, errors): Tuple of success boolean and list of error messages
    """
    errors = []
    manifest_path = curriculum_path / "manifest.json"
    
    # Load manifest
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return False, [f"Failed to load manifest: {e}"]
    
    # Check each module
    modules = manifest.get("modules", [])
    print(f"Validating {len(modules)} modules...")
    
    for module in modules:
        module_id = module.get("id", "unknown")
        module_type = module.get("module_type", "standard")
        module_rel_path = module.get("path", "")
        module_path = curriculum_path / module_rel_path
        
        print(f"  Checking {module_id} (type: {module_type})...")
        
        if not module_path.exists():
            errors.append(f"Module directory missing: {module_id}")
            continue
        
        # Standard modules need: build_prompt.txt, validator.sh
        # justify_only modules need: justify_questions.json only
        # experiment modules need: experiment_spec.json only
        
        if module_type == "justify_only":
            # Justify-only module (e.g., unicode)
            required_files = ["justify_questions.json"]
            forbidden_files = ["build_prompt.txt", "validator.sh"]
            
            for file in required_files:
                if not (module_path / file).exists():
                    errors.append(f"{module_id}: Missing required file {file}")
            
            for file in forbidden_files:
                if (module_path / file).exists():
                    errors.append(
                        f"{module_id}: Should not have {file} (justify_only type)"
                    )
        
        elif module_type == "experiment":
            # Experiment module
            required_files = ["experiment_spec.json"]
            
            for file in required_files:
                if not (module_path / file).exists():
                    errors.append(f"{module_id}: Missing required file {file}")
        
        else:
            # Standard BJH module
            required_files = ["build_prompt.txt", "validator.sh"]
            
            for file in required_files:
                if not (module_path / file).exists():
                    errors.append(f"{module_id}: Missing required file {file}")
            
            # Check optional files (don't error, just warn)
            optional_files = ["justify_questions.json", "bug.patch"]
            for file in optional_files:
                if not (module_path / file).exists():
                    print(f"    ⚠️  Optional file missing: {file}")
    
    if errors:
        return False, errors
    else:
        print(f"✅ All {len(modules)} modules validated successfully!")
        return True, []


if __name__ == "__main__":
    curriculum_path = Path(__file__).parent.parent / "curricula" / "cs336_a1"
    
    success, errors = verify_curriculum_manifest(curriculum_path)
    
    if not success:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n✅ CURRICULUM MANIFEST VALIDATION PASSED")
        sys.exit(0)
