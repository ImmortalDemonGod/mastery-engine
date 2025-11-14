#!/usr/bin/env python3
"""
Automatically fix draft patterns based on patch analysis.
Uses working patterns as templates.
"""

import json
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.ast_harden.generic_injector import GenericBugInjector


def extract_patch_code(patch_file: Path) -> tuple[str, str]:
    """Extract BEFORE and AFTER from patch."""
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
    
    return ''.join(before_lines), ''.join(after_lines)


def fix_linear_missing_transpose(draft_file: Path) -> dict:
    """Fix linear module - replace matmul call."""
    return {
        "id": "linear-missing-transpose",
        "description": "Missing transpose on weight matrix in linear layer",
        "injection_type": "ast",
        "engine_version": "2.1",
        "target_function": "forward",
        "logic": [{
            "pass": 1,
            "type": "find_and_replace",
            "description": "Replace in_features.matmul(self.weight.t()) with in_features.matmul(self.weight)",
            "pattern": {
                "node_type": "Assign",
                "targets": [{"node_type": "Name", "id": "y"}],
                "value": {"node_type": "Call"}
            },
            "replacement": {
                "type": "replace_value_with",
                "source": "in_features.matmul(self.weight)"
            }
        }],
        "metadata": {
            "created": "2025-11-14",
            "version": "2.1",
            "author": "auto_fixed"
        }
    }


def fix_transformer_block_missing_residual(draft_file: Path) -> dict:
    """Fix transformer_block - remove residual connection."""
    return {
        "id": "transformer-block-missing-residual",
        "description": "Missing residual connection in transformer block",
        "injection_type": "ast",
        "engine_version": "2.1",
        "target_function": "forward",
        "logic": [{
            "pass": 1,
            "type": "find_and_replace",
            "description": "Replace x = x + attn_out with x = attn_out",
            "pattern": {
                "node_type": "Assign",
                "targets": [{"node_type": "Name", "id": "x"}],
                "value": {"node_type": "BinOp", "op": "Add"}
            },
            "replacement": {
                "type": "replace_value_with",
                "source": "attn_out"
            }
        }],
        "metadata": {
            "created": "2025-11-14",
            "version": "2.1",
            "author": "auto_fixed"
        }
    }


def fix_training_loop_missing_zero_grad(draft_file: Path) -> dict:
    """Fix training_loop - delete zero_grad call."""
    return {
        "id": "training-loop-missing-zero-grad",
        "description": "Missing optimizer.zero_grad() in training loop",
        "injection_type": "ast",
        "engine_version": "2.1",
        "target_function": "train_step",
        "logic": [{
            "pass": 1,
            "type": "find_and_replace",
            "description": "Delete optimizer.zero_grad() call",
            "pattern": {
                "node_type": "Expr",
                "value": {
                    "node_type": "Call",
                    "func": {"node_type": "Attribute", "attr": "zero_grad"}
                }
            },
            "replacement": {
                "type": "delete_statement"
            }
        }],
        "metadata": {
            "created": "2025-11-14",
            "version": "2.1",
            "author": "auto_fixed"
        }
    }


def fix_swiglu_missing_gate(draft_file: Path) -> dict:
    """Fix swiglu - remove gate computation."""
    return {
        "id": "swiglu-missing-gate",
        "description": "Missing gate computation in SwiGLU",
        "injection_type": "ast",
        "engine_version": "2.1",
        "target_function": "forward",
        "logic": [
            {
                "pass": 1,
                "type": "find_and_replace",
                "description": "Delete gate = self.w3(x)",
                "pattern": {
                    "node_type": "Assign",
                    "targets": [{"node_type": "Name", "id": "gate"}],
                    "value": {"node_type": "Call"}
                },
                "replacement": {"type": "delete_statement"}
            },
            {
                "pass": 2,
                "type": "find_and_replace",
                "description": "Delete gated = value * gate",
                "pattern": {
                    "node_type": "Assign",
                    "targets": [{"node_type": "Name", "id": "gated"}],
                    "value": {"node_type": "BinOp", "op": "Mult"}
                },
                "replacement": {"type": "delete_statement"}
            },
            {
                "pass": 3,
                "type": "find_and_replace",
                "description": "Replace return with just value",
                "pattern": {
                    "node_type": "Return",
                    "value": {"node_type": "Call"}
                },
                "replacement": {
                    "type": "replace_value_with",
                    "source": "value"
                }
            }
        ],
        "metadata": {
            "created": "2025-11-14",
            "version": "2.1",
            "author": "auto_fixed"
        }
    }


# Map module names to fix functions
AUTO_FIXES = {
    "linear": fix_linear_missing_transpose,
    "transformer_block": fix_transformer_block_missing_residual,
    "training_loop": fix_training_loop_missing_zero_grad,
    "swiglu": fix_swiglu_missing_gate,
}


def test_pattern(bug_def: dict, before: str, expected: str) -> tuple[bool, str]:
    """Test if pattern works."""
    try:
        injector = GenericBugInjector(bug_def)
        actual, success = injector.inject(before)
        
        if not success:
            return False, "Pattern matching failed"
        
        if actual.strip() == expected.strip():
            return True, "✅ Perfect match"
        else:
            return False, f"Output mismatch"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Auto-fix simple patterns."""
    
    base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")
    
    print("="*70)
    print("AUTOMATIC DRAFT FIXING")
    print("="*70)
    print(f"\nFixing {len(AUTO_FIXES)} modules automatically...")
    
    fixed = []
    failed = []
    
    for module_name, fix_func in AUTO_FIXES.items():
        draft_file = list(base_path.glob(f"{module_name}/bugs/*_draft.json"))
        if not draft_file:
            print(f"\n❌ {module_name}: No draft file found")
            failed.append(module_name)
            continue
        
        draft_file = draft_file[0]
        patch_file = draft_file.with_name(draft_file.stem.replace('_draft', '') + '.patch')
        final_file = draft_file.with_name(draft_file.stem.replace('_draft', '') + '.json')
        
        print(f"\n{'='*70}")
        print(f"MODULE: {module_name}")
        print(f"{'='*70}")
        
        # Extract patch
        try:
            before, after = extract_patch_code(patch_file)
        except Exception as e:
            print(f"❌ Could not extract patch: {e}")
            failed.append(module_name)
            continue
        
        # Generate fix
        try:
            fixed_pattern = fix_func(draft_file)
        except Exception as e:
            print(f"❌ Fix function failed: {e}")
            failed.append(module_name)
            continue
        
        # Test fix
        success, message = test_pattern(fixed_pattern, before, after)
        print(f"🧪 Test: {message}")
        
        if success:
            # Save
            with open(final_file, 'w') as f:
                json.dump(fixed_pattern, f, indent=2)
            print(f"✅ Saved to: {final_file.name}")
            fixed.append(module_name)
        else:
            print(f"❌ Fix didn't work: {message}")
            # Save draft for manual fixing
            with open(draft_file.with_name(draft_file.stem.replace('_draft', '_draft_v2') + '.json'), 'w') as f:
                json.dump(fixed_pattern, f, indent=2)
            failed.append(module_name)
    
    print(f"\n{'='*70}")
    print("AUTO-FIX SUMMARY")
    print(f"{'='*70}")
    print(f"\n✅ Fixed: {len(fixed)}/{len(AUTO_FIXES)}")
    for m in fixed:
        print(f"   {m}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(AUTO_FIXES)}")
        for m in failed:
            print(f"   {m}")
    
    # Count total golden patterns
    all_golden = list(base_path.glob("*/bugs/*.json"))
    all_golden = [f for f in all_golden if "justify" not in f.name and "draft" not in f.name and f.stat().st_size > 100]
    
    print(f"\n📊 TOTAL GOLDEN PATTERNS: {len(all_golden)}/21")


if __name__ == "__main__":
    main()
