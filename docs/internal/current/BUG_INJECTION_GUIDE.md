# Bug Injection System Guide

**For Curriculum Authors**

## Overview

The Mastery Engine uses **runtime bug injection** to create realistic debugging challenges. Bugs are injected into the student's own correct code using AST pattern matching, not pre-written buggy files.

## The Two-Tier Architecture

Every bug requires TWO files:

### 1. `.patch` File (SOURCE OF TRUTH)
- Human-readable diff format
- Primary artifact, version-controlled
- Schema-independent (survives engine upgrades)
- Created with standard `diff -u correct.py buggy.py > bug.patch`

### 2. `.json` File (COMPILED ARTIFACT)
- AST-based declarative pattern
- Auto-generated from .patch using LLM tool
- Can be regenerated anytime (not sacred)
- Schema-dependent (tied to engine version)
- Consumed by GenericBugInjector at runtime

## How It Works

```
Student's correct code → GenericBugInjector.inject(bug.json) → Buggy code → Student debugs
```

**Why this approach:**
- Student debugs code THEY wrote (realistic)
- Bugs work on ANY correct implementation
- Engine upgrades just regenerate .json from .patch
- No AST knowledge required to author bugs

## Creating a Bug (4 Steps)

### Step 1: Write Buggy Version

```bash
cd curricula/my_curriculum/modules/my_module/
cp solution.py buggy_version.py
# Edit buggy_version.py to introduce the bug
```

### Step 2: Create .patch (Source Artifact)

```bash
diff -u solution.py buggy_version.py > bugs/my_bug.patch
```

### Step 3: Generate .json (Compiled Artifact)

```bash
engine create-bug my_module --patch bugs/my_bug.patch --output bugs/my_bug.json
```

The LLM tool will:
- Analyze the .patch diff
- Generate AST pattern to match the code
- Test injection on your solution
- Provide diagnostics if it fails

### Step 4: Write Symptom Description

```bash
cat > bugs/my_bug_symptom.txt << EOF
Runtime Error: Output array has missing elements

Expected: [1, 2, 3, 4, 5]
Got: [1, 2, 3]

Debug hint: Check that ALL elements are included in the final result.
EOF
```

## When LLM Tool Fails

The `create-bug` tool may fail for:
- Control flow modifications (if statement changes)
- Complex statement deletions (Expr nodes)
- Multiple transformation passes

**Solution**: Manually write the .json by studying examples in `curricula/cs336_a1/modules/*/bugs/*.json`

## AST Pattern Examples

### Statement Deletion

```json
{
  "pattern": {
    "node_type": "Expr",
    "value": {
      "node_type": "Call",
      "func": {
        "node_type": "Attribute",
        "attr": "extend",
        "value": {"node_type": "Name", "id": "result"}
      }
    }
  },
  "replacement": {"type": "delete_statement"}
}
```

### Value Replacement

```json
{
  "pattern": {
    "node_type": "Assign",
    "targets": [{"node_type": "Name", "id": "bias_correction1"}],
    "value": {"node_type": "BinOp", "op": "Sub"}
  },
  "replacement": {
    "type": "replace_value_with",
    "source": "lr"
  }
}
```

### Node Replacement

```json
{
  "pattern": {
    "node_type": "BinOp",
    "op": "Mult",
    "left": {"node_type": "Name"},
    "right": {"node_type": "Call"}
  },
  "replacement": {
    "type": "replace_with",
    "source": "node.right"
  }
}
```

### Keyword Argument Removal

```json
{
  "pattern": {
    "node_type": "Call",
    "func": {"node_type": "Attribute", "attr": "sum"}
  },
  "replacement": {
    "type": "remove_keyword_arg",
    "name": "keepdim"
  }
}
```

## Testing Your Bug

```python
from engine.ast_harden.generic_injector import GenericBugInjector
from pathlib import Path

# Load bug definition
bug_json = Path('curricula/my_module/bugs/my_bug.json')
solution = Path('curricula/my_module/solution.py').read_text()

# Test injection
injector = GenericBugInjector(bug_json)
buggy_code, success = injector.inject(solution, debug=True)

if success:
    print("✅ Bug injection successful")
    print(buggy_code)
else:
    print("❌ Bug injection failed - check diagnostics")
```

## File Organization

```
curricula/my_curriculum/modules/my_module/
├── bugs/
│   ├── my_bug.patch           # Source of truth (durable)
│   ├── my_bug.json            # Compiled pattern (regenerable)
│   └── my_bug_symptom.txt     # Student-facing description
├── solution.py                 # Your implementation
└── test_cases.json            # Validation tests
```

## Golden Examples

Study these working bugs in cs336_a1:

**Simple Patterns:**
- `silu/bugs/missing_multiply.json` - Replace with path
- `linear/bugs/missing_transpose.json` - Replace value

**Medium Complexity:**
- `rmsnorm/bugs/missing_keepdim.json` - Remove keyword arg
- `softmax/bugs/no_subtract_max.json` - Multi-pass tracking

**Complex Patterns:**
- `attention/bugs/missing_scale.json` - Multiple deletions
- `adamw/bugs/missing_bias_correction.json` - Mixed operations

## Schema Evolution

When engine upgrades to v3.0:
1. .patch files remain unchanged (source of truth)
2. Regenerate all .json files: `engine regenerate-bugs --all`
3. Test each bug injection
4. Commit new .json files

## Troubleshooting

**Pattern not found:**
- Check that pattern matches BEFORE code (not AFTER)
- Use `debug=True` to see AST structure
- Simplify pattern (avoid over-specification)
- Add specific variable names with "id" field

**Multiple matches:**
- Add disambiguating operator ("op" field)
- Use specific variable "id" fields
- Check if variable is assigned multiple times

**Injection fails:**
- Verify .patch shows the intended transformation
- Test with full function (not code snippet)
- Check for indentation issues
- Try manual .json if LLM tool fails

## Best Practices

1. **One bug per file** - Keep bugs focused
2. **Test on clean code** - Verify injection works
3. **Write clear symptoms** - Student-facing, actionable
4. **Use simple patterns** - Avoid over-specification
5. **Version control .patch** - It's the source of truth
6. **Regenerate .json** - When schema changes
