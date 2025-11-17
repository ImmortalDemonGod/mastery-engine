# CLI Command Reference

**Status**: Production-ready (78% test coverage)  
**Last Updated**: 2025-11-12

## Quick Reference

### Core Workflow Commands

```bash
# Initialize curriculum
engine init

# Check current status
engine status

# View current module or explore by ID
engine show [module_id]

# Submit work (auto-detects stage)
engine submit
```

### Curriculum Management

```bash
# List all modules with progress
engine curriculum-list

# Reset a specific module
engine progress-reset <module_id>

# Start harden challenge (harden stage only)
engine start-challenge
```

## Command Details

### `engine submit`
**Context-aware submission** - Auto-detects your current stage and validates accordingly.

**Build Stage**: Runs validator.sh to test your implementation  
**Justify Stage**: Opens $EDITOR for answers, validates with LLM  
**Harden Stage**: Tests your bug fix in shadow worktree

**Zero breaking changes**: Legacy commands (`submit-build`, `submit-justification`, `submit-fix`) still work.

### `engine show [module_id]`
**Read-only display** - Safe, idempotent inspection of module state.

- No arguments: Shows current module (build prompt, justify questions, or harden symptom)
- With module_id: Preview any module before starting it
- **Guarantees**: Never modifies state, safe to run anytime

### `engine start-challenge`
**Explicit harden initialization** - Only available in harden stage.

- Creates shadow worktree with your correct implementation
- Injects bug from module's .json pattern
- Displays symptom for you to debug
- **Safety**: Prompts for confirmation before creating worktree

### `engine curriculum-list`
**Progress overview** - View all modules with completion status.

- ✅ Completed modules
- 🔵 Current module
- ⚪ Pending modules
- Shows module IDs for use with `show` command

### `engine progress-reset <module_id>`
**Module reset** - Start a module over from scratch.

- Resets state to "not started"
- Removes harden worktree if exists
- **Preserves**: Your implementation files in modes/
- **Safety**: Interactive confirmation required

## Design Principles

1. **Context-awareness**: Commands detect your stage automatically
2. **Explicit intent**: Write operations require confirmation
3. **Read-only guarantee**: `show` never modifies state
4. **Backward compatibility**: Legacy commands maintained during transition

## Test Coverage

- Overall: 78% (production-ready)
- Core modules: 7 at 100%
- CLI handlers: 69%
- Pass rate: 145/145 (100%)

## Migration from Legacy Commands

| Legacy | New | Notes |
|--------|-----|-------|
| `submit-build` | `submit` | Auto-detects build stage |
| `submit-justification` | `submit` | Auto-detects justify stage |
| `submit-fix` | `submit` | Auto-detects harden stage |
| `next` (read) | `show` | Read-only, guaranteed safe |
| `next` (write) | `start-challenge` | Explicit, harden-only |

Legacy commands still work but are soft-deprecated.
