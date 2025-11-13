# Justify-Only Module Design

## Purpose

Extend the Build-Justify-Harden (BJH) framework to support **theory-only** modules that assess conceptual understanding without requiring implementation.

## Motivation

Not all learning is implementation. The curriculum includes foundational concepts that:
- **Are prerequisite knowledge** (e.g., Unicode fundamentals before BPE)
- **Lack clear implementation tasks** (e.g., UTF-8 encoding theory)
- **Require formal assessment** (can't be skipped)

Example: Unicode encoding concepts are essential for understanding byte-level tokenization, but implementing UTF-8 encoding from scratch is not a learning objective (Python handles it).

## Design

### Module Structure

Justify-only modules contain **only** `justify_questions.json`:

```
modules/unicode/
├── README.md                    # Explains theory-only nature
├── justify_questions.json       # Assessment questions
└── (no build_prompt.txt)        # No implementation task
└── (no validator.sh)            # No code to validate
└── (no bugs/)                   # No code to harden
```

### Manifest Declaration

Modules declare their type in `manifest.json`:

```json
{
  "id": "unicode",
  "name": "Unicode and Text Encoding Foundations",
  "path": "modules/unicode",
  "baseline_perf_seconds": 0.15,
  "dependencies": [],
  "module_type": "justify_only"
}
```

**Field**: `module_type` (optional, default: "standard")
- `"standard"`: Full BJH cycle (build → justify → harden)
- `"justify_only"`: Justify stage only (skip build/harden)

### Stage Progression

For `module_type: "justify_only"` modules:

1. **Initialization**: Start at "justify" stage (not "build")
2. **Completion**: Justify pass → next module (skip "harden")
3. **User Flow**:
   ```
   Standard module:  build → justify → harden → next
   Justify-only:                justify         → next
   ```

### Implementation Requirements

#### 1. Schema Updates (`engine/schemas.py`)

Add `module_type` field to `ModuleMetadata`:

```python
class ModuleMetadata(BaseModel):
    id: str
    name: str
    path: str
    baseline_perf_seconds: Optional[float] = None
    dependencies: list[str] = Field(default_factory=list)
    module_type: str = "standard"  # NEW: "standard" | "justify_only"
```

#### 2. State Management (`engine/state.py`)

Update `mark_stage_complete` to handle justify-only:

```python
def mark_stage_complete(self, stage: str, module_type: str = "standard") -> None:
    """Advance to next stage, accounting for module type."""
    if module_type == "justify_only":
        # Justify-only: justify → next module
        if stage == "justify":
            module_id = f"module_{self.current_module_index}"
            if module_id not in self.completed_modules:
                self.completed_modules.append(module_id)
            self.current_module_index += 1
            self.current_stage = "justify"  # Start next module at justify or build
        return
    
    # Standard: build → justify → harden → next
    if stage == "build":
        self.current_stage = "justify"
    elif stage == "justify":
        self.current_stage = "harden"
    elif stage == "harden":
        module_id = f"module_{self.current_module_index}"
        if module_id not in self.completed_modules:
            self.completed_modules.append(module_id)
        self.current_module_index += 1
        self.current_stage = "build"
```

#### 3. Module Initialization (`engine/curriculum.py`)

When starting a justify-only module, set initial stage correctly:

```python
def get_initial_stage(self, module: ModuleMetadata) -> str:
    """Determine starting stage based on module type."""
    if module.module_type == "justify_only":
        return "justify"
    return "build"
```

#### 4. Command Validation (`engine/main.py`)

Commands like `build`, `harden` should error on justify-only modules:

```python
def show_build():
    # ... existing checks ...
    if current_module.module_type == "justify_only":
        raise InvalidStageError(
            f"Module '{current_module.name}' is theory-only and has no build stage.\n"
            f"Use 'mastery justify' to answer conceptual questions."
        )
```

#### 5. File Existence Checks

Update all file path lookups to handle missing build_prompt.txt, validator.sh, bugs/:

```python
def get_build_prompt_path(self, curriculum_id: str, module_metadata: ModuleMetadata) -> Path:
    """Get path to build prompt, or None for justify-only modules."""
    if module_metadata.module_type == "justify_only":
        return None
    return self.get_module_path(curriculum_id, module_metadata) / "build_prompt.txt"
```

### User Experience

#### Standard Module
```bash
$ mastery status
📍 Current: Module 1/21 - Softmax (Build stage)

$ mastery build
[Shows build_prompt.txt]

$ mastery submit
✓ Tests pass! Moving to Justify stage.

$ mastery justify
[Asks conceptual questions]

$ mastery harden
[Provides buggy code to fix]

$ mastery status
📍 Current: Module 2/21 - Cross-Entropy (Build stage)
```

#### Justify-Only Module
```bash
$ mastery status
📍 Current: Module 19/22 - Unicode Foundations (Justify stage - Theory Only)

$ mastery build
✗ Error: Module 'Unicode Foundations' is theory-only and has no build stage.
  Use 'mastery justify' to answer conceptual questions.

$ mastery justify
[Asks Unicode theory questions]

$ mastery justify
✓ All questions answered correctly! Module complete.

$ mastery status
📍 Current: Module 20/22 - BPE Tokenizer (Build stage)
```

### Testing

Add test cases for justify-only modules:

```python
def test_justify_only_module_skips_build():
    """Verify justify-only modules start at justify stage."""
    progress = UserProgress(
        curriculum_id="cs336_a1",
        current_module_index=18,  # Unicode module
        current_stage="build"
    )
    # Should auto-transition to justify when module_type detected
    assert progress.current_stage == "justify"

def test_justify_only_module_skips_harden():
    """Verify justify stage completion advances to next module."""
    progress = UserProgress(
        curriculum_id="cs336_a1",
        current_module_index=18,  # Unicode module
        current_stage="justify"
    )
    progress.mark_stage_complete("justify", module_type="justify_only")
    assert progress.current_module_index == 19  # Next module
    assert progress.current_stage == "build"  # Standard next module starts at build
```

### Migration Path

**Backward Compatibility**:
- Default `module_type: "standard"` means existing manifests work unchanged
- Optional field, no breaking changes

**Incremental Rollout**:
1. Add schema field (non-breaking)
2. Update state management (handles both types)
3. Update UI/commands (clear error messages)
4. Add tests
5. Document feature
6. Create justify-only modules (unicode, etc.)

## Current Implementation Status

**Completed**:
- ✅ Unicode module created (`curricula/cs336_a1/modules/unicode/`)
- ✅ 5 comprehensive Unicode questions in `justify_questions.json`
- ✅ README explaining theory-only nature
- ✅ Manifest updated with `module_type: "justify_only"` field
- ✅ Unicode set as dependency for `bpe_tokenizer`

**Pending Engine Implementation**:
- ⏸️ Schema updates to support `module_type` field
- ⏸️ State management updates for justify-only progression
- ⏸️ Command validation (error on `build`/`harden` for justify-only)
- ⏸️ File path handling (graceful missing build_prompt.txt)
- ⏸️ Tests for justify-only module flow

**Workaround Until Engine Supports**:

Until engine implementation, the unicode module can be handled manually:
1. User reaches module 19 (unicode)
2. Current stage shows "build" (engine default)
3. User runs `mastery justify` (questions load correctly)
4. After passing justify, manually advance: `mastery next` or edit state
5. Document this as temporary limitation

## Future Extensions

### Other Module Types

The `module_type` field enables future module types:

- `"build_only"`: Implementation without formal assessment (e.g., setup tasks)
- `"experiment"`: Design/execute experiments, interpret results
- `"reading"`: Read paper, summarize key findings
- `"project"`: Open-ended integration project

### Chaining Theory and Practice

Dependency graph with mixed types:

```
unicode (justify-only)
  ↓
bpe_tokenizer (standard)
  ↓
tokenizer_class (standard)
  ↓
tokenization_experiments (experiment)
```

## Benefits

1. **Pedagogical Flexibility**: Not all learning fits implementation mold
2. **Formal Assessment**: Theory can't be skipped (must pass justify)
3. **Efficient**: Avoids busywork implementations of known algorithms
4. **Extensible**: Framework for other module types (experiments, readings)

## Example: Unicode Module

**Context**: Before implementing byte-level BPE, students need deep Unicode understanding.

**Why Justify-Only?**:
- UTF-8 encoding exists in Python (no need to reimplement)
- Understanding is critical (why byte-level vs character-level)
- Theory connects to practical tokenization decisions

**Assessment**:
- 5 questions covering UTF-8, normalization, grapheme clusters, etc.
- Rigorous explanations required (not multiple choice)
- Must demonstrate deep understanding to pass

**Outcome**: Students enter BPE module with solid Unicode foundation, ready to understand byte-level tokenization rationale.

---

**Status**: Design complete, awaiting engine implementation.

**Next Steps**:
1. Implement schema changes
2. Update state management
3. Add command validation
4. Write tests
5. Document user-facing feature
