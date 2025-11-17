# Competitive Programming Accelerator Curriculum

**Systematic rating-based progression from 0 to 1899 via pattern mastery**

## Overview

This curriculum fuses two authoritative sources:
- **CP Roadmap**: Rating-based topic progression (what to learn when)
- **DSA Taxonomy**: Hierarchical pattern breakdown (how to practice each topic)

Powered by the Mastery Engine's **Build-Justify-Harden** loop for deep, resilient skill development.

## Architecture

### The Canonical Source of Truth

**CRITICAL**: This curriculum uses a **single source of truth** architecture:

```
canonical_curriculum.json
    ↓ (generates)
manifest.json
    ↓ (powers)
Mastery Engine
```

#### `canonical_curriculum.json`
- **Human-curated**, machine-readable curriculum database
- Contains ALL curriculum information: topics, dependencies, resources, problems
- Structured, analyzable, version-controlled
- **THIS IS THE ONLY FILE YOU SHOULD EDIT** to modify the curriculum

#### `manifest.json`
- **Auto-generated** from canonical source via `scripts/generate_manifest.py`
- **NEVER EDIT MANUALLY** - Manual edits will fail CI
- Mastery Engine's runtime configuration
- Regenerated automatically on every curriculum change

### Why This Architecture?

**Problem**: Previous approach tried to generate structured artifacts from unstructured text sources, leading to:
- Ambiguous dependencies
- Monolithic modules (entire "Dynamic Programming" as one module)
- Unverifiable metadata

**Solution**: Establish a canonical, structured source that can be:
- Validated programmatically (dependency cycles, schema compliance)
- Analyzed (graph theory on dependencies)
- Generated from deterministically (manifest.json)

## Curriculum Structure

### Rating-Based Progression

Modules are organized by Codeforces rating brackets:

| Bracket | Focus | Example Modules |
|---------|-------|-----------------|
| **0-999** | Foundation | Sorting, Two Pointers (Opposite), Strings |
| **1000-1199** | Core Algorithms | Binary Search, Hash Tables, Sliding Window |
| **1200-1399** | Recursion & DP | Recursion Fundamentals, DP Foundations |
| **1400-1599** | Advanced Patterns | DP Knapsack, Binary Search on Answer, Graphs |
| **1600-1899** | Specialized | Segment Trees, Tries, Game Theory, Advanced DP |

### Granular Module Design

**Key Innovation**: Large topics are decomposed into rating-appropriate sub-modules.

**Example: Dynamic Programming**
```
Traditional (Monolithic):
└── dynamic_programming (20 hours, 1200-1899)  ❌ Too broad

Our Approach (Granular):
├── dp_foundations (8 hrs, 1200-1399)          ✅ Recursion → Memoization
├── dp_knapsack (8 hrs, 1400-1599)             ✅ 0/1 and Unbounded
├── dp_on_grids (6 hrs, 1400-1599)             ✅ Path counting
├── dp_with_binary_search (10 hrs, 1600-1899) ✅ Advanced optimization
```

**Benefits**:
- Progressive difficulty within a topic
- Precise dependencies (only advanced DP needs binary search)
- Manageable module size (5-10 hours each)

## Dependencies

Dependencies are validated to be:
1. **Acyclic**: No circular dependencies (verified via topological sort)
2. **Precise**: Only true prerequisites, not just related concepts
3. **Pedagogically sound**: Reviewed by competitive programming experts

**Example Dependencies**:
```
sorting ──────┬──> two_pointers_basics ──> two_pointers_sliding_window
              │
              └──> binary_search_on_index ──> binary_search_on_answer
              
recursion_basics ──> dp_foundations ──┬──> dp_knapsack
                                      │
                                      └──> dp_on_grids
                                      
binary_search_on_answer + dp_foundations ──> dp_with_binary_search
```

## Working with the Curriculum

### Making Changes

**To add/modify a module:**

1. **Edit the canonical source**:
   ```bash
   vim curricula/cp_accelerator/canonical_curriculum.json
   ```

2. **Validate your changes**:
   ```bash
   uv run python scripts/generate_manifest.py --validate-only
   ```
   This checks for:
   - Missing dependency IDs
   - Circular dependencies
   - Schema compliance

3. **Regenerate the manifest**:
   ```bash
   uv run python scripts/generate_manifest.py
   ```

4. **Commit BOTH files**:
   ```bash
   git add curricula/cp_accelerator/canonical_curriculum.json
   git add curricula/cp_accelerator/manifest.json
   git commit -m "curriculum: Add graph shortest paths module"
   ```

### CI Enforcement

The CI pipeline (`validate_cp_manifest.yml`) enforces curriculum integrity:

1. **Validation**: Checks canonical source for cycles and missing dependencies
2. **Regeneration**: Generates manifest.json from canonical source
3. **Diff Check**: **Fails if manifest.json was manually edited**
4. **Schema Check**: Validates JSON structure

**Result**: Impossible to merge a PR with:
- Circular dependencies
- Manually edited manifest
- Invalid schema

## Module Content Structure

Each module directory follows this structure:

```
modules/<module_id>/
├── build_prompt.txt           # Theory + canonical problem
├── justify_questions.json     # Deep conceptual questions
├── validator.sh               # Local test runner
├── test_cases.json            # Example test cases
└── bugs/
    ├── bug_name.patch         # Bug definition (source)
    ├── bug_name.json          # Bug injection pattern (compiled)
    └── bug_name_symptom.txt   # Student-facing error description
```

## Content Generation Pipeline

### Phase 1: Scaffolding (Automated)

`scripts/ingest_cp_content.py` generates initial content:

```bash
uv run python scripts/ingest_cp_content.py --module two_pointers_basics
```

This creates:
- `build_prompt.txt` from taxonomy + roadmap resources
- `test_cases.json` from problem examples
- `validator.sh` (standardized template)
- Scaffolded `justify_questions.json`

### Phase 2: Curation (Manual)

Human expert refines:
1. **Justify Questions**: Deepen Socratic questioning
2. **Bug Creation**: 
   - Write reference solution (`.solutions/`)
   - Create buggy variants
   - Generate `.patch` files: `diff -u clean.py buggy.py > bug.patch`
   - Write realistic `_symptom.txt` files
3. **Validation**: Test all bugs inject correctly

## Quality Standards

### Canonical Curriculum
- ✅ All dependency IDs must exist
- ✅ No circular dependencies
- ✅ Rating brackets follow Roadmap
- ✅ Granular modules (5-10 hours each)
- ✅ Precise, pedagogically sound dependencies

### Module Content
- ✅ Build prompt synthesizes theory + canonical problem
- ✅ Justify questions test deep understanding (not memorization)
- ✅ Local validator catches common errors
- ✅ 2-3 bugs per module demonstrating core pitfalls
- ✅ Bug symptoms are realistic (e.g., "WA on test 5", not generic errors)

## Current Status

**Version**: 1.0.0  
**Last Updated**: 2025-11-17  
**Modules**: 11 (foundation complete)

### Completed Modules
- ✅ `sorting` - Merge sort with bug injection
- 🔄 `two_pointers_basics` - Scaffolded, needs curation
- 🔄 `two_pointers_sliding_window` - Scaffolded
- 🔄 `hash_table_basics` - Scaffolded
- 🔄 ... 7 more foundation modules

### Roadmap
- **Short-term**: Complete all 11 foundation modules (0-1399 rating)
- **Medium-term**: Expand to 19 modules (full DSA Taxonomy coverage)
- **Long-term**: Add advanced modules (1900+ rating)

## Sources

- **Roadmap**: [CP Rating-Based Guide](https://docs.google.com/document/d/1-7Co93b504uyXyMjjE8bnLJP3d3QXvp_m1UjvbvdR2Y)
- **Taxonomy**: [DSA-Taxonomies Repository](https://github.com/Yassir-aykhlf/DSA-Taxonomies)

## Contributing

See [`docs/current/BUG_INJECTION_GUIDE.md`](../../docs/current/BUG_INJECTION_GUIDE.md) for curriculum authoring guidelines.

**Golden Rule**: Never edit `manifest.json` directly. Always edit `canonical_curriculum.json` and regenerate.
