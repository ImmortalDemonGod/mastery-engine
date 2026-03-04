# Competitive Programming Accelerator Curriculum

**Systematic rating-based progression from 0 to 1899 via pattern mastery**

## 📜 Attribution

**Sources:** Problem data and taxonomy adapted from:
- **[LeetCode](https://leetcode.com)** - Problem descriptions, test cases, and difficulty ratings
- **[DSA Taxonomies](https://github.com/Yassir-aykhlf/DSA-Taxonomies)** - Hierarchical pattern classification by Yassir Aykhlf

### Content Ownership

- **Problem Descriptions:** Remain property of LeetCode LLC
- **Taxonomy Structure:** Derived from DSA Taxonomies repository
- **Problem Selection:** Curated via rating analysis and pattern distribution

### Mastery Engine Contributions (Original)

The following were **engineered for this platform**:
1. **Automated Breadth-First Generation:** Parsing 38 LeetCode problems into structured JSON schemas
2. **Pattern-Problem Mapping:** Automated linkage between abstract patterns and concrete problems
3. **LIBRARY Mode Implementation:** Freeform curriculum navigation (vs LINEAR sequences)
4. **Canonical Source Architecture:** Single source of truth via `canonical_curriculum.json`

## Overview

This curriculum fuses two authoritative sources for systematic skill development, powered by the Mastery Engine's **Build-Justify-Harden** loop.

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

**Version**: 1.0.0 (Demonstration)  
**Last Updated**: 2025-11-19  
**Problems**: 38 across 19 patterns (breadth-first)  

### 🎯 Demonstration Architecture Note

This curriculum serves as a **proof-of-concept for the automated content generation pipeline**. The system demonstrates its ability to:

1. **Parse** unstructured LeetCode problem data
2. **Transform** into structured JSON schemas  
3. **Generate** build prompts, test cases, and validators at scale
4. **Organize** content in the `patterns/{pattern}/problems/{problem}/` hierarchy

**Content Status:**
- ✅ `sorting/lc_912` - Fully implemented with manual curation
- ✅ `hash_table/lc_1` (Two Sum) - Manually curated reference implementation  
- 🤖 **36 other problems** - Auto-generated via `scripts/generate_module.py --all --limit-per-pattern 2`

**Why Breadth-First?**  
Rather than deeply implementing 2-3 patterns, we generated 2 problems per pattern across all 19 patterns to demonstrate:
- **Scalability**: Pipeline handles diverse problem types (graphs, DP, trees, etc.)
- **Consistency**: Generated artifacts follow standardized schemas
- **Taxonomy Coverage**: Full competitive programming skill map

**For Portfolio Reviewers:**  
This demonstrates data engineering and metaprogramming capabilities. The automated generation pipeline (`scripts/generate_module.py`) is the engineering achievement, not the individual problem implementations.

### Roadmap
- **Phase 8 Complete**: Breadth-first scaffolding (38 problems)
- **Future**: Depth-first curation (full justify questions + bug injection for each problem)

## Sources

- **Roadmap**: [CP Rating-Based Guide](https://docs.google.com/document/d/1-7Co93b504uyXyMjjE8bnLJP3d3QXvp_m1UjvbvdR2Y)
- **Taxonomy**: [DSA-Taxonomies Repository](https://github.com/Yassir-aykhlf/DSA-Taxonomies)

## Contributing

See [`docs/current/BUG_INJECTION_GUIDE.md`](../../docs/current/BUG_INJECTION_GUIDE.md) for curriculum authoring guidelines.

**Golden Rule**: Never edit `manifest.json` directly. Always edit `canonical_curriculum.json` and regenerate.
