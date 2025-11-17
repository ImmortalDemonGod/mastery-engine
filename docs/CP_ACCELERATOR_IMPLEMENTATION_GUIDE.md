# CP Accelerator - Implementation Guide

**Status:** Foundation Complete | Phase 1 Ready  
**Date:** November 16, 2025

## Executive Summary

This guide documents the systematic implementation of the "Competitive Programming Accelerator" curriculum pack for the Mastery Engine. The design synthesizes two complementary sources:

1. **DSA Pattern Taxonomy** (GitHub) - Hierarchical knowledge base with 19 patterns and 1000+ LeetCode problems
2. **CP Roadmap** (Google Doc) - Rating-driven learning path (0-1900+) with curated resources

The result is a world-class, interactive curriculum that guides users from newbie to candidate master through the proven Build-Justify-Harden pedagogical loop.

---

## Architecture Overview

### Directory Structure

```
curricula/cp_accelerator/
├── manifest.json           # Master curriculum definition
└── modules/
    ├── sorting/
    │   ├── build_prompt.txt
    │   ├── justify_questions.json
    │   ├── validator.sh
    │   ├── test_cases.json
    │   ├── solution.py        # Student workspace
    │   └── bugs/
    │       ├── bug1.patch
    │       └── bug1_symptom.txt
    ├── two_pointers/
    └── ... (19 total patterns)

.solutions/cp_accelerator/
└── modules/
    └── two_pointers/
        └── solution.py        # Reference implementation

scripts/
└── ingest_cp_content.py      # Automated content generation
```

### Schema Extensions

**Modified:** `engine/schemas.py`

```python
class ModuleMetadata(BaseModel):
    # ... existing fields ...
    dependencies: list[str] = Field(default_factory=list)  # NEW
    metadata: dict = Field(default_factory=dict)           # NEW
```

**Purpose:** Support dependency graph and CP-specific metadata (rating_bracket, priority, estimated_hours).

---

## Implementation Phases

### ✅ Phase 0: Foundation (COMPLETE)

**Deliverables:**
- [x] Extended `ModuleMetadata` schema with dependencies and metadata dict
- [x] Created `curricula/cp_accelerator/` directory structure
- [x] Authored initial `manifest.json` with 5 sample modules
- [x] Built `scripts/ingest_cp_content.py` automation pipeline
- [x] Documentation (this file)

**Validation:**
```bash
# Verify schema changes don't break existing curriculum
uv run python -m pytest tests/ -k test_curriculum

# Check manifest validates
uv run python -c "from engine.schemas import CurriculumManifest; import json; CurriculumManifest(**json.load(open('curricula/cp_accelerator/manifest.json')))"
```

### 🔄 Phase 1: Automated Scaffolding (IN PROGRESS)

**Objective:** Generate base content for all 19 modules using `ingest_cp_content.py`.

**Prerequisites:**
1. Clone DSA Taxonomy repo:
   ```bash
   cd ~/repos
   git clone https://github.com/Yassir-aykhlf/DSA-Taxonomies.git
   ```

2. Map roadmap resources to patterns (see Appendix A)

**Execution:**

For each pattern in the taxonomy:

```bash
# Example: Two Pointers pattern
uv run python scripts/ingest_cp_content.py \
  --pattern two_pointers \
  --taxonomy-path ~/repos/DSA-Taxonomies \
  --rating-bracket "0-999" \
  --resources \
    "https://www.youtube.com/watch?v=..." \
    "https://leetcode.com/discuss/study-guide/..."
```

**Output per module:**
- `build_prompt.txt` - Auto-generated from taxonomy + canonical problem
- `justify_questions.json` - LLM-scaffolded conceptual questions
- `validator.sh` - Executable test runner
- `test_cases.json` - Template for public test cases

**Quality Checklist:**
- [ ] Build prompt clearly explains pattern with mermaid diagram
- [ ] Canonical problem is representative and well-chosen
- [ ] Validator script is executable and syntactically correct
- [ ] Test cases JSON has correct schema

**Estimated Time:** 2-3 hours for all 19 modules

---

### Phase 2: Expert Curation (NOT STARTED)

**Objective:** Human-in-the-loop refinement for pedagogical excellence.

**Tasks per module:**

1. **Fill Test Cases** (`test_cases.json`)
   - Extract examples from LeetCode problem statement
   - Add 2-3 edge cases (empty input, single element, max constraints)
   - Verify test cases are correct by running against reference solution

2. **Refine Justify Questions** (`justify_questions.json`)
   - Review LLM-generated questions for clarity and depth
   - Add failure modes with specific keywords
   - Write model answers that demonstrate mastery

3. **Author Reference Solution** (`.solutions/cp_accelerator/modules/{pattern}/solution.py`)
   - Write clean, idiomatic implementation
   - Add inline comments explaining key insights
   - Verify it passes all test cases

4. **Create Bug Patches** (`bugs/`)
   - Introduce 2-3 subtle, realistic bugs (off-by-one, edge case handling, wrong termination)
   - Generate `.patch` files: `diff -u reference.py buggy.py > bug1.patch`
   - Write `bug1_symptom.txt` describing the failure in CP terms

**Quality Bar:** Use `cs336_a1` modules as exemplars.

**Estimated Time:** 3-4 hours per module × 19 = 60-75 hours total

---

### Phase 3: CI/CD Quality Pipeline (NOT STARTED)

**Objective:** Automated validation to prevent regressions.

**Implementation:** `.github/workflows/validate_cp_curriculum.yml`

```yaml
name: CP Curriculum Quality Gate

on:
  pull_request:
    paths:
      - 'curricula/cp_accelerator/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Schema Validation
        run: |
          # Validate manifest.json
          uv run python -c "from engine.schemas import CurriculumManifest; ..."
          
      - name: Golden Path Tests
        run: |
          # For each module, run validator.sh against reference solution
          # MUST PASS
          
      - name: Silent Bug Tests
        run: |
          # For each bug patch, apply to reference and run validator
          # MUST FAIL
```

**Success Criteria:**
- All reference solutions pass local validators
- All bug patches cause validator failures
- Manifest schema validates
- No broken external links

**Estimated Time:** 8 hours (includes testing and debugging)

---

## Usage Examples

### For Content Authors

**Generate a new module:**
```bash
uv run python scripts/ingest_cp_content.py \
  --pattern dynamic_programming \
  --taxonomy-path ~/repos/DSA-Taxonomies \
  --rating-bracket "1200-1399" \
  --resources \
    "https://www.youtube.com/watch?v=oBt53YbR9Kk" \
    "https://usaco.guide/gold/intro-dp"
```

**Curate the module:**
```bash
# 1. Fill test cases
vim curricula/cp_accelerator/modules/dynamic_programming/test_cases.json

# 2. Write reference solution
vim .solutions/cp_accelerator/modules/dynamic_programming/solution.py

# 3. Test it
cd curricula/cp_accelerator/modules/dynamic_programming
./validator.sh

# 4. Create bug
cp .solutions/.../solution.py buggy.py
# ... edit buggy.py to introduce off-by-one error ...
diff -u .solutions/.../solution.py buggy.py > bugs/off_by_one.patch
echo "Wrong Answer on Test 3: Expected 5, got 4" > bugs/off_by_one_symptom.txt
```

### For Students

**Initialize the curriculum:**
```bash
uv run python -m engine.main init cp_accelerator
```

**Complete a module (Build stage):**
```bash
uv run python -m engine.main show    # See the challenge
vim cs336_basics/solution.py         # Implement it
uv run python -m engine.main submit  # Validate
```

**Progress through BJH loop:**
- Build → Justify (answer conceptual questions)
- Justify → Harden (debug injected bug)
- Harden → Next module (automatic)

---

## Appendix A: Pattern-to-Roadmap Mapping

| Pattern ID | Taxonomy File | Rating Bracket | Priority | Resources |
|:---|:---|:---|:---|:---|
| `sorting` | `5. Sorting.md` | 0-999 | Vital | [Language docs] |
| `two_pointers` | `1. Two Pointers.md` | 0-999 | Helpful | [SecondThread tutorial] |
| `hash_table` | `2. Hash Table.md` | 0-999 | Vital | [CS Dojo video] |
| `binary_search` | `7. Binary Search.md` | 1000-1199 | Vital | [Errichto video], [Topcoder tutorial] |
| `dynamic_programming` | `12. Dynamic Programming.md` | 1200-1399 | Vital | [MIT OCW], [USACO Guide] |
| `graphs_trees` | `6. Traversal Algorithms.md` | 1400-1599 | Vital | [William Fiset playlist] |
| `greedy` | `10. Greedy Algorithms.md` | 1600-1899 | Vital | [Algorithms Live] |
| ... | ... | ... | ... | ... |

*Note: Resource URLs to be filled during Phase 1 execution.*

---

## Success Metrics

### Module Quality (Per Module)
- [ ] Build prompt > 500 words with clear explanation
- [ ] ≥3 test cases with edge coverage
- [ ] ≥2 justify questions with model answers
- [ ] ≥2 realistic bug patches
- [ ] Reference solution < 100 lines
- [ ] All tests pass in CI

### Curriculum Completeness
- [ ] All 19 patterns from taxonomy implemented
- [ ] Dependency graph is acyclic (topological ordering valid)
- [ ] Rating progression: 0-999 → 1000-1199 → ... → 1600-1899
- [ ] Vital topics prioritized before Helpful topics in each bracket

### Student Experience
- [ ] Can complete first module in < 2 hours
- [ ] Justify feedback is actionable (not generic)
- [ ] Bugs are subtle enough to require debugging (not obvious typos)
- [ ] External resources are high-quality (> 90% upvote ratio)

---

## Risk Mitigation

### Risk: Scraped test cases insufficient
**Mitigation:** 
- Explicitly state local validation is not exhaustive
- Prompt students to submit to online judge for full validation
- Focus on pattern understanding over exact correctness

### Risk: LLM-generated questions are shallow
**Mitigation:**
- Phase 2 human curation is mandatory, not optional
- Use `cs336_a1` questions as quality benchmarks
- Iterate until Socratic depth matches exemplars

### Risk: 75+ hour curation time is prohibitive
**Mitigation:**
- Prioritize Vital topics in 0-1599 range first (12 modules)
- Community contributions for Helpful/Advanced topics
- Incremental rollout: MVP with 5 modules, then expand

---

## Next Actions

**Immediate (Today):**
1. Clone DSA-Taxonomies repo
2. Run ingestion script for `two_pointers` as proof-of-concept
3. Manually curate `two_pointers` module to completion
4. Test full BJH loop with `engine init cp_accelerator`

**This Week:**
1. Complete Phase 1 for all Foundation tier patterns (5 modules)
2. Begin Phase 2 curation for Foundation tier
3. Draft CI/CD workflow (Phase 3)

**This Month:**
1. Complete all 19 modules through Phase 2
2. Deploy CI/CD pipeline
3. Recruit 3 beta testers for feedback

---

## References

- [DSA Taxonomy GitHub](https://github.com/Yassir-aykhlf/DSA-Taxonomies)
- [CP Roadmap Google Doc](https://docs.google.com/document/d/1-7Co93b504uyXyMjjE8bnLJP3d3QXvp_m1UjvbvdR2Y)
- Mastery Engine: `docs/architecture/MASTERY_ENGINE.md`
- CS336 Exemplar Curriculum: `curricula/cs336_a1/`

---

**Status:** Foundation complete. Ready for Phase 1 execution.  
**Last Updated:** November 16, 2025
