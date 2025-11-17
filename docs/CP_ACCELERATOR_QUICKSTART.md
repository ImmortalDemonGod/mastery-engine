# CP Accelerator - Quick Start Guide

**Status:** Phase 0 Complete | Ready for Content Generation  
**Last Updated:** November 16, 2025

## What is CP Accelerator?

A systematic, interactive curriculum that guides you from **competitive programming newbie (rating 0)** to **candidate master (1900+)** through 19 core algorithmic patterns.

**Key Features:**
- ✅ Rating-driven progression based on proven roadmap
- ✅ 1000+ LeetCode problems organized by pattern
- ✅ Build-Justify-Harden pedagogical loop
- ✅ Automated content generation from DSA Taxonomy
- ✅ Dependency graph ensures proper learning order

---

## Foundation Complete (Phase 0)

### What's Been Built

#### 1. **Extended Schema** (`engine/schemas.py`)
```python
class ModuleMetadata(BaseModel):
    # ... existing fields ...
    dependencies: list[str] = []     # NEW: Prerequisite modules
    metadata: dict = {}              # NEW: Rating, priority, hours
```

**Purpose:** Support dependency graphs and CP-specific metadata.

#### 2. **Curriculum Structure**
```
curricula/cp_accelerator/
├── manifest.json              # 5 sample modules with metadata
└── modules/                   # Empty, ready for Phase 1
```

**Sample Modules:**
- `sorting` (0-999, Vital, 4 hours)
- `two_pointers` (0-999, Helpful, 6 hours) → depends on sorting
- `hash_table` (0-999, Vital, 5 hours)
- `binary_search` (1000-1199, Vital, 8 hours) → depends on sorting
- `dynamic_programming` (1200-1399, Vital, 20 hours) → depends on binary_search

#### 3. **Content Pipeline** (`scripts/ingest_cp_content.py`)

Automated tool that:
1. Parses DSA Taxonomy markdown files
2. Extracts pattern explanations + mermaid diagrams
3. Selects canonical LeetCode problems
4. Generates `build_prompt.txt` with embedded problem
5. Scaffolds `justify_questions.json` (LLM-assisted)
6. Creates `validator.sh` + `test_cases.json` templates

**Usage:**
```bash
uv run python scripts/ingest_cp_content.py \
  --pattern two_pointers \
  --taxonomy-path ~/repos/DSA-Taxonomies \
  --rating-bracket "0-999" \
  --resources "https://youtube.com/..." "https://usaco.guide/..."
```

#### 4. **Documentation**
- `docs/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md` - Full technical blueprint
- `docs/CP_ACCELERATOR_QUICKSTART.md` - This file

---

## Next Steps (Phase 1)

### Prerequisites

1. **Clone DSA Taxonomy Repo:**
   ```bash
   cd ~/repos
   git clone https://github.com/Yassir-aykhlf/DSA-Taxonomies.git
   ```

2. **Verify Foundation:**
   ```bash
   cd /Volumes/Totallynotaharddrive/assignment1-basics
   
   # Check schema validates
   uv run python -c "
   from engine.schemas import CurriculumManifest
   import json
   manifest = json.load(open('curricula/cp_accelerator/manifest.json'))
   CurriculumManifest(**manifest)
   print('✓ Manifest validates')
   "
   ```

### Execute Phase 1: Automated Scaffolding

**Goal:** Generate base content for all 19 patterns.

**Time Estimate:** 2-3 hours

**Process:**

For each pattern in `Appendix A` of the Implementation Guide:

```bash
# Example: Two Pointers
uv run python scripts/ingest_cp_content.py \
  --pattern two_pointers \
  --taxonomy-path ~/repos/DSA-Taxonomies \
  --rating-bracket "0-999" \
  --resources \
    "https://www.youtube.com/watch?v=..." \
    "https://codeforces.com/blog/entry/..."

# Verify output
ls -la curricula/cp_accelerator/modules/two_pointers/
# Should see: build_prompt.txt, justify_questions.json, 
#             validator.sh, test_cases.json, bugs/
```

**Quality Checklist per Module:**
- [ ] `build_prompt.txt` > 500 words with mermaid diagram
- [ ] Canonical problem clearly stated with link
- [ ] `validator.sh` is executable (`chmod +x`)
- [ ] `test_cases.json` has correct schema
- [ ] `bugs/` directory exists

**Batch Script (Optional):**
```bash
#!/bin/bash
# scripts/ingest_all_patterns.sh

TAXONOMY=~/repos/DSA-Taxonomies

patterns=(
  "two_pointers:0-999:https://..."
  "binary_search:1000-1199:https://..."
  # ... add all 19
)

for entry in "${patterns[@]}"; do
  IFS=':' read -r pattern bracket resource <<< "$entry"
  uv run python scripts/ingest_cp_content.py \
    --pattern "$pattern" \
    --taxonomy-path "$TAXONOMY" \
    --rating-bracket "$bracket" \
    --resources "$resource"
done
```

---

## After Phase 1: Manual Curation (Phase 2)

Once all 19 modules are scaffolded:

### For Each Module

1. **Fill Test Cases** (`test_cases.json`)
   - Extract from LeetCode examples
   - Add 2-3 edge cases
   - Verify against reference solution

2. **Refine Justify Questions** (`justify_questions.json`)
   - Review LLM generations
   - Add failure modes
   - Write model answers

3. **Create Reference Solution** (`.solutions/cp_accelerator/modules/{pattern}/solution.py`)
   - Clean, idiomatic code
   - Inline comments
   - Verify passes all tests

4. **Author Bug Patches** (`bugs/`)
   - 2-3 subtle bugs per module
   - Generate `.patch` files
   - Write symptom descriptions

**Time Estimate:** 3-4 hours per module × 19 = 60-75 hours

**Prioritization Strategy:**
- Start with **Vital topics in 0-1599 range** (12 modules)
- Community contributions for Helpful/Advanced topics
- Incremental rollout: MVP with 5 modules → expand

---

## Testing the System

### After First Module Complete

```bash
# Initialize curriculum
uv run python -m engine.main init cp_accelerator

# Check status
uv run python -m engine.main status
# Should show: Module 1/5, Stage: BUILD

# View challenge
uv run python -m engine.main show

# Implement solution
vim curricula/cp_accelerator/modules/sorting/solution.py

# Submit
uv run python -m engine.main submit
```

### Full BJH Loop Test

1. **Build:** Implement sorting algorithm → submit
2. **Justify:** Answer "Why is O(n log n) optimal for comparison sorts?"
3. **Harden:** Debug injected bug (e.g., off-by-one in merge)

---

## Monitoring Progress

### Track Implementation Status

Create `docs/CP_MODULE_STATUS.md`:

| Module | Phase 1 | Phase 2 | CI Tests | Status |
|:---|:---:|:---:|:---:|:---|
| sorting | ✅ | ⏳ | ❌ | In curation |
| two_pointers | ✅ | ❌ | ❌ | Scaffolded |
| hash_table | ⏳ | ❌ | ❌ | In progress |
| ... | ... | ... | ... | ... |

### Quality Metrics

**Per Module:**
- Build prompt word count
- Number of test cases
- Justify questions depth
- Bug patch realism

**Overall:**
- Modules complete / 19
- Dependency graph validated
- CI pipeline green
- Beta tester feedback

---

## Resources

### Implementation References
- [DSA Taxonomy GitHub](https://github.com/Yassir-aykhlf/DSA-Taxonomies)
- [CP Roadmap Doc](https://docs.google.com/document/d/1-7Co93b504uyXyMjjE8bnLJP3d3QXvp_m1UjvbvdR2Y)
- `docs/CP_ACCELERATOR_IMPLEMENTATION_GUIDE.md` - Technical blueprint

### Mastery Engine Docs
- `docs/architecture/MASTERY_ENGINE.md`
- `curricula/cs336_a1/` - Exemplar curriculum

### Example Patterns
- Two Pointers: `Taxonomies/1. Two Pointers.md`
- Dynamic Programming: `Taxonomies/12. Dynamic Programming.md`
- Binary Search: `Taxonomies/7. Binary Search.md`

---

## Troubleshooting

### Schema Validation Fails
```bash
# Check for typos in manifest.json
cat curricula/cp_accelerator/manifest.json | jq .

# Validate against schema
uv run python -c "
from engine.schemas import CurriculumManifest
import json
manifest = json.load(open('curricula/cp_accelerator/manifest.json'))
try:
    CurriculumManifest(**manifest)
    print('✓ Valid')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

### Ingestion Script Errors
```bash
# Check taxonomy file exists
ls ~/repos/DSA-Taxonomies/Taxonomies/

# Run with debug output
uv run python scripts/ingest_cp_content.py \
  --pattern two_pointers \
  --taxonomy-path ~/repos/DSA-Taxonomies \
  2>&1 | tee ingest.log
```

### Module Not Loading in Engine
```bash
# Check directory structure
tree -L 2 curricula/cp_accelerator/modules/

# Verify build_prompt.txt exists
ls curricula/cp_accelerator/modules/*/build_prompt.txt

# Re-init engine
rm -rf .mastery_engine_worktree .mastery_progress.json
uv run python -m engine.main init cp_accelerator
```

---

## Success Criteria (MVP)

**Phase 0:** ✅ COMPLETE
- Schema extended
- Manifest created
- Pipeline built
- Documentation written

**Phase 1:** 🎯 TARGET
- [ ] All 19 modules scaffolded
- [ ] Each has build_prompt.txt, justify_questions.json, validator.sh
- [ ] Manifest updated with all modules
- [ ] Dependency graph validated

**Phase 2:** 🎯 TARGET (Vital modules only)
- [ ] 12 Vital modules (0-1599 range) fully curated
- [ ] Reference solutions written
- [ ] Test cases filled
- [ ] Bug patches created

**Phase 3:** 🎯 TARGET
- [ ] CI/CD pipeline deployed
- [ ] Golden path tests passing
- [ ] Silent bug tests failing correctly
- [ ] Ready for beta testing

---

## Contact & Contributions

This is a systematic, open implementation. Contributions welcome:

1. **Module Curation:** Help fill test cases and create bug patches
2. **Resource Links:** Update roadmap resource mappings
3. **Quality Review:** Test modules through full BJH loop
4. **Bug Reports:** File issues in GitHub

**Current Status:** Phase 0 complete, Phase 1 ready to execute.

---

**Last Updated:** November 16, 2025  
**Maintainer:** Mastery Engine Development Team
