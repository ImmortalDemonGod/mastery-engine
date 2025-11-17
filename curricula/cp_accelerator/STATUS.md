# CP Accelerator - Current Status

**Last Updated**: 2025-11-17  
**Version**: 2.0.0  
**Status**: ✅ All 19 patterns parsed, resources partially extracted

## What's Complete ✅

### 1. Systematic Taxonomy Parsing
- **All 19 DSA Taxonomy patterns** parsed from actual markdown files
- Problem IDs extracted via regex from source
- Pattern descriptions pulled from actual taxonomy text
- Problem counts verified (29-95 problems per pattern)

### 2. Roadmap Integration  
- **RoadmapResources.md** added to repo (actual source file)
- All 5 rating brackets parsed: 0-999, 1000-1199, 1200-1399, 1400-1599, 1600-1899
- Priorities mapped (Vital/Helpful) per bracket
- Each topic mapped to correct rating bracket and priority

### 3. Architecture
- Canonical curriculum JSON (single source of truth)
- Manifest auto-generated with validation
- CI enforcement (topological sort, diff check)
- Dependency graph validated (no cycles)

## Current Curriculum

### All 19 Topics by Rating Bracket

**0-999 (Foundation - 3 topics)**:
- `sorting` (Vital) - 29 problems in taxonomy
- `two_pointers` (Helpful) - 50 problems in taxonomy
- `linked_list` (Helpful) - 25 problems in taxonomy

**1000-1199 (Core Algorithms - 4 topics)**:
- `hash_table` (Vital) - 31 problems in taxonomy
- `stack_queue` (Vital) - 36 problems in taxonomy
- `binary_search` (Helpful) - 26 problems in taxonomy
- `prefix_sum` (Helpful) - 40 problems in taxonomy

**1200-1399 (Intermediate - 2 topics)**:
- `traversal` (Vital) - 95 problems in taxonomy (DFS, BFS, trees, graphs)
- `dynamic_programming` (Vital) - 49 problems in taxonomy

**1400-1599 (Advanced - 6 topics)**:
- `heap` (Helpful) - 27 problems in taxonomy
- `greedy` (Vital) - 39 problems in taxonomy
- `backtracking` (Helpful) - 32 problems in taxonomy
- `divide_conquer` (Helpful) - 55 problems in taxonomy
- `union_find` (Helpful) - 61 problems in taxonomy
- `design` (Helpful) - 62 problems in taxonomy

**1600-1899 (Expert - 4 topics)**:
- `trie` (Helpful) - 55 problems in taxonomy
- `bit_manipulation` (Vital) - 54 problems in taxonomy
- `segment_tree` (Vital) - 61 problems in taxonomy
- `combinatorics` (Vital) - 63 problems in taxonomy

**Total**: 19 topics, 830+ problems available

## What's Verified ✅

| Data Field | Status | Source |
|------------|--------|--------|
| Topic IDs | ✅ Verified | Manually curated mapping |
| Topic names | ✅ Verified | Taxonomy markdown files |
| Descriptions | ✅ Verified | Extracted from markdown (line 1-3) |
| Problem IDs | ✅ Verified | Regex parse from taxonomy |
| Problem titles | ✅ Verified | Regex parse from taxonomy |
| Problem counts | ✅ Verified | Counted from taxonomy files |
| Rating brackets | ✅ Verified | RoadmapResources.md |
| Priorities | ✅ Verified | RoadmapResources.md (Vital/Helpful) |
| Dependencies | ✅ Verified | Manually curated + topological sort |

## Known Issue: Resource Extraction ⚠️

**Problem**: Roadmap YouTube/blog URLs not being extracted

**Current State**:
- Each topic has 1 resource (taxonomy link only)
- RoadmapResources.md contains 30+ curated YouTube videos and blogs
- Parser regex not capturing these URLs correctly

**Example from RoadmapResources.md**:
```markdown
* Binary search - [Binary Search tutorial (C++ and Python)](https://www.youtube.com/watch?v=GU7DpgHINWQ)
```

**Expected**: Extract URL → Add to `binary_search` topic resources  
**Actual**: Resources list empty (except taxonomy link)

**Root Cause**: `_extract_topics_with_resources()` regex pattern not matching markdown format

## Next Steps

### Immediate (Fix Resource Extraction)
1. [ ] Debug regex pattern in `_extract_topics_with_resources()`
2. [ ] Test with sample lines from RoadmapResources.md
3. [ ] Re-run parser to extract all YouTube/blog URLs
4. [ ] Verify: each topic should have 1-5 resources

### Short-term (Content Generation)
5. [ ] Implement `ingest_cp_content.py` (generate build_prompt.txt)
6. [ ] Scaffold all 19 modules with prompts and test cases
7. [ ] Create validators for each module

### Medium-term (Expert Curation)
8. [ ] Refine justify questions for each module
9. [ ] Author 2-3 bugs per module (38-57 total bugs)
10. [ ] Test all bugs inject correctly

## File Locations

**Source Files**:
- `.sources/cp_accelerator/dsa_taxonomies/` - Cloned taxonomy repo (19 markdown files)
- `RoadmapResources.md` - Roadmap with rating brackets and resources

**Generated Files**:
- `curricula/cp_accelerator/canonical_curriculum.json` - Source of truth (19 topics)
- `curricula/cp_accelerator/manifest.json` - Auto-generated from canonical

**Scripts**:
- `scripts/parse_sources.py` - Parse taxonomy + roadmap → canonical JSON
- `scripts/generate_manifest.py` - Canonical JSON → manifest.json (with validation)

## Quality Metrics

**Parsing Coverage**: 19/19 topics (100%)  
**Problem Coverage**: 830+ problems mapped  
**Rating Coverage**: 0-1899 (5 brackets)  
**Resource Extraction**: 0% (needs fix)  
**Dependency Validation**: 100% (no cycles)  
**Source Verification**: 100% (all data traceable)

## Comparison: V1 vs V2

| Aspect | V1 (Placeholder) | V2 (Current) |
|--------|------------------|--------------|
| Topics | 7 (partial) | 19 (complete) |
| Source | Guessed | Parsed from files |
| Problems | Invented | Extracted via regex |
| Descriptions | Made up | Actual taxonomy text |
| Roadmap | Hardcoded dict | Parsed from markdown |
| Resources | Placeholder URLs | Taxonomy links (roadmap URLs pending fix) |
| Verifiable | No | Yes (line numbers) |

## Commands

**Regenerate from sources**:
```bash
uv run python scripts/parse_sources.py
uv run python scripts/generate_manifest.py
```

**Validate**:
```bash
uv run python scripts/generate_manifest.py --validate-only
```

**View stats**:
```bash
uv run python -c "
import json
with open('curricula/cp_accelerator/canonical_curriculum.json') as f:
    data = json.load(f)
print(f'Topics: {len(data[\"topics\"])}')
print(f'Version: {data[\"version\"]}')
"
```

---

**Status**: ✅ 19/19 topics parsed | ⚠️ Resource extraction needs fix | 🔄 Content generation pending
