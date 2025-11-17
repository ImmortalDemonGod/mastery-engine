# Batch Migration Execution Guide

**Status:** ✅ APPROVED - Ready for execution  
**Estimated Duration:** 2-3 hours  
**Expected Success Rate:** 85-90% (15-17 bugs)

---

## Pre-Flight Checklist

### 1. Environment Setup

```bash
# Verify OpenAI API key is configured
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI_API_KEY
```

**Required:** OpenAI API key with sufficient credits (~$5-10 estimated)

### 2. Verify System Status

```bash
# Check golden dataset exists
ls -la curricula/cs336_a1/modules/*/bugs/*.json

# Expected: 3 files
# - softmax/bugs/no_subtract_max.json
# - silu/bugs/missing_multiply.json  
# - rmsnorm/bugs/missing_keepdim.json

# Count remaining patches
find curricula/cs336_a1/modules -name "*.patch" -type f | wc -l
# Expected: 21 total patches
```

### 3. Test Single Bug (Dry Run)

Before running the full batch, test with a single bug to verify the system works:

```bash
# Pick a simple module for testing (e.g., adamw)
engine create-bug adamw \
  --patch curricula/cs336_a1/modules/adamw/bugs/[patch_name].patch \
  --symptom curricula/cs336_a1/modules/adamw/symptom.txt

# Verify output
cat curricula/cs336_a1/modules/adamw/bugs/[output].json
```

**Success Criteria:**
- ✅ JSON file created
- ✅ Valid JSON syntax
- ✅ Passes schema validation
- ✅ Pattern looks semantically correct

If dry run fails:
- Check API key configuration
- Verify patch file format
- Review error messages
- Adjust prompt if needed

---

## Batch Migration Execution

### Step 1: Run Migration Script

```bash
# Execute batch migration
python scripts/migrate_bugs_llm.py

# Redirect output to log file (optional)
python scripts/migrate_bugs_llm.py 2>&1 | tee migration_log.txt
```

### Step 2: Monitor Progress

The script will display real-time progress:

```
============================================================
BATCH BUG MIGRATION: .patch → .json
============================================================

📁 Scanning curricula/cs336_a1...
✨ Found 18 patch files to migrate

🤖 Initializing LLM Bug Author...
   Loaded 3 golden examples

============================================================
[1/18] attention: missing_mask.patch
============================================================
📝 Loaded symptom from symptom.txt
🤖 LLM Attempt 1/3...
  ✅ Validation passed!

✅ Success! Wrote .../attention/bugs/missing_mask.json
```

### Step 3: Review Summary

At completion, the script shows:

```
============================================================
MIGRATION SUMMARY
============================================================
✅ Successful: 15
❌ Failed: 3
Total migrated: 15/18
```

---

## Human Review Process

### Review Checklist

For each generated JSON file:

1. **Syntax Check** ✅ (automated)
   ```bash
   python -c "import json; json.load(open('file.json'))"
   ```

2. **Schema Validation** ✅ (automated)
   - Verify all required fields present
   - Check node_type values are valid
   - Ensure replacement strategies are correct

3. **Semantic Review** 👁️ (manual)
   - Pattern matches the intended AST structure
   - Transformation logic is correct
   - Description accurately reflects the bug

4. **Injection Test** 🧪 (semi-automated)
   ```bash
   # Test with reference implementation
   cd curricula/cs336_a1/modules/[module]
   # Run generic injector test
   ```

5. **E2E Validation** 🔍 (select bugs)
   ```bash
   # For critical bugs, run full E2E test
   engine init cs336_a1
   # Implement module
   # Run through harden stage
   ```

### Review Template

```markdown
## [Module Name] - [Bug ID]

**Generated JSON:** ✅/❌  
**Schema Valid:** ✅/❌  
**Pattern Correct:** ✅/❌/🤔  
**Transformation Correct:** ✅/❌/🤔  
**Injection Test:** ✅/❌  

**Notes:**
[Any issues or observations]

**Status:** APPROVED / NEEDS REVISION / MANUAL AUTHORING
```

---

## Handling Failures

### LLM Generation Failures

If a bug fails after 3 attempts:

**Option 1: Manual JSON Authoring**
```bash
# Use golden examples as reference
cp curricula/cs336_a1/modules/silu/bugs/missing_multiply.json \
   curricula/cs336_a1/modules/[module]/bugs/[bug].json

# Edit manually to match the patch
vim curricula/cs336_a1/modules/[module]/bugs/[bug].json
```

**Option 2: Prompt Adjustment**
```python
# Modify system or user prompt in bug_author.py
# Add specific guidance for this bug type
# Retry single bug generation
```

**Option 3: Simplify Bug**
```python
# Break complex bug into multiple simpler bugs
# Or use legacy .patch approach as fallback
```

### Common Failure Patterns

| Pattern | Cause | Fix |
|---------|-------|-----|
| Invalid JSON | Hallucination | Check temperature setting |
| Missing fields | Incomplete schema | Add explicit field requirements |
| Wrong pattern | AST misunderstanding | Add similar example to golden dataset |
| Failed injection | Logic error | Manual review and correction |

---

## Post-Migration Tasks

### 1. Commit Generated Files

```bash
# Stage all new JSON files
git add curricula/cs336_a1/modules/*/bugs/*.json

# Commit with descriptive message
git commit -m "feat: LLM-generated bug definitions for 15 modules

Generated using Phase 4 LLM bug authoring tool.

Success rate: 15/18 (83%)
Manual review: COMPLETE
Quality: All bugs validated with GenericBugInjector

Modules migrated:
- attention
- cross_entropy
- [... list all ...]

Remaining (manual authoring):
- [... list failures ...]
"
```

### 2. Update Documentation

```bash
# Update curriculum README
vim curricula/cs336_a1/README.md

# Mark migration complete
echo "Bug Migration Status: 18/21 complete (86%)" >> STATUS.md
```

### 3. Cleanup

```bash
# Remove v2 files if any
find . -name "*_v2.json" -delete

# Archive migration log
mv migration_log.txt docs/logs/migration_$(date +%Y%m%d).txt
```

---

## Troubleshooting

### API Issues

**Problem:** `openai.AuthenticationError`  
**Solution:** 
```bash
export OPENAI_API_KEY='your-key-here'
# Or add to .env file
```

**Problem:** `openai.RateLimitError`  
**Solution:**
- Wait and retry
- Reduce parallelism
- Upgrade API tier

**Problem:** `openai.APIConnectionError`  
**Solution:**
- Check internet connection
- Verify API endpoint
- Try again later

### Script Issues

**Problem:** "Golden dataset not found"  
**Solution:**
```bash
# Verify golden files exist
ls curricula/cs336_a1/modules/*/bugs/*.json
```

**Problem:** "Patch file not found"  
**Solution:**
```bash
# Check patch files exist
find curricula/cs336_a1/modules -name "*.patch"
```

### Quality Issues

**Problem:** Generated JSON doesn't match expected bug  
**Solution:**
1. Review LLM output for misunderstanding
2. Check patch file is correct format
3. Add more context to prompt
4. Manually author JSON

---

## Success Metrics

### Target Metrics

- **Success Rate:** >80% (15+ bugs)
- **Time:** <3 hours total
- **Quality:** >90% pass human review

### Actual Results

**To be filled after migration:**

```
Total Patches: 18
Successful: __
Failed: __
Success Rate: __%

Time Taken: __ hours
Manual Review Time: __ minutes

Quality Issues: __
Bugs Requiring Manual Authoring: __
```

---

## Contact & Support

**Issues:** Review `docs/PHASE4_LLM_TOOL.md`  
**Questions:** Check golden examples in `curricula/cs336_a1/modules/{softmax,silu,rmsnorm}/bugs/`  
**Emergency:** Fallback to manual JSON authoring

---

## Next Steps After Migration

1. **Commit all reviewed files**
2. **Update project documentation**
3. **Run full E2E tests on select modules**
4. **Consider Phase 5 enhancements:**
   - Expand to `justify_questions.json`
   - Scaffold `build_prompt.txt`
   - Visual bug designer UI

---

**Guide Version:** 1.0  
**Last Updated:** November 13, 2025  
**Status:** Ready for Execution
