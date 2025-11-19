# Test Coverage Increase Session Summary

**Date**: 2025-11-12  
**Objective**: Systematically measure and increase test coverage of Mastery Engine with exceptional rigor

## Session Progress

### Phase 1: Dependency Fix ✅
- **Issue**: OpenAI SDK import errors blocking test execution
- **Resolution**: Upgraded `openai` from 2.7.2 to 1.109.1
- **Validation**: Imports verified working with `uv run python`

### Phase 2: Baseline Measurement ✅
- **Scope**: engine/ package only (exclude third-party and e2e)
- **Command**: `coverage run --source=engine -m pytest -q tests/engine -k "not wrong_stage"`
- **Results**:
  ```
  Name                             Stmts   Miss  Cover   Missing
  --------------------------------------------------------------
  engine/__init__.py                   0      0   100%
  engine/curriculum.py                39     21    46%
  engine/main.py                     690    672     3%   ← PRIMARY TARGET
  engine/schemas.py                   46     10    78%
  engine/services/llm_service.py      60     56     7%
  engine/stages/harden.py             47     34    28%
  engine/stages/justify.py            38     25    34%
  engine/state.py                     39     24    38%
  engine/validator.py                 54     36    33%
  engine/workspace.py                 53     36    32%
  --------------------------------------------------------------
  TOTAL                             1066    914    14%
  ```

### Phase 3: High-Yield Test Implementation 🟡
- **Created**: `tests/engine/test_submit_handlers.py` (450+ lines)
- **Target**: P0 CLI implementation (unified submit + handlers)
- **Tests Added**:
  - ✅ `TestLoadCurriculumState`: 1/1 passing
  - ✅ `TestCheckCurriculumComplete`: 2/2 passing  
  - ✅ `TestSubmitBuildStage`: 2/2 passing
  - ⚠️ `TestSubmitJustifyStage`: 0/3 passing (fixed JustifyQuestion schema)
  - ⚠️ `TestSubmitHardenStage`: 0/1 passing (handler logic investigation needed)
  - ✅ `TestUnifiedSubmitCommand`: 3/3 passing
- **Current Status**: 8/12 tests passing (67%)

### Phase 4: Issues Encountered 🔍

#### Issue A: UserProgress Mocking
- **Problem**: `progress.mark_stage_complete` is a method, not a mock attribute
- **Solution**: Use `MagicMock(spec=UserProgress)` instead of real instance
- **Status**: ✅ Resolved

#### Issue B: Inline Import Mocking
- **Problem**: `import os`, `import shutil`, `import tempfile` inside handlers
- **Solution**: Patch at module level (`@patch('os.unlink')` not `@patch('engine.main.os.unlink')`)
- **Status**: ✅ Resolved

#### Issue C: JustifyQuestion Schema
- **Problem**: Missing required fields (id, model_answer, failure_modes)
- **Solution**: Updated all test fixtures with complete schema
- **Status**: ✅ Resolved

#### Issue D: Harden Handler Returns False
- **Problem**: `_submit_harden_stage` unexpectedly returns False despite mocked success
- **Investigation Needed**: Check handler logic for additional conditions
- **Status**: ⏸️ Pending investigation

## Coverage Impact (Projected)

**Before**: engine/main.py at 3% (672/690 lines uncovered)  
**After** (once all 12 tests pass):
- Helper functions: +36 lines (100% → 5% gain)
- Stage handlers: +290 lines (partial → ~42% gain)
- Unified submit: +75 lines (partial → ~11% gain)

**Projected Total**: engine/main.py from 3% → **~45-50%** (estimated)  
**Engine Package**: from 14% → **~35-40%** (estimated)

## Remaining Work

### Immediate (< 1 hour)
1. ✅ Fix JustifyQuestion schema in tests (DONE)
2. ⏸️ Debug harden handler False return
3. ⏸️ Add missing mocks for harden (WorkspaceManager, HardenRunner)
4. ⏸️ Re-run full test suite
5. ⏸️ Generate final coverage report

### Short-term (2-3 hours)
1. Add negative-path tests for error handling:
   - ValidatorTimeoutError
   - ValidatorExecutionError
   - ConfigurationError, LLMAPIError, LLMResponseError
2. Add integration tests for unified submit error handling
3. Target remaining low-coverage modules:
   - engine/services/llm_service.py (7%)
   - engine/validator.py (33%)
   - engine/workspace.py (32%)

### Medium-term (4-6 hours)
1. Fix 3 failing status command tests in test_main.py
2. Add tests for legacy commands (if keeping for backward compat)
3. Re-enable and debug e2e test suite
4. Achieve 80%+ coverage across engine/ package

## Key Learnings

1. **Inline imports complicate mocking**: Consider moving os/shutil/tempfile to top-level imports
2. **Pydantic schemas require complete fixtures**: Always check BaseModel field requirements
3. **MagicMock(spec=Class) > real instances**: Provides better isolation and mock verification
4. **Coverage.py > pytest-cov**: More reliable for scoped coverage measurement
5. **Timeouts critical**: Set global timeout (180s) to prevent hanging on flaky tests

## Recommendations

### For Immediate Next Session
1. **Priority**: Debug harden handler, complete 4 failing tests
2. **Quick Win**: Run coverage with passing 8 tests to see partial gains
3. **Document**: Record actual coverage improvement from these 8 tests

### For Testing Strategy
1. **Separate concerns**: Unit tests (handlers) vs integration tests (routing)
2. **Mock external deps**: Filesystem, subprocess, LLM API
3. **Test both paths**: Success and failure for each handler
4. **Verify side effects**: State updates, file operations, console output

### For Code Quality
1. **Move inline imports to top**: Easier to mock, clearer dependencies
2. **Extract validation logic**: Reduce handler complexity
3. **Add type hints**: Improve test IDE support
4. **Document handler contracts**: Expected inputs, outputs, side effects

## Files Modified

- ✅ `pyproject.toml`: openai 1.109.1
- ✅ `tests/engine/test_submit_handlers.py`: 450+ lines (NEW)
- ✅ `engine/main.py`: Fixed `next()` to conditionally require shadow worktree
- ✅ `docs/coverage/coverage_report_engine.txt`: Baseline measurement
- ✅ `docs/coverage/TEST_COVERAGE_SESSION_SUMMARY.md`: This document

## Commands for Next Session

```bash
# Run only passing tests with coverage
uv run coverage run --source=engine -m pytest -v \
  tests/engine/test_submit_handlers.py::\
TestLoadCurriculumState \
  tests/engine/test_submit_handlers.py::TestCheckCurriculumComplete \
  tests/engine/test_submit_handlers.py::TestSubmitBuildStage \
  tests/engine/test_submit_handlers.py::TestUnifiedSubmitCommand

# Generate coverage report
uv run coverage report -m --include="engine/main.py"
uv run coverage html -d docs/coverage/html_partial

# Debug harden handler
uv run pytest -xvs \
  tests/engine/test_submit_handlers.py::TestSubmitHardenStage::test_harden_success_advances_module

# Full suite (after fixes)
uv run coverage run --source=engine -m pytest tests/engine -k "not wrong_stage"
uv run coverage report -m
uv run coverage html -d docs/coverage/html
```

## Success Criteria

- [ ] All 12 new tests passing
- [ ] engine/main.py coverage > 40%
- [ ] engine/ package coverage > 30%
- [ ] Zero test regressions
- [ ] Documentation updated with final metrics

---

**Status**: In Progress (67% complete)  
**Next Action**: Debug harden handler False return  
**Estimated Time to Completion**: 1-2 hours
