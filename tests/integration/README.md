# Integration Tests

This directory contains integration tests that make **real API calls** to external services.

## LLM Integration Tests

**File**: `test_llm_integration.py`  
**Service**: OpenAI API (gpt-4o-mini)  
**Cost**: ~$0.006 per full test run

### What These Tests Validate

1. **Fast Filter Logic**: Shallow answers caught without API calls
2. **LLM Acceptance**: Deep, correct answers properly accepted
3. **LLM Rejection**: Incomplete/incorrect answers rejected with Socratic feedback
4. **Error Handling**: Missing API key produces clear guidance
5. **Decision Boundary**: Fast filter vs. LLM routing works correctly
6. **Timeout Handling**: API timeout doesn't crash the system

### Running Integration Tests

#### Prerequisites

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Or use .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

#### Run Integration Tests Only

```bash
# Run all integration tests
uv run pytest tests/integration -m integration -v

# Run specific test
uv run pytest tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer -v
```

#### Run All Tests EXCEPT Integration

```bash
# Skip expensive integration tests in regular test runs
uv run pytest -m "not integration"
```

#### Run Everything (Unit + Integration + E2E)

```bash
# Run all tests including integration
uv run pytest
```

### Cost Management

| Test | API Calls | Cost |
|------|-----------|------|
| `test_fast_filter_blocks_shallow_answer` | 0 (mocked) | $0 |
| `test_llm_accepts_deep_correct_answer` | 1 | $0.003 |
| `test_llm_rejects_conceptual_error_with_socratic_feedback` | 1 | $0.003 |
| `test_error_handling_missing_api_key` | 0 | $0 |
| `test_fast_filter_vs_llm_decision_boundary` | 0 (mocked) | $0 |
| `test_llm_api_timeout_handling` | 0 (timeout) | $0 |
| **TOTAL PER RUN** | **2** | **$0.006** |

### When to Run These Tests

**Always Run (Free)**:
- `test_fast_filter_blocks_shallow_answer`
- `test_error_handling_missing_api_key`
- `test_fast_filter_vs_llm_decision_boundary`

**Run Before Commit (Costs ~$0.006)**:
- Full integration test suite
- Catches LLM prompt regressions
- Validates API integration

**Run in CI (Optional)**:
- Set up OPENAI_API_KEY as secret
- Run integration tests on main branch only
- Budget: ~$0.60/month for 100 runs

### Skipping When No API Key

Tests automatically skip if `OPENAI_API_KEY` is not set:

```python
@pytest.fixture(scope="module")
def check_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping live LLM integration tests")
```

### Test Output

**Success**:
```
tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer PASSED
```

**Skipped (no API key)**:
```
tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer SKIPPED
(OPENAI_API_KEY not set - skipping live LLM integration tests)
```

**Failure**:
```
tests/integration/test_llm_integration.py::test_llm_accepts_deep_correct_answer FAILED
AssertionError: Deep answer should be accepted. Feedback: [actual feedback]
```

### Best Practices

1. **Local Development**: Run integration tests before pushing changes to Justify stage
2. **CI/CD**: Run on main branch only to limit costs
3. **Pull Requests**: Run unit tests only (skip integration)
4. **Release**: Run full suite including integration
5. **Cost Monitoring**: Track API usage at https://platform.openai.com/usage

### Debugging Failed Tests

If LLM tests fail:

1. **Check API Key**: Ensure `OPENAI_API_KEY` is set correctly
2. **Check API Status**: Visit https://status.openai.com
3. **Check Model**: Verify gpt-4o-mini is available
4. **Review Prompt**: Check if justify question changed
5. **Review Model Answer**: Verify expected answer is still valid
6. **Check Feedback**: LLM may have improved and now catches edge cases

### Adding New Integration Tests

```python
import pytest

pytestmark = pytest.mark.integration  # Mark as integration test

def test_new_llm_feature(check_api_key, softmax_questions):
    """
    Test new LLM functionality.
    
    Cost: ~$0.003 (one API call)
    """
    # Your test here
    pass
```

### FAQ

**Q: Why not mock the LLM in all tests?**  
A: Integration tests catch real issues like prompt regressions, API changes, and model behavior changes.

**Q: Can I run these in CI for free?**  
A: No, they use real API calls. Budget ~$0.60/month for 100 CI runs.

**Q: What if I don't have an API key?**  
A: Tests automatically skip. Core functionality is covered by unit tests with mocked LLM.

**Q: How often should I run these?**  
A: Before commits that touch Justify stage, LLM service, or justify questions.

**Q: Can I use a different model?**  
A: Yes, set `OPENAI_MODEL=gpt-4o` or modify `LLMService(model="your-model")`.
