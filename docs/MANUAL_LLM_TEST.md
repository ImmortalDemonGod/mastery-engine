# Manual LLM Integration Test Procedure

## Purpose
This document guides you through a **one-time manual test** of the live OpenAI API integration for the Justify stage. This test validates that:
1. The LLMService correctly communicates with OpenAI's API
2. Prompts are properly formatted and sent
3. Responses are correctly parsed and processed
4. The engine handles both acceptance and rejection correctly
5. Error handling works for missing/invalid API keys

**Note**: This test uses real API calls and will incur a small cost (~$0.01). It is **not automated** to avoid recurring API charges in CI.

---

## Prerequisites

### 1. Obtain OpenAI API Key
- Visit: https://platform.openai.com/api-keys
- Create a new API key
- Copy the key (starts with `sk-...`)

### 2. Configure Environment
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# Replace sk-your-api-key-here with your actual key
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Initialize Mastery Engine
```bash
# Clean any existing state
rm -rf ~/.mastery_progress.json .mastery_engine_worktree

# Initialize with cs336_a1 curriculum
uv run python -m engine.main init cs336_a1
```

---

## Test Procedure

### Test 1: Fast Filter (No LLM Call)

This validates that shallow answers are caught by the fast filter WITHOUT calling the LLM.

**Steps**:
```bash
# Set state to justify stage
cat > ~/.mastery_progress.json << 'EOF'
{
  "curriculum_id": "cs336_a1",
  "current_module_index": 0,
  "current_stage": "justify",
  "completed_modules": []
}
EOF

# Submit a shallow answer (should trigger fast filter)
uv run python -m engine.main submit-justification "It improves numerical stability"
```

**Expected Output**:
- ✅ Error message indicating shallow answer detected
- ✅ Feedback mentioning specific keywords (e.g., "stability")
- ✅ **NO API call made** (check logs - should not see OpenAI API calls)
- ✅ State remains at justify stage (not advanced)

**Record**:
- Fast filter keyword matched: ________________
- Feedback quality (1-5): ________________
- API call made? (should be NO): ________________

---

### Test 2: Deep Answer - LLM Acceptance

This validates the complete LLM evaluation flow with a correct answer.

**Steps**:
```bash
# Submit a comprehensive answer
uv run python -m engine.main submit-justification "The subtract-max trick exploits the mathematical identity: softmax(x) = softmax(x + c) for any constant c, because both numerator and denominator are scaled by exp(c) which cancels out. By choosing c = -max(x), we shift the input range to (-inf, 0], ensuring the largest exponent is exp(0) = 1.0. This prevents overflow (exp(large positive) → inf) while gracefully handling underflow (exp(large negative) → 0). Without this trick, inputs like x + 100 would cause exp() to overflow to infinity, producing NaN in the division."
```

**Expected Output**:
- ✅ API call made to OpenAI (check logs for request/response)
- ✅ LLM evaluation result: "accept" or "reject_with_feedback"
- ✅ If accepted: state advances to next stage
- ✅ If feedback: Socratic guidance provided

**Record**:
- API call successful? ________________
- LLM verdict (accept/reject): ________________
- Response time (seconds): ________________
- Feedback quality (1-5): ________________
- State advanced? ________________

**Save API Interaction**:
```bash
# Logs should show the exact prompt sent and response received
# Copy these for documentation
```

---

### Test 3: Deep Answer - LLM Rejection

This validates that the LLM correctly identifies conceptual errors.

**Steps**:
```bash
# Reset to justify stage
cat > ~/.mastery_progress.json << 'EOF'
{
  "curriculum_id": "cs336_a1",
  "current_module_index": 0,
  "current_stage": "justify",
  "completed_modules": []
}
EOF

# Submit an answer with a conceptual error
uv run python -m engine.main submit-justification "The subtract-max trick is an approximation that trades accuracy for performance. By normalizing the values to a smaller range, we reduce precision but gain speed. This is acceptable because the difference is negligible for most applications."
```

**Expected Output**:
- ✅ API call made to OpenAI
- ✅ LLM identifies the misconception ("approximation" is wrong)
- ✅ Socratic feedback guides toward correct understanding
- ✅ State remains at justify stage (not advanced)

**Record**:
- API call successful? ________________
- LLM correctly identified error? ________________
- Feedback quality (1-5): ________________
- Feedback was Socratic? ________________

---

### Test 4: Error Handling - Missing API Key

This validates graceful failure when API key is not configured.

**Steps**:
```bash
# Temporarily remove API key
mv .env .env.backup

# Try to submit an answer that would require LLM
uv run python -m engine.main submit-justification "A detailed answer that bypasses fast filter but has no obvious keywords triggering it"

# Restore API key
mv .env.backup .env
```

**Expected Output**:
- ✅ Clear error message: "OpenAI API key not found"
- ✅ Instructions to set OPENAI_API_KEY
- ✅ Link to https://platform.openai.com/api-keys
- ✅ No crash or stack trace shown to user

**Record**:
- Error message clear? ________________
- User guidance provided? ________________
- Graceful failure? ________________

---

## Documentation

### Record Results in MASTERY_WORKLOG.md

Add the following section to `MASTERY_WORKLOG.md`:

```markdown
## Sprint 6 - Ticket #20: Live LLM Integration Test

**Date**: [YYYY-MM-DD]
**Tester**: [Your Name]

### Test Results

#### Test 1: Fast Filter (No LLM)
- ✅ Fast filter correctly caught shallow answer
- Keyword matched: [keyword]
- No API call made: ✅

#### Test 2: LLM Acceptance Path
- ✅ API call successful
- Model used: gpt-4o-mini
- Response time: [X.X seconds]
- Verdict: [accept/reject]
- State advanced: [yes/no]

**Prompt Sent**:
```
[Copy exact prompt from logs]
```

**Response Received**:
```json
[Copy exact JSON response]
```

#### Test 3: LLM Rejection Path
- ✅ API call successful
- Conceptual error identified: ✅
- Feedback quality: [1-5]
- Socratic guidance: ✅

#### Test 4: Error Handling
- ✅ Missing API key handled gracefully
- Clear error message: ✅
- User guidance provided: ✅

### Conclusion
Live LLM integration is **[WORKING/NEEDS FIXES]**.

### Cost Analysis
- Total API calls: [X]
- Estimated cost: ~$[X.XX]
- Model: gpt-4o-mini

### Notes
[Any observations, issues, or recommendations]
```

---

## Validation Checklist

After completing all tests, verify:

- [ ] Fast filter works without LLM calls
- [ ] LLM correctly evaluates deep answers
- [ ] LLM identifies conceptual errors
- [ ] Socratic feedback is helpful and educational
- [ ] Error messages are clear and actionable
- [ ] API key validation works
- [ ] State transitions correctly on acceptance
- [ ] State remains unchanged on rejection
- [ ] Logs show request/response details
- [ ] No crashes or unhandled exceptions

---

## Troubleshooting

### API Key Not Found
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY=sk-...` is set correctly
- Check no typos or extra spaces

### API Connection Errors
- Verify internet connection
- Check OpenAI API status: https://status.openai.com
- Ensure API key is valid and has credits

### Rate Limiting
- Wait 60 seconds between tests
- Check your API quota at https://platform.openai.com/usage

### Unexpected Verdicts
- Review the exact prompt sent (in logs)
- Check if model answer in `justify_questions.json` is clear
- Consider if failure modes need refinement

---

## Success Criteria

The test is considered **PASSING** if:
1. ✅ Fast filter prevents unnecessary API calls
2. ✅ LLM correctly accepts high-quality answers
3. ✅ LLM correctly rejects conceptual errors
4. ✅ Feedback is educational and Socratic
5. ✅ Error handling is graceful
6. ✅ No crashes or stack traces
7. ✅ API interaction is properly logged

---

## Cleanup

After testing:
```bash
# Reset to clean state
rm -rf ~/.mastery_progress.json .mastery_engine_worktree

# Optional: Remove .env to prevent accidental API usage
# (Keep .env.example for future reference)
```

**IMPORTANT**: Document results in `MASTERY_WORKLOG.md` before cleanup!
