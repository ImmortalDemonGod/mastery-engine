# Module Generation Phase 3.1: Diagnostic Report

**Date:** November 18, 2025  
**Test Problem:** LC-200 (Number of Islands)  
**Objective:** Test robustness on different problem types

## Test Results

### Problem Characteristics

**LC-200 vs LC-912 Comparison:**

| Aspect | LC-912 (Sorting) | LC-200 (Islands) |
|--------|------------------|------------------|
| Input type | 1D array (`nums`) | 2D array (`grid`) |
| Input element | Integers | Strings (`"0"`, `"1"`) |
| Output type | 1D array | Single integer |
| Data structure | Simple list | Matrix/grid |

### Build Prompt Generation: ✅ SUCCESS

**Generated content (lines 9-45):**
```markdown
Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land)...

**Example 1:**
```
**Input:** grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
**Output:** 1
```

**Example 2:**
```
**Input:** grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
**Output:** 3
```

**Constraints:**
- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` is `'0'` or `'1'`.
```

**Result:** ✅ Complete, readable, accurate  
**Source:** Extracted from `description_html` field (full data available)

### Test Case Generation: ❌ FAILURE

**Generated content:**
```json
{
  "problem": "Number of Islands",
  "tests": [
    {
      "id": 1,
      "input": {
        "grid": "["
      },
      "expected": 1,
      "note": "Example 1"
    }
  ]
}
```

**Result:** ❌ Incomplete - truncated input  
**Source:** Extracted from `examples` array (data quality issue)

## Root Cause Analysis

### Data Quality Issue in Enriched Curriculum

**Problem:** The `examples` array has incomplete data for LC-200:

```json
{
  "examples": [
    {
      "text": "<p><strong class=\"example\">Example 1:</strong></p>",
      "input": "</strong> grid = [",
      "output": "</strong> 1",
      "explanation": ""
    }
  ]
}
```

**Why:** The enrichment script (`scripts/enrich_problems.py`) parsed examples from a specific HTML structure but didn't capture the full `<pre>` block content for examples embedded in the description.

**Impact:**
- Build prompts: ✅ No impact (uses `description_html`)
- Test cases: ❌ Cannot parse truncated input

### Where the Data IS Available

The full examples ARE present in the `description` field:

```html
<pre>
<strong>Input:</strong> grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
<strong>Output:</strong> 1
</pre>
```

**Location:** `problem_data['description']` (HTML)

## Solution Strategies

### Strategy 1: Fix Data at Source (Long-term)
**Approach:** Fix `scripts/enrich_problems.py` to capture full `<pre>` content  
**Pros:** 
- Fixes root cause
- Benefits all modules
- Clean data separation

**Cons:**
- Requires re-enrichment of all 874 problems
- Time investment for data pipeline fix

**Status:** Recommended for long-term

### Strategy 2: Fallback Parsing (Short-term)
**Approach:** When `examples` array has incomplete data, extract from `description_html`  
**Pros:**
- Works with current data
- Handles edge cases gracefully
- No re-enrichment needed

**Cons:**
- Adds complexity to generation script
- HTML parsing dependency

**Status:** Recommended for immediate implementation

### Strategy 3: Hybrid Approach (Optimal)
**Approach:** 
1. Implement fallback parsing (Strategy 2) for robustness
2. Fix enrichment script (Strategy 1) for future data quality
3. Gradually re-enrich problems as needed

**Pros:**
- Immediate functionality
- Long-term data quality
- Graceful degradation

**Cons:**
- Two systems to maintain temporarily

**Status:** Recommended

## Implementation Plan for Strategy 2

### New Method: `extract_examples_from_description()`

```python
def extract_examples_from_description(self, description_html: str) -> List[Dict]:
    """
    Extract examples from <pre> blocks in description HTML.
    
    Fallback when examples array has incomplete data.
    
    Args:
        description_html: Full problem description with examples
        
    Returns:
        List of parsed examples with input/output/explanation
    """
    soup = BeautifulSoup(description_html, 'html.parser')
    examples = []
    
    # Find all <pre> blocks (they contain examples)
    pre_blocks = soup.find_all('pre')
    
    for pre in pre_blocks:
        text = pre.get_text()
        
        # Extract input
        input_match = re.search(r'Input:\s*(.+?)(?=Output:|$)', text, re.DOTALL)
        # Extract output  
        output_match = re.search(r'Output:\s*(.+?)(?=Explanation:|$)', text, re.DOTALL)
        # Extract explanation (optional)
        explanation_match = re.search(r'Explanation:\s*(.+?)$', text, re.DOTALL)
        
        if input_match and output_match:
            examples.append({
                'input': input_match.group(1).strip(),
                'output': output_match.group(1).strip(),
                'explanation': explanation_match.group(1).strip() if explanation_match else ''
            })
    
    return examples
```

### Enhanced `generate_test_cases()` Logic

```python
def generate_test_cases(self, problem_data: Dict) -> Dict:
    # Try examples array first
    examples = problem_data.get('examples', [])
    
    # Check if examples are complete (heuristic: input longer than 20 chars)
    if not examples or all(len(ex.get('input', '')) < 20 for ex in examples):
        # Fallback: extract from description HTML
        examples = self.extract_examples_from_description(
            problem_data['description_html']
        )
    
    # Continue with existing logic...
```

## Expected Outcomes After Fix

### Test Cases for LC-200:
```json
{
  "problem": "Number of Islands",
  "tests": [
    {
      "id": 1,
      "input": {
        "grid": [
          ["1","1","1","1","0"],
          ["1","1","0","1","0"],
          ["1","1","0","0","0"],
          ["0","0","0","0","0"]
        ]
      },
      "expected": 1,
      "note": "Example 1"
    },
    {
      "id": 2,
      "input": {
        "grid": [
          ["1","1","0","0","0"],
          ["1","1","0","0","0"],
          ["0","0","1","0","0"],
          ["0","0","0","1","1"]
        ]
      },
      "expected": 3,
      "note": "Example 2"
    }
  ]
}
```

## Additional Robustness Improvements Needed

### 1. Dynamic Module Directory Creation
**Current:** Hardcoded to `sorting` directory  
**Needed:** Create module directory based on problem's pattern/topic

```python
# In main():
module_dir_name = problem_data['title'].lower().replace(' ', '_')
output_dir = generator.modules_dir / module_dir_name
```

### 2. Generic Input Parsing
**Current:** Assumes variable name is `nums` for sorting  
**Needed:** Detect variable names dynamically

```python
def parse_example_input(self, input_string: str) -> Dict:
    # Enhanced to handle any variable name
    match = re.match(r'(\w+)\s*=\s*(.+)', input_string)
    if match:
        var_name, var_value = match.groups()
        return {var_name: eval(var_value)}
```

### 3. Edge Case Generation by Problem Type
**Current:** Hardcoded sorting edge cases  
**Needed:** Different edge cases for graphs, matrices, strings

```python
def generate_edge_cases(self, problem_data: Dict, base_tests: List) -> List:
    # Detect problem type from topics
    topics = [t.lower() for t in problem_data.get('topics', [])]
    
    if 'matrix' in topics or 'grid' in problem_data['title'].lower():
        # Matrix edge cases
        return [
            {"input": {"grid": [[]]}, "expected": 0, "note": "Empty grid"},
            {"input": {"grid": [["0"]]}, "expected": 0, "note": "Single cell water"},
            {"input": {"grid": [["1"]]}, "expected": 1, "note": "Single cell land"}
        ]
    elif 'array' in topics and 'sort' in problem_data['title'].lower():
        # Sorting edge cases
        return [...existing sorting edges...]
```

## Next Steps

1. ✅ Document diagnostic findings (this file)
2. 🔄 Implement `extract_examples_from_description()` method
3. 🔄 Add fallback logic to `generate_test_cases()`
4. 🔄 Fix module directory creation
5. 🔄 Enhance input parsing for generic variable names
6. 🔄 Test on LC-200 again
7. 🔄 Validate generated test cases
8. 📋 Test on additional diverse problems
9. 📋 Long-term: Fix enrichment script

## Conclusion

The test on LC-200 successfully identified:
1. ✅ Build prompt generation is robust (works with description HTML)
2. ❌ Test case generation needs fallback (examples array incomplete)
3. 🔄 Need dynamic module directory creation
4. 🔄 Need generic input variable name parsing
5. 🔄 Need problem-type-aware edge case generation

These are exactly the kinds of issues we wanted to discover in Phase 3.1. The fixes are straightforward and will make the system truly generic.

**Status:** Diagnostic complete, ready to implement fixes.
