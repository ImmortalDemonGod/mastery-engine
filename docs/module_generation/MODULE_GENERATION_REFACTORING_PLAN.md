# Module Generation Refactoring Plan

**Date:** November 18, 2025  
**Objective:** Evolve `ingest_cp_content.py` → `generate_module.py` to leverage enriched curriculum data

## Executive Summary

We will systematically refactor the existing `scripts/ingest_cp_content.py` to become the production-grade `scripts/generate_module.py`. The key transformation is **switching from raw markdown parsing to structured data consumption** from our enriched `canonical_curriculum.json`.

---

## Current State Analysis

### What `ingest_cp_content.py` Does Well ✅

1. **Solid architecture**: Clean class-based design (`CPContentIngestion`)
2. **Complete scaffolding**: Generates all required files (build_prompt, test_cases, validator, justify)
3. **Working templates**: Mermaid diagrams, validator scripts, directory structure

### What Needs Evolution ⚠️

| Component | Current Issue | Available Improvement |
|-----------|--------------|----------------------|
| **Data Source** | Parses raw markdown files | Rich `canonical_curriculum.json` with 874 problems |
| **Problem Statement** | Placeholder: "Visit link above" | Full HTML description (1088 chars for LC-912) |
| **Examples** | Template: "TO BE FILLED" | Actual examples with input/output/explanation |
| **Test Cases** | Empty templates | Parseable from examples |
| **Constraints** | Not included | Available in description HTML |
| **Metadata** | Manual | Difficulty, acceptance rate, topic tags |

---

## Refactoring Strategy

### Phase 1: Data Source Migration (P0 - Critical)

**Goal:** Replace markdown parsing with structured data extraction.

#### Step 1.1: Create Data Extraction Function

```python
def extract_problem_data(self, problem_id: str) -> Dict:
    """
    Extract problem from canonical_curriculum.json
    
    Args:
        problem_id: LeetCode ID (e.g., "LC-912")
        
    Returns:
        Rich problem data including description, examples, metadata
    """
    curriculum_path = self.curriculum_root / "canonical_curriculum.json"
    with open(curriculum_path) as f:
        data = json.load(f)
    
    # Find problem across all topics
    for topic in data['topics']:
        for problem in topic['problems']:
            if problem['id'] == problem_id:
                return {
                    'id': problem['id'],
                    'title': problem['title'],
                    'url': problem['url'],
                    'difficulty': problem.get('difficulty'),
                    'acceptance_rate': problem.get('acceptance_rate'),
                    'description_html': problem.get('description', ''),
                    'examples': problem.get('examples', []),
                    'constraints': problem.get('constraints', []),
                    'hints': problem.get('hints', []),
                    'topics': problem.get('topics', []),
                    'similar_problems': problem.get('similar_problems', []),
                    'topic_name': topic['name']
                }
    
    raise ValueError(f"Problem {problem_id} not found in curriculum")
```

#### Step 1.2: Replace `parse_taxonomy_file`

**Before:**
```python
# Line 35-114: Complex regex parsing of markdown
pattern_data = self.parse_taxonomy_file(pattern_id)
```

**After:**
```python
# Simple structured data extraction
problem_data = self.extract_problem_data(problem_id)
```

### Phase 2: Build Prompt Enhancement (P0)

**Goal:** Generate rich, complete prompts with actual content.

#### Step 2.1: Add HTML-to-Markdown Converter

```python
from bs4 import BeautifulSoup
import html2text

def format_description(self, html_description: str) -> str:
    """Convert HTML description to clean markdown."""
    # Remove HTML tags, preserve structure
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # Don't wrap lines
    return h.handle(html_description)
```

#### Step 2.2: Add Example Formatter

```python
def format_examples(self, examples: List[Dict]) -> str:
    """Format examples for build prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"**Example {i}:**")
        formatted.append(f"```")
        formatted.append(f"Input: {ex['input']}")
        formatted.append(f"Output: {ex['output']}")
        formatted.append(f"```")
        if ex.get('explanation'):
            # Clean HTML from explanation
            clean_exp = BeautifulSoup(ex['explanation'], 'html.parser').get_text()
            formatted.append(f"*Explanation:* {clean_exp}")
        formatted.append("")
    return "\n".join(formatted)
```

#### Step 2.3: Enhanced Build Prompt Template

**Additions to current template:**
```python
def generate_build_prompt(self, problem_data: Dict) -> str:
    """Enhanced prompt with actual content."""
    prompt = f"""# Build Challenge: {problem_data['title']}

## Pattern Overview
{problem_data.get('pattern_description', 'Description from topic...')}

## Problem Statement

{self.format_description(problem_data['description_html'])}

**Difficulty:** {problem_data['difficulty']} | **Acceptance Rate:** {problem_data['acceptance_rate']}
**Topics:** {', '.join(problem_data['topics'][:5])}

**Problem Link:** {problem_data['url']}

## Examples

{self.format_examples(problem_data['examples'])}

## Implementation Requirements

1. **Language:** Python 3
2. **File:** `solution.py`
3. **Function Signature:** Derive from problem examples
4. **Constraints:** See problem statement above

... (rest of template)
"""
    return prompt
```

### Phase 3: Test Case Generation (P0)

**Goal:** Parse examples into executable test cases.

#### Step 3.1: Example Input Parser

```python
def parse_example_input(self, input_string: str, problem_type: str = "array") -> Dict:
    """
    Parse example input string to Python dict.
    
    Examples:
        "nums = [5,2,3,1]" → {"nums": [5, 2, 3, 1]}
        "n = 4, edges = [[3,1,2],[3,2,3]]" → {"n": 4, "edges": [[3,1,2],[3,2,3]]}
    """
    # Clean HTML artifacts
    input_clean = BeautifulSoup(input_string, 'html.parser').get_text()
    
    # Parse key=value pairs
    result = {}
    assignments = input_clean.split(',')
    
    current_key = None
    current_value = ""
    
    for part in assignments:
        if '=' in part:
            if current_key:
                # Finish previous assignment
                result[current_key] = eval(current_value.strip())
            
            key, value = part.split('=', 1)
            current_key = key.strip()
            current_value = value.strip()
        else:
            # Continue multi-part value (nested arrays)
            current_value += ',' + part.strip()
    
    # Finish last assignment
    if current_key:
        result[current_key] = eval(current_value.strip())
    
    return result
```

#### Step 3.2: Automated Test Case Generation

**Before:**
```python
def create_test_cases_template(self, problem_info: Dict) -> Dict:
    return {
        "tests": [
            {"id": 1, "input": {}, "expected": None, "note": "TO BE FILLED"}
        ]
    }
```

**After:**
```python
def generate_test_cases(self, problem_data: Dict) -> Dict:
    """Generate actual test cases from problem examples."""
    tests = []
    
    for i, example in enumerate(problem_data['examples'], 1):
        try:
            # Parse input and output
            input_dict = self.parse_example_input(example['input'])
            output = eval(example['output'].strip())
            
            tests.append({
                "id": i,
                "input": input_dict,
                "expected": output,
                "note": f"Example {i} from problem"
            })
        except Exception as e:
            print(f"Warning: Could not parse example {i}: {e}")
            # Fallback to template
            tests.append({
                "id": i,
                "input": {},
                "expected": None,
                "note": f"Example {i} - MANUAL PARSING REQUIRED"
            })
    
    # Add edge cases (future enhancement)
    # tests.extend(self.generate_edge_cases(problem_data))
    
    return {
        "problem": problem_data['title'],
        "source": problem_data['url'],
        "difficulty": problem_data['difficulty'],
        "tests": tests
    }
```

### Phase 4: CLI Modernization (P1)

**Goal:** Simplify interface to match new data source.

**Before:**
```bash
python scripts/ingest_cp_content.py \
  --pattern two_pointers \
  --taxonomy-path ~/DSA-Taxonomies \
  --resources https://... https://...
```

**After:**
```bash
python scripts/generate_module.py --problem-id LC-912

# Or batch mode:
python scripts/generate_module.py --topic "Sorting Algorithms"  # All problems in topic
python scripts/generate_module.py --all                          # All 874 problems
```

**Updated argparse:**
```python
def main():
    parser = argparse.ArgumentParser(
        description="Generate CP module from enriched curriculum"
    )
    parser.add_argument("--problem-id", help="Specific problem (e.g., LC-912)")
    parser.add_argument("--topic", help="Generate all problems in topic")
    parser.add_argument("--all", action="store_true", help="Generate all problems")
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    
    args = parser.parse_args()
    
    curriculum_path = Path("curricula/cp_accelerator/canonical_curriculum.json")
    generator = ModuleGenerator(curriculum_path)
    
    if args.problem_id:
        generator.generate_module(args.problem_id, force=args.force)
    elif args.topic:
        generator.generate_topic(args.topic, force=args.force)
    elif args.all:
        generator.generate_all(force=args.force)
    else:
        parser.error("Specify --problem-id, --topic, or --all")
```

---

## Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Create `extract_problem_data()` method
- [ ] Test extraction for LC-912 (sorting)
- [ ] Replace `parse_taxonomy_file` calls
- [ ] Verify directory structure preserved

### Phase 2: Content Enhancement (Week 1-2)
- [ ] Add BeautifulSoup/html2text dependencies
- [ ] Implement `format_description()` 
- [ ] Implement `format_examples()`
- [ ] Update build prompt template
- [ ] Generate LC-912 and compare to manual version

### Phase 3: Test Automation (Week 2)
- [ ] Implement `parse_example_input()`
- [ ] Add robust error handling for parsing
- [ ] Generate test cases for LC-912
- [ ] Validate against validator.sh

### Phase 4: Scale & Polish (Week 3)
- [ ] Update CLI arguments
- [ ] Add batch generation modes
- [ ] Test on 10 diverse problems
- [ ] Document edge cases and limitations

### Phase 5: Integration (Week 3-4)
- [ ] Rename script to `generate_module.py`
- [ ] Update `scripts/README.md`
- [ ] Update CI workflows if needed
- [ ] Generate modules for full curriculum

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Manual effort per module | 2-4 hours | 5-10 minutes |
| Build prompt completeness | 40% (placeholders) | 95% (actual content) |
| Test case accuracy | 0% (empty templates) | 90%+ (auto-parsed) |
| Modules created | 1 (sorting) | 874 (all free problems) |
| Data consistency | Manual → prone to errors | Automated → guaranteed |

---

## Risk Mitigation

### Risk 1: Example Parsing Complexity
**Mitigation:** Start with simple cases (arrays, numbers), add support incrementally for complex types (trees, graphs)

### Risk 2: HTML Formatting Quality
**Mitigation:** Manual review of first 10 generated modules, iterate on formatters

### Risk 3: Breaking Existing Module
**Mitigation:** 
1. Keep existing `sorting/` module as reference
2. Generate to `sorting-generated/` first
3. Compare outputs before replacing

---

## Next Steps (Developer Action Plan)

1. **Create feature branch:**
   ```bash
   git checkout -b feature/generate-module-from-curriculum
   ```

2. **Start with Phase 1:**
   - Copy `ingest_cp_content.py` to `generate_module.py`
   - Add `extract_problem_data()` method
   - Test extraction: `python -c "from scripts.generate_module import *; ..."`

3. **Validate on LC-912:**
   ```bash
   python scripts/generate_module.py --problem-id LC-912 --output /tmp/sorting-test
   diff -r curricula/cp_accelerator/modules/sorting /tmp/sorting-test
   ```

4. **Iterate and expand** following the phased plan above.

---

## Conclusion

This refactoring transforms a **prototype scaffold tool** into a **production module generator** by leveraging our investment in curriculum enrichment. The result will be:

- ✅ **Consistent quality** across all modules
- ✅ **Scalability** to 874 problems
- ✅ **Maintainability** through automation
- ✅ **Rich content** from official sources

The existing code provides a solid foundation; we're enhancing it, not replacing it.
