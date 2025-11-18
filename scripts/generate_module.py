#!/usr/bin/env python3
"""
Module Generation from Enriched Curriculum

This script generates complete module assets from our enriched curriculum data:
1. Extract problem data from canonical_curriculum.json
2. Generate build_prompt.txt with actual problem statements and examples
3. Auto-generate test_cases.json from parsed examples
4. Scaffold justify_questions.json (future: LLM-enhanced)
5. Create validator.sh templates

Usage:
    python scripts/generate_module.py --problem-id LC-912
    python scripts/generate_module.py --topic "Sorting Algorithms"
    python scripts/generate_module.py --all
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModuleGenerator:
    """Generate module assets from enriched curriculum data."""
    
    def __init__(self, curriculum_root: Path):
        self.curriculum_root = Path(curriculum_root)
        self.modules_dir = curriculum_root / "modules"
        self.canonical_path = curriculum_root / "canonical_curriculum.json"
        
        # Load curriculum data
        with open(self.canonical_path) as f:
            self.curriculum_data = json.load(f)
        
    def extract_problem_data(self, problem_id: str) -> Dict:
        """
        Extract problem data from canonical_curriculum.json.
        
        Args:
            problem_id: LeetCode problem ID (e.g., "LC-912")
            
        Returns:
            Rich problem data including description, examples, metadata
        """
        # Find problem across all topics
        for topic in self.curriculum_data['topics']:
            for problem in topic['problems']:
                if problem['id'] == problem_id:
                    return {
                        'id': problem['id'],
                        'title': problem['title'],
                        'url': problem['url'],
                        'difficulty': problem.get('difficulty', 'Unknown'),
                        'acceptance_rate': problem.get('acceptance_rate', 'N/A'),
                        'description_html': problem.get('description', ''),
                        'examples': problem.get('examples', []),
                        'constraints': problem.get('constraints', []),
                        'hints': problem.get('hints', []),
                        'topics': problem.get('topics', []),
                        'similar_problems': problem.get('similar_problems', []),
                        'topic_name': topic['name'],
                        'has_solution': problem.get('has_solution', False)
                    }
        
        raise ValueError(f"Problem {problem_id} not found in curriculum")
        
    def parse_example_input(self, input_string: str) -> Dict:
        """
        Parse example input string to Python dict.
        
        Examples:
            "nums = [5,2,3,1]" → {"nums": [5, 2, 3, 1]}
            "n = 4, edges = [[3,1,2]]" → {"n": 4, "edges": [[3,1,2]]}
        """
        # Clean HTML tags
        soup = BeautifulSoup(input_string, 'html.parser')
        input_clean = soup.get_text().strip()
        
        # Parse key=value pairs
        result = {}
        
        # Handle simple case: single variable
        if '=' in input_clean:
            # Split by comma, but be careful with nested structures
            parts = []
            current = ""
            bracket_depth = 0
            
            for char in input_clean:
                if char in '[{':
                    bracket_depth += 1
                elif char in ']}':
                    bracket_depth -= 1
                elif char == ',' and bracket_depth == 0:
                    parts.append(current.strip())
                    current = ""
                    continue
                current += char
            
            if current.strip():
                parts.append(current.strip())
            
            # Parse each part
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        result[key] = eval(value)
                    except:
                        result[key] = value
        
        return result
        
    def generate_build_prompt(self, pattern_data: Dict, canonical_problem: Dict, 
                             roadmap_resources: List[str]) -> str:
        """
        Generate build_prompt.txt content.
        
        Args:
            pattern_data: Parsed taxonomy data
            canonical_problem: Selected problem for implementation
            roadmap_resources: Links to video/blog tutorials from roadmap
            
        Returns:
            Formatted build prompt string
        """
        prompt = f"""# Build Challenge: {canonical_problem['title']}

## Pattern Overview

{pattern_data['description']}

## Pattern Taxonomy

```mermaid
{pattern_data['mermaid_diagram']}
```

## Your Task

Implement a solution to the following problem, which is a canonical example of this pattern.

### Problem: {canonical_problem['number']}. {canonical_problem['title']}

**Problem Link:** {canonical_problem['url']}

**Problem Statement:**
[Note: Visit the problem link above for the complete statement, examples, and constraints]

## Implementation Requirements

1. **Language:** Python 3 (or your preferred language)
2. **File:** `solution.py`
3. **Function Signature:** Will be provided based on the problem
4. **Constraints:** Must pass all example test cases from the problem statement

## Local Validation

After implementing your solution:

```bash
engine submit
```

**Important Note on Validation:**
- The local validator runs against example test cases extracted from the problem statement
- Passing local tests is **necessary but not sufficient** for full correctness
- For complete validation, submit your solution to {canonical_problem['url']}
- The Mastery Engine focuses on verifying you understand the pattern and can implement a working solution

## Learning Resources

These curated resources will help you understand the pattern deeply:

"""
        
        for idx, resource in enumerate(roadmap_resources, 1):
            prompt += f"{idx}. {resource}\n"
            
        prompt += """
## Common Pitfalls

- Off-by-one errors in pointer movement
- Not handling edge cases (empty arrays, single elements)
- Incorrect termination conditions

## Hints

If you're stuck, consider:
1. Drawing the pointer positions on paper for a small example
2. What invariant should be maintained at each step?
3. What is the termination condition?

## Submission

Once your solution passes local tests:
```bash
engine submit
```
"""
        return prompt
        
    def scaffold_justify_questions(self, pattern_name: str) -> List[Dict]:
        """
        Generate scaffolded justify_questions.json.
        
        In production, this would use an LLM to generate deep questions.
        For now, returns template questions.
        """
        return [
            {
                "id": f"{pattern_name}_conceptual",
                "question": f"Explain the core invariant that the {pattern_name.replace('_', ' ')} pattern maintains. Why does this invariant guarantee correctness?",
                "model_answer": "[To be filled by expert curator]",
                "failure_modes": [
                    {
                        "category": "Vague Hand-Waving",
                        "keywords": ["moves", "checks", "looks at"],
                        "feedback": "Instead of describing what the code does, explain *why* the algorithm is correct. What property is preserved at each step?"
                    }
                ]
            },
            {
                "id": f"{pattern_name}_complexity",
                "question": f"Analyze the time and space complexity of your solution. How does {pattern_name.replace('_', ' ')} achieve better complexity than a naive approach?",
                "model_answer": "[To be filled by expert curator]",
                "failure_modes": []
            }
        ]
        
    def create_validator_template(self, problem_info: Dict) -> str:
        """Generate validator.sh template."""
        return f"""#!/bin/bash
# Local validator for {problem_info['title']}
# Runs solution against example test cases

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
TEST_CASES="$SCRIPT_DIR/test_cases.json"

# Check if solution exists
if [ ! -f "$SCRIPT_DIR/solution.py" ]; then
    echo "❌ solution.py not found"
    exit 1
fi

# Run tests
python3 << 'EOF'
import json
import sys
from pathlib import Path

# Import user solution
sys.path.insert(0, str(Path(__file__).parent))
from solution import solve

# Load test cases
with open("test_cases.json") as f:
    test_cases = json.load(f)

passed = 0
failed = 0

for i, test in enumerate(test_cases["tests"], 1):
    try:
        result = solve(**test["input"])
        expected = test["expected"]
        
        if result == expected:
            print(f"✓ Test {{i}}: PASS")
            passed += 1
        else:
            print(f"✗ Test {{i}}: FAIL")
            print(f"  Input: {{test['input']}}")
            print(f"  Expected: {{expected}}")
            print(f"  Got: {{result}}")
            failed += 1
    except Exception as e:
        print(f"✗ Test {{i}}: ERROR - {{e}}")
        failed += 1

print(f"\\nResults: {{passed}}/{{passed + failed}} passed")
sys.exit(0 if failed == 0 else 1)
EOF
"""
        
    def generate_test_cases(self, problem_data: Dict) -> Dict:
        """
        Generate test_cases.json from problem examples.
        
        Args:
            problem_data: Extracted problem data with examples
            
        Returns:
            Complete test cases JSON structure
        """
        tests = []
        
        for i, example in enumerate(problem_data['examples'], 1):
            try:
                # Parse input
                input_dict = self.parse_example_input(example['input'])
                
                # Parse output (clean HTML and evaluate)
                soup = BeautifulSoup(example['output'], 'html.parser')
                output_clean = soup.get_text().strip()
                try:
                    expected = eval(output_clean)
                except:
                    expected = output_clean
                
                # Clean explanation
                exp_soup = BeautifulSoup(example.get('explanation', ''), 'html.parser')
                explanation = exp_soup.get_text().strip()
                
                tests.append({
                    "id": i,
                    "input": input_dict,
                    "expected": expected,
                    "note": f"Example {i}" + (f": {explanation[:50]}..." if explanation else "")
                })
            except Exception as e:
                print(f"Warning: Could not parse example {i}: {e}")
                # Add placeholder
                tests.append({
                    "id": i,
                    "input": {},
                    "expected": None,
                    "note": f"Example {i} - PARSE ERROR: {str(e)[:50]}"
                })
        
        # Add common edge cases for array sorting
        if 'sort' in problem_data['title'].lower():
            tests.extend([
                {
                    "id": len(tests) + 1,
                    "input": {"nums": [1]},
                    "expected": [1],
                    "note": "Single element"
                },
                {
                    "id": len(tests) + 2,
                    "input": {"nums": []},
                    "expected": [],
                    "note": "Empty array edge case"
                },
                {
                    "id": len(tests) + 3,
                    "input": {"nums": [3, 2, 1]},
                    "expected": [1, 2, 3],
                    "note": "Reverse sorted"
                },
                {
                    "id": len(tests) + 4,
                    "input": {"nums": [1, 2, 3, 4, 5]},
                    "expected": [1, 2, 3, 4, 5],
                    "note": "Already sorted"
                },
                {
                    "id": len(tests) + 5,
                    "input": {"nums": [-5, -1, 0, 3, 8]},
                    "expected": [-5, -1, 0, 3, 8],
                    "note": "Negative numbers"
                }
            ])
        
        return {
            "problem": problem_data['title'],
            "source": problem_data['url'],
            "note": "These are example test cases from the problem statement. Full validation requires online judge submission.",
            "tests": tests
        }
        
    def ingest_pattern(self, pattern_id: str, rating_bracket: str, 
                      roadmap_resources: List[str]) -> None:
        """
        Complete ingestion pipeline for one pattern.
        
        Args:
            pattern_id: Module ID (e.g., "two_pointers")
            rating_bracket: Rating range (e.g., "0-999")
            roadmap_resources: List of resource URLs from roadmap
        """
        print(f"\n{'='*60}")
        print(f"Ingesting pattern: {pattern_id}")
        print(f"{'='*60}\n")
        
        # Parse taxonomy
        print("📖 Parsing taxonomy file...")
        pattern_data = self.parse_taxonomy_file(pattern_id)
        print(f"   Found {len(pattern_data['problems'])} problems in taxonomy")
        
        # Select canonical problem
        print("🎯 Selecting canonical problem...")
        canonical = self.select_canonical_problem(pattern_data['problems'])
        if not canonical:
            raise ValueError(f"No problems found for pattern: {pattern_id}")
        print(f"   Selected: {canonical['number']}. {canonical['title']}")
        
        # Create module directory
        module_dir = self.modules_dir / pattern_id
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / "bugs").mkdir(exist_ok=True)
        
        # Generate build_prompt.txt
        print("📝 Generating build_prompt.txt...")
        build_prompt = self.generate_build_prompt(pattern_data, canonical, roadmap_resources)
        (module_dir / "build_prompt.txt").write_text(build_prompt)
        
        # Scaffold justify_questions.json
        print("❓ Scaffolding justify_questions.json...")
        justify_questions = self.scaffold_justify_questions(pattern_id)
        with open(module_dir / "justify_questions.json", 'w') as f:
            json.dump(justify_questions, f, indent=2)
            
        # Create validator.sh
        print("⚙️  Creating validator.sh...")
        validator = self.create_validator_template(canonical)
        validator_path = module_dir / "validator.sh"
        validator_path.write_text(validator)
        validator_path.chmod(0o755)
        
        # Create test_cases.json template
        print("🧪 Creating test_cases.json template...")
        test_cases = self.create_test_cases_template(canonical)
        with open(module_dir / "test_cases.json", 'w') as f:
            json.dump(test_cases, f, indent=2)
            
        print(f"\n✅ Module '{pattern_id}' scaffolded successfully!")
        print(f"📁 Location: {module_dir}")
        print(f"\n⚠️  Manual curation required:")
        print(f"   1. Fill test cases in test_cases.json")
        print(f"   2. Refine justify_questions.json")
        print(f"   3. Create bug patches in bugs/ directory")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CP module from enriched curriculum"
    )
    parser.add_argument("--problem-id", help="Specific problem (e.g., LC-912)")
    parser.add_argument("--output-dir", help="Output directory (defaults to modules/[slug])")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    if not args.problem_id:
        parser.error("--problem-id is required for PoC")
    
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    curriculum_root = repo_root / "curricula" / "cp_accelerator"
    
    # Initialize generator
    print(f"\n{'='*70}")
    print(f"Module Generation PoC: {args.problem_id}")
    print(f"{'='*70}\n")
    
    generator = ModuleGenerator(curriculum_root)
    
    # Extract problem data
    print(f"📊 Extracting problem data from canonical_curriculum.json...")
    try:
        problem_data = generator.extract_problem_data(args.problem_id)
        print(f"   ✅ Found: {problem_data['title']}")
        print(f"   📍 Topic: {problem_data['topic_name']}")
        print(f"   📈 Difficulty: {problem_data['difficulty']} ({problem_data['acceptance_rate']})")
        print(f"   📝 Examples: {len(problem_data['examples'])}")
    except ValueError as e:
        print(f"   ❌ Error: {e}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use existing module directory (sorting for LC-912)
        output_dir = generator.modules_dir / "sorting"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test cases
    print(f"\n🧪 Generating test_cases.json...")
    test_cases = generator.generate_test_cases(problem_data)
    test_cases_path = output_dir / "test_cases.json"
    
    if test_cases_path.exists() and not args.force:
        print(f"   ⚠️  File exists: {test_cases_path}")
        print(f"   💡 Use --force to overwrite")
    else:
        with open(test_cases_path, 'w') as f:
            json.dump(test_cases, f, indent=2)
        print(f"   ✅ Generated: {test_cases_path}")
        print(f"   📊 Test count: {len(test_cases['tests'])}")
    
    print(f"\n{'='*70}")
    print(f"✅ PoC Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output: {output_dir}")
    print(f"\n🔍 Next: Compare generated file to original:")
    print(f"   git diff {test_cases_path}")
    
    return 0


if __name__ == "__main__":
    main()
