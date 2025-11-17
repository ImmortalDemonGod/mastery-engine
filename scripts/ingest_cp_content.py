#!/usr/bin/env python3
"""
Competitive Programming Content Ingestion Pipeline

This script automates Phase 1 of the CP Accelerator curriculum generation:
1. Parse DSA Taxonomy markdown files to extract pattern explanations
2. Select canonical problems from the taxonomy
3. Generate build_prompt.txt with embedded problem statements
4. Scaffold justify_questions.json using LLM
5. Create validator.sh and test_cases.json from public test cases

Usage:
    python scripts/ingest_cp_content.py --pattern two_pointers --taxonomy-path ~/DSA-Taxonomies
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class CPContentIngestion:
    """Automated content ingestion for CP Accelerator curriculum."""
    
    def __init__(self, taxonomy_root: Path, curriculum_root: Path):
        self.taxonomy_root = Path(taxonomy_root)
        self.curriculum_root = Path(curriculum_root)
        self.modules_dir = curriculum_root / "modules"
        
    def parse_taxonomy_file(self, pattern_name: str) -> Dict:
        """
        Parse a taxonomy markdown file to extract pattern info.
        
        Args:
            pattern_name: Name of the pattern file (e.g., "Two Pointers")
            
        Returns:
            Dict containing pattern info: description, mermaid_diagram, problems
        """
        # Try multiple naming conventions
        potential_files = [
            self.taxonomy_root / "Taxonomies" / f"{pattern_name}.md",
            self.taxonomy_root / "Taxonomies" / f"{pattern_name.replace('_', ' ').title()}.md",
        ]
        
        taxonomy_file = None
        for f in potential_files:
            if f.exists():
                taxonomy_file = f
                break
                
        if not taxonomy_file:
            raise FileNotFoundError(f"Could not find taxonomy file for pattern: {pattern_name}")
            
        with open(taxonomy_file) as f:
            content = f.read()
            
        # Extract description (usually first paragraph after header)
        description_match = re.search(r'^#[^#].*?\n\n(.*?)\n\n', content, re.MULTILINE | re.DOTALL)
        description = description_match.group(1) if description_match else ""
        
        # Extract mermaid diagram
        mermaid_match = re.search(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
        mermaid_diagram = mermaid_match.group(1) if mermaid_match else ""
        
        # Extract problem links (looking for LeetCode problem numbers)
        problem_pattern = r'\[(\d+)\.\s+(.*?)\]\((https://leetcode\.com/problems/[^)]+)\)'
        problems = []
        for match in re.finditer(problem_pattern, content):
            problems.append({
                "number": match.group(1),
                "title": match.group(2),
                "url": match.group(3)
            })
            
        return {
            "description": description.strip(),
            "mermaid_diagram": mermaid_diagram.strip(),
            "problems": problems
        }
        
    def select_canonical_problem(self, problems: List[Dict]) -> Optional[Dict]:
        """
        Select the best canonical problem for the Build stage.
        
        Strategy: Choose first Easy/Medium problem as it's usually representative.
        """
        if not problems:
            return None
        # For now, just return the first problem
        # In production, you'd fetch difficulty from LeetCode API
        return problems[0] if problems else None
        
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
        
    def create_test_cases_template(self, problem_info: Dict) -> Dict:
        """Generate test_cases.json template."""
        return {
            "problem": problem_info["title"],
            "source": problem_info["url"],
            "note": "These are example test cases from the problem statement. Full validation requires online judge submission.",
            "tests": [
                {
                    "id": 1,
                    "input": {},
                    "expected": None,
                    "note": "Example 1 from problem statement - TO BE FILLED"
                }
            ]
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
    parser = argparse.ArgumentParser(description="Ingest CP content from DSA Taxonomy")
    parser.add_argument("--pattern", required=True, help="Pattern ID (e.g., two_pointers)")
    parser.add_argument("--taxonomy-path", required=True, help="Path to DSA-Taxonomies repo")
    parser.add_argument("--rating-bracket", default="0-999", help="Rating bracket (default: 0-999)")
    parser.add_argument("--resources", nargs="+", default=[], 
                       help="Learning resource URLs from roadmap")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    curriculum_root = repo_root / "curricula" / "cp_accelerator"
    
    # Run ingestion
    ingestion = CPContentIngestion(
        taxonomy_root=Path(args.taxonomy_path),
        curriculum_root=curriculum_root
    )
    
    ingestion.ingest_pattern(
        pattern_id=args.pattern,
        rating_bracket=args.rating_bracket,
        roadmap_resources=args.resources
    )
    
    print("\n" + "="*60)
    print("Ingestion complete! Next steps:")
    print("="*60)
    print("1. Review generated files in curricula/cp_accelerator/modules/")
    print("2. Manually curate justify questions")
    print("3. Create reference solution in .solutions/")
    print("4. Author bug patches in bugs/")
    print("5. Run: engine init cp_accelerator")


if __name__ == "__main__":
    main()
