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
from jinja2 import Environment, FileSystemLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ModuleGenerator:
    """Generate module assets from enriched curriculum data."""
    
    def __init__(self, curriculum_root: Path):
        self.curriculum_root = Path(curriculum_root)
        self.modules_dir = curriculum_root / "modules"  # Legacy
        self.patterns_dir = curriculum_root / "patterns"  # Library mode
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
        
    def format_description(self, html_description: str) -> str:
        """
        Convert HTML description to clean markdown.
        
        Args:
            html_description: HTML content from LeetCode
            
        Returns:
            Markdown-formatted description
        """
        soup = BeautifulSoup(html_description, 'html.parser')
        
        # Convert common HTML tags to markdown
        # <strong> -> **text**
        for tag in soup.find_all('strong'):
            tag.replace_with(f"**{tag.get_text()}**")
        
        # <code> -> `text`
        for tag in soup.find_all('code'):
            tag.replace_with(f"`{tag.get_text()}`")
        
        # <pre> -> code block
        for tag in soup.find_all('pre'):
            code_text = tag.get_text()
            tag.replace_with(f"\n```\n{code_text}\n```\n")
        
        # <ul> and <li> -> markdown lists
        for ul in soup.find_all('ul'):
            items = []
            for li in ul.find_all('li'):
                items.append(f"- {li.get_text().strip()}")
            ul.replace_with('\n' + '\n'.join(items) + '\n')
        
        # Get final text
        text = soup.get_text()
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        
        return text.strip()
    
    def format_examples(self, examples: List[Dict]) -> str:
        """
        Format examples for build prompt.
        
        Args:
            examples: List of example dicts with input/output/explanation
            
        Returns:
            Formatted examples string
        """
        formatted = []
        
        for i, ex in enumerate(examples, 1):
            formatted.append(f"**Example {i}:**")
            formatted.append("")
            
            # Clean and format input
            inp_soup = BeautifulSoup(ex.get('input', ''), 'html.parser')
            input_text = inp_soup.get_text().strip()
            formatted.append(f"```")
            formatted.append(f"Input: {input_text}")
            
            # Clean and format output
            out_soup = BeautifulSoup(ex.get('output', ''), 'html.parser')
            output_text = out_soup.get_text().strip()
            formatted.append(f"Output: {output_text}")
            formatted.append(f"```")
            
            # Add explanation if present
            if ex.get('explanation'):
                exp_soup = BeautifulSoup(ex['explanation'], 'html.parser')
                explanation_text = exp_soup.get_text().strip()
                if explanation_text:
                    formatted.append(f"*Explanation:* {explanation_text}")
            
            formatted.append("")
        
        return '\n'.join(formatted)
    
    def format_constraints(self, constraints: List[str], description_html: str = "") -> str:
        """
        Format constraints from either constraints list or extract from HTML.
        
        Args:
            constraints: List of constraint strings
            description_html: HTML description to extract from if no constraints list
            
        Returns:
            Formatted constraints string
        """
        if constraints:
            return '\n'.join(f"- {c}" for c in constraints)
        
        # Try to extract from description HTML
        soup = BeautifulSoup(description_html, 'html.parser')
        
        # Look for "Constraints:" section
        constraints_text = []
        found_constraints = False
        
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if 'Constraints:' in text or 'constraints:' in text.lower():
                found_constraints = True
                continue
            if found_constraints and text:
                # Extract constraint lines
                if text.startswith(('-', '•', '*')):
                    constraints_text.append(text)
                elif '<' in text or '≤' in text or '<=' in text:
                    constraints_text.append(f"- {text}")
        
        if constraints_text:
            return '\n'.join(constraints_text)
        
        return ""
    
    def extract_examples_from_description(self, description_html: str) -> List[Dict]:
        """
        Extract examples from <pre> blocks in description HTML.
        
        Fallback method when examples array has incomplete/truncated data.
        
        Args:
            description_html: Full problem description with examples in <pre> tags
            
        Returns:
            List of parsed examples with input/output/explanation
        """
        soup = BeautifulSoup(description_html, 'html.parser')
        examples = []
        
        # Find all <pre> blocks (they typically contain examples)
        pre_blocks = soup.find_all('pre')
        
        for pre in pre_blocks:
            text = pre.get_text()
            
            # Skip if this doesn't look like an example
            if 'Input:' not in text and 'Output:' not in text:
                continue
            
            # Extract input (everything between "Input:" and "Output:")
            input_match = re.search(r'Input:\s*(.+?)(?=Output:|Explanation:|$)', text, re.DOTALL | re.IGNORECASE)
            # Extract output (everything between "Output:" and "Explanation:" or end)
            output_match = re.search(r'Output:\s*(.+?)(?=Explanation:|$)', text, re.DOTALL | re.IGNORECASE)
            # Extract explanation (optional)
            explanation_match = re.search(r'Explanation:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
            
            if input_match and output_match:
                examples.append({
                    'input': input_match.group(1).strip(),
                    'output': output_match.group(1).strip(),
                    'explanation': explanation_match.group(1).strip() if explanation_match else ''
                })
        
        return examples
        
    def generate_build_prompt(self, problem_data: Dict, topic_data: Dict = None,  
                             resources: List[str] = None) -> str:
        """
        Generate build_prompt.txt content using Jinja2 template.
        
        Args:
            problem_data: Extracted problem data from canonical curriculum
            topic_data: Topic information (description, etc.)
            resources: Learning resource URLs
            
        Returns:
            Formatted build prompt string
        """
        # Set up Jinja2 environment
        script_dir = Path(__file__).parent
        template_dir = script_dir / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template('build_prompt.jinja2')
        
        # Format the data
        # Support both 'description' and 'description_html' field names
        description = problem_data.get('description_html') or problem_data.get('description', '')
        
        formatted_problem = {
            'title': problem_data['title'],
            'url': problem_data['url'],
            'difficulty': problem_data.get('difficulty', 'Unknown'),
            'acceptance_rate': problem_data.get('acceptance_rate', 'N/A'),
            'topics': problem_data.get('topics', [])[:5],  # Top 5 topics
            'description_formatted': self.format_description(description),
            'examples_formatted': self.format_examples(problem_data.get('examples', [])),
            'constraints_formatted': self.format_constraints(
                problem_data.get('constraints', []),
                description
            ),
            'hints': problem_data.get('hints', [])
        }
        
        # Topic data (if available)
        topic = topic_data or {'description': 'Learn to solve problems using systematic approaches.'}
        
        # Render template
        return template.render(
            problem=formatted_problem,
            topic=topic,
            resources=resources or []
        )
        
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
        
        # Get examples (with fallback to description if incomplete)
        examples = problem_data.get('examples', [])
        
        # Check if examples are complete (heuristic: input should be > 20 chars)
        if not examples or all(len(ex.get('input', '')) < 20 for ex in examples):
            print(f"   ⚠️  Examples array incomplete, extracting from description HTML...")
            examples = self.extract_examples_from_description(
                problem_data.get('description_html', problem_data.get('description', ''))
            )
            if examples:
                print(f"   ✅ Extracted {len(examples)} examples from description")
        
        for i, example in enumerate(examples, 1):
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
    parser.add_argument("--all", action="store_true", help="Generate all problems (use with --limit-per-pattern)")
    parser.add_argument("--limit-per-pattern", type=int, help="Limit number of problems per pattern (for breadth-first population)")
    
    args = parser.parse_args()
    
    # Branch: batch mode or single problem mode
    if args.all:
        return batch_generate_all(args)
    
    if not args.problem_id:
        parser.error("--problem-id is required (or use --all for batch mode)")
    
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
        # Create module directory based on problem title
        module_name = problem_data['title'].lower().replace(' ', '_').replace('-', '_')
        # Remove special characters
        module_name = re.sub(r'[^a-z0-9_]', '', module_name)
        output_dir = generator.modules_dir / module_name
        print(f"   📁 Module directory: {module_name}")
    
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
    
    # Generate build prompt
    print(f"\n📝 Generating build_prompt.txt...")
    
    # Get topic description from curriculum
    topic_desc = None
    for topic in generator.curriculum_data['topics']:
        for p in topic['problems']:
            if p['id'] == args.problem_id:
                topic_desc = {'description': topic.get('description', 'Learn to solve problems systematically.')}
                break
    
    # Get resources (from roadmap or defaults)
    resources = [
        "https://www.youtube.com/watch?v=kPRA0W1kECg",
        "https://www.geeksforgeeks.org/sorting-algorithms/"
    ]
    
    build_prompt = generator.generate_build_prompt(problem_data, topic_desc, resources)
    build_prompt_path = output_dir / "build_prompt.txt"
    
    if build_prompt_path.exists() and not args.force:
        print(f"   ⚠️  File exists: {build_prompt_path}")
        print(f"   💡 Use --force to overwrite")
    else:
        with open(build_prompt_path, 'w') as f:
            f.write(build_prompt)
        print(f"   ✅ Generated: {build_prompt_path}")
        print(f"   📊 Size: {len(build_prompt)} chars")
    
    print(f"\n{'='*70}")
    print(f"✅ Phase 2 Complete!")
    print(f"{'='*70}")
    print(f"\n📁 Output: {output_dir}")
    print(f"\n🔍 Compare generated files to originals:")
    print(f"   git diff {test_cases_path}")
    print(f"   git diff {build_prompt_path}")
    
    return 0


def batch_generate_all(args):
    """Generate problems in batch mode for Library curriculum."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    curriculum_root = repo_root / "curricula" / "cp_accelerator"
    
    generator = ModuleGenerator(curriculum_root)
    
    limit = args.limit_per_pattern or 999  # No limit if not specified
    
    print(f"\n{'='*70}")
    print(f"Batch Generation - Library Mode")
    print(f"{'='*70}")
    print(f"Limit per pattern: {limit if limit < 999 else 'None (all)'}")
    print(f"Force overwrite: {args.force}")
    print()
    
    total_generated = 0
    total_skipped = 0
    
    # Read canonical curriculum
    with open(generator.canonical_path) as f:
        canonical = json.load(f)
    
    # Iterate through topics (= patterns)
    for topic in canonical['topics']:
        topic_name = topic['name']
        # Convert to pattern_id (e.g., "Sorting Algorithms" -> "sorting")
        pattern_id = topic_name.lower().replace(' ', '_').replace('-', '_')
        pattern_id = re.sub(r'[^a-z0-9_]', '', pattern_id)
        pattern_id = pattern_id.replace('algorithms', '').replace('algorithm', '').strip('_')
        
        print(f"\n📦 Pattern: {topic_name} ({pattern_id})")
        print(f"   Problems available: {len(topic['problems'])}")
        
        # Create pattern directory
        pattern_dir = generator.patterns_dir / pattern_id
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        problems_dir = pattern_dir / "problems"
        problems_dir.mkdir(exist_ok=True)
        
        generated_count = 0
        skipped_count = 0
        
        # Generate up to limit problems for this pattern
        for problem in topic['problems'][:limit]:
            problem_id = problem['id']
            
            # Convert problem ID to directory name (e.g., "LC-912" -> "lc_912")
            problem_slug = problem_id.lower().replace('-', '_')
            problem_dir = problems_dir / problem_slug
            
            # Check if exists
            if problem_dir.exists() and not args.force:
                print(f"   ⏭️  {problem_id}: Already exists, skipping")
                skipped_count += 1
                continue
            
            # Generate problem content
            try:
                problem_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate build_prompt.txt
                problem_data = problem  # Already has all data from canonical
                build_prompt = generator.generate_build_prompt(problem_data, 
                    {'description': topic.get('description', 'Master this pattern.')},
                    topic.get('resources', [])[:3]  # Top 3 resources
                )
                (problem_dir / "build_prompt.txt").write_text(build_prompt)
                
                # Generate test_cases.json
                test_cases = generator.generate_test_cases(problem_data)
                with open(problem_dir / "test_cases.json", 'w') as f:
                    json.dump(test_cases, f, indent=2)
                
                # Generate validator.sh
                validator = generator.create_validator_template(problem_data)
                validator_path = problem_dir / "validator.sh"
                validator_path.write_text(validator)
                validator_path.chmod(0o755)
                
                print(f"   ✅ {problem_id}: Generated at {problem_dir.name}/")
                generated_count += 1
                
            except Exception as e:
                print(f"   ❌ {problem_id}: Error - {e}")
        
        total_generated += generated_count
        total_skipped += skipped_count
        print(f"   Summary: {generated_count} generated, {skipped_count} skipped")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Batch Generation Complete!")
    print(f"{'='*70}")
    print(f"Total generated: {total_generated}")
    print(f"Total skipped:   {total_skipped}")
    print(f"\n📁 Output: {generator.patterns_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    main()
