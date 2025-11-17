#!/usr/bin/env python3
"""
Enrich canonical curriculum with full LeetCode problem details.

Uses the LeetCode API (https://leetcode-api-pied.vercel.app) to fetch:
- Problem description
- Examples with inputs/outputs
- Constraints
- Hints
- Editorial solutions
- Complexity analysis
- Similar problems

This is CRITICAL - without this data, we cannot generate module content.

Usage:
    python scripts/enrich_problems.py
    python scripts/enrich_problems.py --rate-limit 2  # 2 second delay between requests
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    import requests
except ImportError:
    print("❌ ERROR: requests library not found")
    print("   Install: pip install requests")
    sys.exit(1)


class LeetCodeEnricher:
    """Fetch full problem details from LeetCode API."""
    
    API_BASE = "https://leetcode-api-pied.vercel.app"
    
    def __init__(self, rate_limit_seconds: float = 1.0):
        self.rate_limit = rate_limit_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CP-Accelerator-Curriculum-Builder/1.0'
        })
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "premium": 0
        }
    
    def fetch_problem(self, problem_id: str) -> Optional[Dict]:
        """
        Fetch full problem details from LeetCode API.
        
        Args:
            problem_id: Problem ID like "LC-1" or "LC-167"
        
        Returns:
            Dict with full problem data or None if failed
        """
        # Extract numeric ID (LC-1 → 1)
        if problem_id.startswith("LC-"):
            numeric_id = problem_id[3:]
        else:
            numeric_id = problem_id
        
        url = f"{self.API_BASE}/problem/{numeric_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if premium - skip immediately
            if data.get("isPaidOnly", False):
                return {"is_premium": True, "skipped": True}
            
            # Check if content is null (problem doesn't exist or API error)
            if data.get("content") is None:
                return None
            
            # Rate limiting
            time.sleep(self.rate_limit)
            
            return self._parse_problem_data(data)
        
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed to fetch {problem_id}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"  ❌ Invalid JSON for {problem_id}")
            return None
    
    def _parse_problem_data(self, raw_data: Dict) -> Dict:
        """Parse API response into structured problem data."""
        
        # Parse stats (JSON string)
        stats = {}
        stats_str = raw_data.get("stats", "{}")
        if isinstance(stats_str, str):
            try:
                stats = json.loads(stats_str)
            except json.JSONDecodeError:
                stats = {}
        
        # Parse similar problems (JSON string)
        similar = []
        similar_str = raw_data.get("similarQuestions", "[]")
        if isinstance(similar_str, str):
            try:
                similar = json.loads(similar_str)
            except json.JSONDecodeError:
                similar = []
        
        # Extract solution content
        solution_data = raw_data.get("solution", {})
        solution_content = ""
        if isinstance(solution_data, dict):
            solution_content = solution_data.get("content", "")
        
        # Extract key fields
        problem = {
            "description": raw_data.get("content", ""),
            "difficulty": raw_data.get("difficulty", ""),
            "examples": self._extract_examples(raw_data.get("content", "")),
            "constraints": self._extract_constraints(raw_data.get("content", "")),
            "hints": raw_data.get("hints", []),
            "similar_problems": similar,
            "topics": [tag.get("name", "") for tag in raw_data.get("topicTags", [])],
            "acceptance_rate": stats.get("acRate", ""),
            "total_accepted": stats.get("totalAccepted", ""),
            "total_submissions": stats.get("totalSubmission", ""),
            "likes": raw_data.get("likes", 0),
            "dislikes": raw_data.get("dislikes", 0),
            "solution": solution_content,
            "has_solution": raw_data.get("hasSolution", False),
            "has_video_solution": raw_data.get("hasVideoSolution", False),
            "is_paid_only": raw_data.get("isPaidOnly", False)
        }
        
        return problem
    
    def _extract_examples(self, content: str) -> List[Dict]:
        """Extract example inputs/outputs from problem description."""
        examples = []
        
        # Simple heuristic: look for "Example 1:", "Example 2:", etc.
        # This is a basic parser - may need refinement based on actual API response
        
        lines = content.split('\n')
        current_example = None
        
        for line in lines:
            if 'Example' in line and ':' in line:
                if current_example:
                    examples.append(current_example)
                current_example = {"text": line, "input": "", "output": "", "explanation": ""}
            elif current_example:
                if 'Input:' in line:
                    current_example["input"] = line.split('Input:')[-1].strip()
                elif 'Output:' in line:
                    current_example["output"] = line.split('Output:')[-1].strip()
                elif 'Explanation:' in line:
                    current_example["explanation"] = line.split('Explanation:')[-1].strip()
        
        if current_example:
            examples.append(current_example)
        
        return examples
    
    def _extract_constraints(self, content: str) -> List[str]:
        """Extract constraints from problem description."""
        constraints = []
        
        # Look for "Constraints:" section
        if 'Constraints:' in content:
            constraints_section = content.split('Constraints:')[-1]
            # Extract lines that look like constraints (bullet points or numbered)
            lines = constraints_section.split('\n')
            for line in lines[:15]:  # Take first ~15 lines after Constraints
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    constraints.append(line)
        
        return constraints
    
    def enrich_curriculum(self, canonical_path: Path, output_path: Path) -> Dict:
        """
        Enrich canonical curriculum with full LeetCode problem details.
        
        Args:
            canonical_path: Path to canonical_curriculum.json
            output_path: Path to write enriched curriculum
        
        Returns:
            Statistics about enrichment process
        """
        print("="*70)
        print("ENRICHING CURRICULUM WITH LEETCODE PROBLEM DETAILS")
        print("="*70)
        
        # Load canonical curriculum
        with open(canonical_path) as f:
            curriculum = json.load(f)
        
        self.stats["total"] = sum(
            len(topic.get("problems", [])) 
            for topic in curriculum.get("topics", [])
        )
        
        print(f"\n📊 Found {self.stats['total']} problems across {len(curriculum['topics'])} topics")
        print(f"⏱️  Rate limit: {self.rate_limit}s between requests")
        print(f"⏳ Estimated time: ~{int(self.stats['total'] * self.rate_limit / 60)} minutes\n")
        
        # Enrich each topic's problems
        for topic_idx, topic in enumerate(curriculum["topics"], 1):
            topic_id = topic["id"]
            problems = topic.get("problems", [])
            
            print(f"\n[{topic_idx}/{len(curriculum['topics'])}] 📖 {topic['name']}")
            print(f"   Processing {len(problems)} problems...")
            
            enriched_problems = []
            
            for prob_idx, problem in enumerate(problems, 1):
                problem_id = problem["id"]
                
                # Check if already enriched
                if "description" in problem and problem["description"]:
                    print(f"   [{prob_idx}/{len(problems)}] ⏭️  {problem_id} already enriched")
                    enriched_problems.append(problem)
                    self.stats["skipped"] += 1
                    continue
                
                print(f"   [{prob_idx}/{len(problems)}] 🔄 Fetching {problem_id}...", end=" ")
                
                # Fetch full details
                details = self.fetch_problem(problem_id)
                
                if details and details.get("is_premium"):
                    # Mark as premium and skip
                    problem["is_premium"] = True
                    enriched_problems.append(problem)
                    self.stats["premium"] += 1
                    print("🔒 PREMIUM (skipped)")
                elif details:
                    # Merge with existing data
                    enriched = {**problem, **details}
                    enriched_problems.append(enriched)
                    self.stats["success"] += 1
                    print(f"✅ ({len(details.get('examples', []))} examples, {len(details.get('hints', []))} hints)")
                else:
                    # Keep original data even if fetch failed
                    enriched_problems.append(problem)
                    self.stats["failed"] += 1
                    print("❌")
            
            topic["problems"] = enriched_problems
        
        # Update metadata
        curriculum["version"] = "2.1.0"  # Increment version
        curriculum["enrichment_status"] = {
            "total_problems": self.stats["total"],
            "enriched": self.stats["success"],
            "failed": self.stats["failed"],
            "skipped": self.stats["skipped"],
            "completion_rate": f"{100 * self.stats['success'] / max(1, self.stats['total']):.1f}%"
        }
        
        # Write enriched curriculum
        with open(output_path, 'w') as f:
            json.dump(curriculum, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ ENRICHMENT COMPLETE")
        print(f"{'='*70}")
        print(f"\n📊 Statistics:")
        print(f"   Total problems: {self.stats['total']}")
        print(f"   ✅ Successfully enriched: {self.stats['success']}")
        print(f"   🔒 Premium (skipped): {self.stats['premium']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⏭️  Skipped (already enriched): {self.stats['skipped']}")
        free_problems = self.stats['total'] - self.stats['premium']
        print(f"   📈 Completion rate: {100 * self.stats['success'] / max(1, free_problems):.1f}% (of free problems)")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Enrich curriculum with LeetCode problem details"
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=1.0,
        help='Seconds to wait between API requests (default: 1.0)'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('curricula/cp_accelerator/canonical_curriculum.json'),
        help='Input canonical curriculum file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('curricula/cp_accelerator/canonical_curriculum.json'),
        help='Output enriched curriculum file (overwrites input by default)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"❌ ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create enricher
    enricher = LeetCodeEnricher(rate_limit_seconds=args.rate_limit)
    
    # Enrich curriculum
    stats = enricher.enrich_curriculum(args.input, args.output)
    
    print(f"\n✅ Enriched curriculum written to: {args.output}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review enriched problems in {args.output}")
    print(f"   2. Run: python scripts/generate_manifest.py")
    print(f"   3. Generate module content: python scripts/generate_modules.py")


if __name__ == '__main__':
    main()
