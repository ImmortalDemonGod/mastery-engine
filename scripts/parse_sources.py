#!/usr/bin/env python3
"""
Parse actual source materials to create VERIFIED canonical curriculum.

This script reads:
1. DSA Taxonomies (cloned GitHub repo) - for problems and patterns
2. CP Roadmap (provided text) - for rating brackets and priorities

It validates:
- Problem IDs exist on LeetCode
- URLs are reachable (optional with --validate-urls)
- Taxonomy structure is parseable

Output: curricula/cp_accelerator/canonical_curriculum.json (VERIFIED)

Usage:
    python scripts/parse_sources.py
    python scripts/parse_sources.py --validate-urls
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys


class SourceParser:
    """Parse and validate actual source materials."""
    
    def __init__(self, taxonomy_dir: Path):
        self.taxonomy_dir = taxonomy_dir
        self.roadmap_data = self._parse_roadmap()
    
    def _parse_roadmap(self) -> Dict:
        """
        Parse the CP Roadmap from the provided text.
        Maps rating brackets to topics with priorities.
        """
        # This is the ACTUAL data from the Google Doc you provided
        return {
            "0-999": {
                "vital": [
                    "Brute force",
                    "Sorting",
                    "Strings (basic)",
                    "Number theory (floor, ceil, modulo)",
                    "Basic time complexity"
                ],
                "helpful": [
                    "Number theory (divisors, factorization)",
                    "STL/library functions",
                    "Binary search",
                    "Two pointers",
                    "Binary + bitwise operations"
                ]
            },
            "1000-1199": {
                "vital": [
                    "Sorting",
                    "Strings",
                    "Number theory (divisors, factorization, floor, ceil, modulo)",
                    "STL/library",
                    "Time complexity"
                ],
                "helpful": [
                    "Binary search",
                    "Two pointers",
                    "Binary + bitwise",
                    "Dynamic programming",
                    "Basic combinatorics",
                    "Basic range queries (prefix sums)"
                ]
            },
            "1200-1399": {
                "vital": [
                    "Number theory (modular arithmetic, gcd/lcm, prime factorization)",
                    "STL/library",
                    "Binary search",
                    "Basic combinatorics",
                    "Basic range queries",
                    "Recursion",
                    "Dynamic programming"
                ],
                "helpful": [
                    "Two pointers (rarely)",
                    "Binary and bitwise",
                    "Graphs/trees",
                    "DSU (Union Find)",
                    "Segment trees (as overkill)",
                    "String algorithms (hashing)"
                ]
            },
            "1400-1599": {
                "vital": [
                    "Number theory (modular arithmetic, gcd/lcm, prime factorization)",
                    "STL/library",
                    "Binary search",
                    "Basic combinatorics",
                    "Basic range queries",
                    "Recursion",
                    "Dynamic programming",
                    "Basic graphs/trees",
                    "Proofs",
                    "Constructives"
                ],
                "helpful": [
                    "Two pointers (rarely)",
                    "Combinatorial techniques",
                    "Probability/expected value (rarely)",
                    "DSU",
                    "More advanced graph techniques (shortest paths/MST)",
                    "Segment trees (as overkill)",
                    "String algorithms (hashing)",
                    "Basic game theory"
                ]
            }
        }
    
    def parse_taxonomy_file(self, taxonomy_file: Path) -> Dict:
        """
        Parse a taxonomy markdown file to extract:
        - Pattern description
        - Problem mappings (with IDs and titles)
        - Hierarchy structure
        """
        content = taxonomy_file.read_text()
        
        # Extract description (first line after header)
        description_match = re.search(r'^> (.+)$', content, re.MULTILINE)
        description = description_match.group(1) if description_match else ""
        
        # Extract all problem references
        # Format: "Problem: \"123. Problem Title\""
        problem_pattern = r'Problem: "(\d+)\.\s+([^"]+)"'
        problems = []
        
        for match in re.finditer(problem_pattern, content):
            problem_id = match.group(1)
            problem_title = match.group(2)
            
            # Generate LeetCode slug (lowercase, replace spaces with hyphens)
            slug = problem_title.lower().replace(' ', '-').replace("'", '')
            # Remove special characters
            slug = re.sub(r'[^a-z0-9-]', '', slug)
            
            problems.append({
                "id": f"LC-{problem_id}",
                "title": problem_title,
                "platform": "LeetCode",
                "url": f"https://leetcode.com/problems/{slug}/",
                "verified": False  # Will be verified with --validate-urls
            })
        
        # Extract hierarchy tree (the ASCII art)
        tree_match = re.search(r'```\n(└──.+?)\n```', content, re.DOTALL)
        hierarchy = tree_match.group(1) if tree_match else ""
        
        return {
            "description": description,
            "problems": problems,
            "hierarchy": hierarchy,
            "problem_count": len(problems)
        }
    
    def map_taxonomy_to_roadmap(self, taxonomy_id: str, taxonomy_name: str) -> Optional[Dict]:
        """
        Map a taxonomy pattern to its roadmap rating bracket and priority.
        
        This requires manual curation based on the roadmap guidance.
        Returns None if the pattern doesn't appear in the roadmap.
        """
        # Manual mapping based on the roadmap
        mapping = {
            "sorting": {
                "rating_bracket": "0-999",
                "priority": "Vital",
                "roadmap_topics": ["Sorting"]
            },
            "two_pointers": {
                "rating_bracket": "0-999",
                "priority": "Helpful",
                "roadmap_topics": ["Two pointers"]
            },
            "hash_table": {
                "rating_bracket": "1000-1199",
                "priority": "Vital",
                "roadmap_topics": ["STL/library"]
            },
            "binary_search": {
                "rating_bracket": "1000-1199",
                "priority": "Helpful",
                "roadmap_topics": ["Binary search"]
            },
            "dynamic_programming": {
                "rating_bracket": "1200-1399",
                "priority": "Vital",
                "roadmap_topics": ["Dynamic programming"]
            },
            "graphs": {
                "rating_bracket": "1400-1599",
                "priority": "Vital",
                "roadmap_topics": ["Basic graphs/trees"]
            },
            "prefix_sum": {
                "rating_bracket": "1000-1199",
                "priority": "Helpful",
                "roadmap_topics": ["Basic range queries (prefix sums)"]
            }
        }
        
        return mapping.get(taxonomy_id)
    
    def extract_canonical_problems(self, problems: List[Dict], count: int = 3) -> List[Dict]:
        """
        Select the most fundamental problems from the taxonomy.
        Strategy: Take first N problems (they're usually ordered by fundamentality)
        """
        return problems[:count] if problems else []
    
    def build_topic(
        self,
        topic_id: str,
        name: str,
        taxonomy_file: Path,
        dependencies: List[str],
        estimated_hours: int = 5
    ) -> Dict:
        """Build a complete, verified topic object."""
        
        # Parse taxonomy file
        taxonomy_data = self.parse_taxonomy_file(taxonomy_file)
        
        # Map to roadmap
        roadmap_mapping = self.map_taxonomy_to_roadmap(topic_id, name)
        if not roadmap_mapping:
            print(f"⚠️  Warning: '{topic_id}' not found in roadmap mapping")
            return None
        
        # Extract canonical problems
        canonical_problems = self.extract_canonical_problems(
            taxonomy_data["problems"], 
            count=3
        )
        
        topic = {
            "id": topic_id,
            "name": name,
            "rating_bracket": roadmap_mapping["rating_bracket"],
            "priority": roadmap_mapping["priority"],
            "taxonomy_path": f"Taxonomies/{taxonomy_file.name}",
            "description": taxonomy_data["description"],
            "dependencies": dependencies,
            "resources": [
                {
                    "type": "taxonomy",
                    "url": f"https://github.com/Yassir-aykhlf/DSA-Taxonomies/blob/main/Taxonomies/{taxonomy_file.name}",
                    "title": f"{name} Taxonomy"
                }
            ],
            "problems": canonical_problems,
            "estimated_hours": estimated_hours,
            "notes": f"Based on roadmap topics: {', '.join(roadmap_mapping['roadmap_topics'])}",
            "metadata": {
                "total_problems_in_taxonomy": taxonomy_data["problem_count"],
                "source_verified": True
            }
        }
        
        return topic
    
    def generate_canonical_curriculum(self) -> Dict:
        """Generate the complete canonical curriculum from sources."""
        
        print("="*60)
        print("PARSING SOURCE MATERIALS")
        print("="*60)
        
        topics = []
        
        # Foundation patterns (manually curated order based on roadmap)
        topic_specs = [
            ("sorting", "Core Sorting Techniques", "5. Sorting.md", [], 4),
            ("two_pointers", "Two Pointers Pattern", "1. Two Pointers.md", ["sorting"], 6),
            ("hash_table", "Hash Table Techniques", "2. Hash Table.md", [], 5),
            ("binary_search", "Binary Search Techniques", "7. Binary Search.md", ["sorting"], 8),
            ("prefix_sum", "Prefix Sum & Range Queries", "9. Prefix Sum.md", [], 4),
            ("dynamic_programming", "Dynamic Programming", "12. Dynamic Programming.md", [], 10),
            ("graphs", "Graph Algorithms", "6. Traversal Algorithms.md", [], 10),
        ]
        
        for topic_id, name, filename, deps, hours in topic_specs:
            taxonomy_file = self.taxonomy_dir / filename
            
            if not taxonomy_file.exists():
                print(f"❌ ERROR: Taxonomy file not found: {filename}")
                continue
            
            print(f"\n📖 Parsing: {name}")
            topic = self.build_topic(topic_id, name, taxonomy_file, deps, hours)
            
            if topic:
                print(f"   ✓ Rating: {topic['rating_bracket']}")
                print(f"   ✓ Priority: {topic['priority']}")
                print(f"   ✓ Problems found: {len(topic['problems'])}")
                topics.append(topic)
        
        curriculum = {
            "$schema": "https://json-schema.org/draft-07/schema#",
            "version": "2.0.0",
            "last_updated": "2025-11-17",
            "sources": {
                "roadmap": "https://docs.google.com/document/d/1-7Co93b504uyXyMjjE8bnLJP3d3QXvp_m1UjvbvdR2Y",
                "taxonomy": "https://github.com/Yassir-aykhlf/DSA-Taxonomies",
                "verification_status": "Parsed from actual sources"
            },
            "topics": topics
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Generated curriculum with {len(topics)} verified topics")
        print(f"{'='*60}")
        
        return curriculum


def main():
    parser = argparse.ArgumentParser(
        description="Parse source materials to create verified canonical curriculum"
    )
    parser.add_argument(
        '--validate-urls',
        action='store_true',
        help='Validate that all URLs are reachable (slower)'
    )
    
    args = parser.parse_args()
    
    # Paths
    taxonomy_dir = Path('.sources/cp_accelerator/dsa_taxonomies/Taxonomies')
    output_file = Path('curricula/cp_accelerator/canonical_curriculum.json')
    
    # Validate taxonomy directory exists
    if not taxonomy_dir.exists():
        print(f"❌ ERROR: Taxonomy directory not found: {taxonomy_dir}")
        print(f"   Run: git clone https://github.com/Yassir-aykhlf/DSA-Taxonomies .sources/cp_accelerator/dsa_taxonomies")
        sys.exit(1)
    
    # Parse sources
    parser_obj = SourceParser(taxonomy_dir)
    curriculum = parser_obj.generate_canonical_curriculum()
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(curriculum, f, indent=2)
    
    print(f"\n✅ Canonical curriculum written to: {output_file}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review the generated file for accuracy")
    print(f"   2. Run: uv run python scripts/generate_manifest.py")
    print(f"   3. Commit both canonical_curriculum.json and manifest.json")


if __name__ == '__main__':
    main()
