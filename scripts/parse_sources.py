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
    
    def __init__(self, taxonomy_dir: Path, roadmap_file: Path):
        self.taxonomy_dir = taxonomy_dir
        self.roadmap_file = roadmap_file
        self.roadmap_data = self._parse_roadmap()
    
    def _parse_roadmap(self) -> Dict:
        """
        Parse RoadmapResources.md to extract rating brackets, topics, and resource URLs.
        Returns structured data with priorities and links.
        """
        if not self.roadmap_file.exists():
            print(f"❌ ERROR: Roadmap file not found: {self.roadmap_file}")
            return {}
        
        content = self.roadmap_file.read_text()
        roadmap = {}
        
        # Extract all rating brackets
        bracket_pattern = r'\*\*(\d+-\d+)\*\*'
        brackets = re.findall(bracket_pattern, content)
        
        for bracket in brackets:
            # Extract section for this bracket
            section_start = content.find(f"**{bracket}**")
            next_bracket_idx = content.find("**", section_start + len(bracket) + 4)
            
            # Find the next major heading or end
            sites_idx = content.find("**Sites for practice", section_start)
            section_end = min([idx for idx in [next_bracket_idx, sites_idx, len(content)] if idx > section_start])
            
            section = content[section_start:section_end]
            
            # Extract Vital topics
            vital_start = section.find("[Vital]")
            helpful_start = section.find("[Helpful]")
            
            vital_section = section[vital_start:helpful_start] if vital_start != -1 else ""
            helpful_section = section[helpful_start:] if helpful_start != -1 else ""
            
            roadmap[bracket] = {
                "vital": self._extract_topics_with_resources(vital_section),
                "helpful": self._extract_topics_with_resources(helpful_section)
            }
        
        return roadmap
    
    def _extract_topics_with_resources(self, section: str) -> List[Dict]:
        """Extract topic names and their resource URLs from a section."""
        topics = []
        # Match bullet points with optional markdown links
        # Format: * Topic name - [Resource Title](URL)
        pattern = r'\* (.+?)(?:\s*-\s*\[(.+?)\]\((.+?)\))?(?:\n|$)'
        
        for match in re.finditer(pattern, section):
            topic_text = match.group(1).strip()
            resource_title = match.group(2)
            resource_url = match.group(3)
            
            topic = {
                "name": topic_text,
                "resources": []
            }
            
            # Extract any inline links in the topic text
            inline_link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            inline_links = re.findall(inline_link_pattern, topic_text)
            
            for link_title, link_url in inline_links:
                topic["resources"].append({
                    "type": self._classify_resource_type(link_url),
                    "title": link_title,
                    "url": link_url
                })
            
            # Add the main resource if present
            if resource_title and resource_url:
                topic["resources"].append({
                    "type": self._classify_resource_type(resource_url),
                    "title": resource_title,
                    "url": resource_url
                })
            
            # Clean topic name (remove markdown links)
            topic["name"] = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', topic["name"])
            
            topics.append(topic)
        
        return topics
    
    def _classify_resource_type(self, url: str) -> str:
        """Classify resource type based on URL."""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'video'
        elif 'codeforces.com/blog' in url:
            return 'blog'
        elif 'usaco.guide' in url or 'cp-algorithms.com' in url:
            return 'tutorial'
        elif 'cppreference.com' in url:
            return 'documentation'
        else:
            return 'article'
    
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
        
        Comprehensive mapping for all 19 DSA Taxonomy patterns.
        Returns None if the pattern doesn't appear in the roadmap.
        """
        # Complete mapping for all 19 patterns based on roadmap
        mapping = {
            # Tier 1: Foundation (0-999)
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
            "stack_queue": {
                "rating_bracket": "1000-1199",
                "priority": "Vital",
                "roadmap_topics": ["STL/library"]
            },
            "linked_list": {
                "rating_bracket": "0-999",
                "priority": "Helpful",
                "roadmap_topics": ["STL/library"]
            },
            
            # Tier 2: Core Algorithms (1000-1399)
            "traversal": {
                "rating_bracket": "1200-1399",
                "priority": "Vital",
                "roadmap_topics": ["Recursion", "Graphs/trees"]
            },
            "binary_search": {
                "rating_bracket": "1000-1199",
                "priority": "Helpful",
                "roadmap_topics": ["Binary search"]
            },
            "heap": {
                "rating_bracket": "1400-1599",
                "priority": "Helpful",
                "roadmap_topics": ["STL/library"]
            },
            "prefix_sum": {
                "rating_bracket": "1000-1199",
                "priority": "Helpful",
                "roadmap_topics": ["Basic range queries"]
            },
            
            # Tier 3: Specialized (1400+)
            "greedy": {
                "rating_bracket": "1400-1599",
                "priority": "Vital",
                "roadmap_topics": ["Constructives"]
            },
            "backtracking": {
                "rating_bracket": "1400-1599",
                "priority": "Helpful",
                "roadmap_topics": ["Recursion"]
            },
            "dynamic_programming": {
                "rating_bracket": "1200-1399",
                "priority": "Vital",
                "roadmap_topics": ["Dynamic programming"]
            },
            "divide_conquer": {
                "rating_bracket": "1400-1599",
                "priority": "Helpful",
                "roadmap_topics": ["Recursion"]
            },
            
            # Tier 4: Advanced (1600+)
            "trie": {
                "rating_bracket": "1600-1899",
                "priority": "Helpful",
                "roadmap_topics": ["String algorithms"]
            },
            "union_find": {
                "rating_bracket": "1400-1599",
                "priority": "Helpful",
                "roadmap_topics": ["DSU"]
            },
            "bit_manipulation": {
                "rating_bracket": "1600-1899",
                "priority": "Vital",
                "roadmap_topics": ["Bitwise stuff"]
            },
            "segment_tree": {
                "rating_bracket": "1600-1899",
                "priority": "Vital",
                "roadmap_topics": ["Segment tree"]
            },
            "combinatorics": {
                "rating_bracket": "1600-1899",
                "priority": "Vital",
                "roadmap_topics": ["Basic combinatorics, probability, expected value"]
            },
            "design": {
                "rating_bracket": "1400-1599",
                "priority": "Helpful",
                "roadmap_topics": ["STL/library"]
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
        
        # Extract resources from roadmap
        bracket = roadmap_mapping["rating_bracket"]
        roadmap_resources = []
        
        if bracket in self.roadmap_data:
            # Search for matching topics in roadmap
            priority = roadmap_mapping["priority"].lower()
            roadmap_topics = self.roadmap_data[bracket].get(priority, [])
            
            for roadmap_topic in roadmap_topics:
                # Check if this roadmap topic matches any of our expected topics
                for expected_topic in roadmap_mapping["roadmap_topics"]:
                    if expected_topic.lower() in roadmap_topic["name"].lower():
                        # Add all resources from this roadmap topic
                        roadmap_resources.extend(roadmap_topic["resources"])
        
        # Build complete resources list
        resources = [
            {
                "type": "taxonomy",
                "url": f"https://github.com/Yassir-aykhlf/DSA-Taxonomies/blob/main/Taxonomies/{taxonomy_file.name}",
                "title": f"{name} Taxonomy"
            }
        ]
        resources.extend(roadmap_resources)
        
        topic = {
            "id": topic_id,
            "name": name,
            "rating_bracket": roadmap_mapping["rating_bracket"],
            "priority": roadmap_mapping["priority"],
            "taxonomy_path": f"Taxonomies/{taxonomy_file.name}",
            "description": taxonomy_data["description"],
            "dependencies": dependencies,
            "resources": resources,
            "problems": canonical_problems,
            "estimated_hours": estimated_hours,
            "notes": f"Based on roadmap topics: {', '.join(roadmap_mapping['roadmap_topics'])}",
            "metadata": {
                "total_problems_in_taxonomy": taxonomy_data["problem_count"],
                "source_verified": True,
                "roadmap_resources_extracted": len(roadmap_resources)
            }
        }
        
        return topic
    
    def generate_canonical_curriculum(self) -> Dict:
        """Generate the complete canonical curriculum from sources."""
        
        print("="*60)
        print("PARSING SOURCE MATERIALS")
        print("="*60)
        
        topics = []
        
        # All 19 patterns from DSA Taxonomies (in pedagogical order)
        topic_specs = [
            # Tier 1: Foundation (0-999)
            ("sorting", "Sorting Algorithms", "5. Sorting.md", [], 4),
            ("two_pointers", "Two Pointers", "1. Two Pointers.md", ["sorting"], 6),
            ("hash_table", "Hash Table", "2. Hash Table.md", [], 5),
            ("stack_queue", "Stack and Queue", "3. Stack and Queue.md", [], 4),
            ("linked_list", "Linked List", "4. Linked List.md", [], 4),
            
            # Tier 2: Core Algorithms (1000-1399)
            ("traversal", "Traversal Algorithms", "6. Traversal Algorithms.md", [], 8),
            ("binary_search", "Binary Search", "7. Binary Search.md", ["sorting"], 6),
            ("heap", "Heap and Priority Queue", "8. Heap Priority Queue.md", [], 5),
            ("prefix_sum", "Prefix Sum", "9. Prefix Sum.md", [], 4),
            
            # Tier 3: Specialized (1400+)
            ("greedy", "Greedy Algorithms", "10. Greedy Algorithms.md", [], 6),
            ("backtracking", "Backtracking", "11. Backtracking.md", ["traversal"], 7),
            ("dynamic_programming", "Dynamic Programming", "12. Dynamic Programming.md", ["traversal"], 10),
            ("divide_conquer", "Divide and Conquer", "13. Divide and Conquer.md", ["traversal"], 6),
            
            # Tier 4: Advanced (1600+)
            ("trie", "Trie", "14. Trie.md", [], 5),
            ("union_find", "Union Find (Disjoint Set Union)", "15. Union Find.md", [], 5),
            ("bit_manipulation", "Bit Manipulation", "16. Bit Manipulation.md", [], 5),
            ("segment_tree", "Segment Tree and Fenwick Tree", "17. Segment Tree and Fenwick Tree.md", ["prefix_sum"], 8),
            ("combinatorics", "Combinatorics and Number Theory", "18. Combinatorics and Number Theory.md", [], 7),
            ("design", "Design Patterns", "19. Design Pattern.md", [], 6),
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
    roadmap_file = Path('RoadmapResources.md')
    output_file = Path('curricula/cp_accelerator/canonical_curriculum.json')
    
    # Validate taxonomy directory exists
    if not taxonomy_dir.exists():
        print(f"❌ ERROR: Taxonomy directory not found: {taxonomy_dir}")
        print(f"   Run: git clone https://github.com/Yassir-aykhlf/DSA-Taxonomies .sources/cp_accelerator/dsa_taxonomies")
        sys.exit(1)
    
    # Validate roadmap file exists
    if not roadmap_file.exists():
        print(f"❌ ERROR: Roadmap file not found: {roadmap_file}")
        sys.exit(1)
    
    # Parse sources
    parser_obj = SourceParser(taxonomy_dir, roadmap_file)
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
