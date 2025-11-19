#!/usr/bin/env python3
"""
Generate manifest.json from canonical_curriculum.json.

This script is the ONLY way to create/update the manifest.
Manual edits to manifest.json are FORBIDDEN and will fail CI.

Usage:
    python scripts/generate_manifest.py
    python scripts/generate_manifest.py --validate-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import deque


class ManifestGenerator:
    """
    Generates manifest.json from canonical curriculum source.
    Validates dependency graph and ensures curriculum integrity.
    """
    
    def __init__(self, canonical_path: Path, output_path: Path):
        self.canonical_path = canonical_path
        self.output_path = output_path
        self.topics = []
        self.topic_ids = set()
    
    def load_canonical(self):
        """Load and validate canonical curriculum."""
        if not self.canonical_path.exists():
            print(f"❌ ERROR: Canonical curriculum not found at {self.canonical_path}")
            sys.exit(1)
        
        with open(self.canonical_path, 'r') as f:
            data = json.load(f)
        
        self.topics = data.get('topics', [])
        self.topic_ids = {t['id'] for t in self.topics}
        
        print(f"✓ Loaded {len(self.topics)} topics from canonical curriculum")
        return data
    
    def validate_dependencies(self) -> bool:
        """
        Validate dependency graph:
        1. All dependency IDs exist
        2. No circular dependencies (topological sort)
        """
        print("\n🔍 Validating dependency graph...")
        
        # Check 1: All dependencies exist
        for topic in self.topics:
            topic_id = topic['id']
            deps = topic.get('dependencies', [])
            
            for dep in deps:
                if dep not in self.topic_ids:
                    print(f"❌ ERROR: Topic '{topic_id}' depends on non-existent topic '{dep}'")
                    return False
        
        print("  ✓ All dependency IDs exist")
        
        # Check 2: Detect cycles via topological sort (Kahn's algorithm)
        in_degree = {tid: 0 for tid in self.topic_ids}
        adj_list = {tid: [] for tid in self.topic_ids}
        
        # Build adjacency list and in-degree counts
        for topic in self.topics:
            topic_id = topic['id']
            for dep in topic.get('dependencies', []):
                adj_list[dep].append(topic_id)
                in_degree[topic_id] += 1
        
        # Topological sort
        queue = deque([tid for tid in self.topic_ids if in_degree[tid] == 0])
        sorted_count = 0
        
        while queue:
            current = queue.popleft()
            sorted_count += 1
            
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if sorted_count != len(self.topic_ids):
            print(f"❌ ERROR: Circular dependency detected!")
            print(f"   Expected {len(self.topic_ids)} topics, but only {sorted_count} are reachable")
            print(f"   Topics involved in cycle:")
            for tid, degree in in_degree.items():
                if degree > 0:
                    print(f"     - {tid} (in_degree={degree})")
            return False
        
        print("  ✓ No circular dependencies detected")
        return True
    
    def generate_manifest(self, canonical_data: dict) -> dict:
        """Generate manifest structure based on curriculum type."""
        print("\n🏗️  Generating manifest.json...")
        
        # Detect curriculum type (default to linear for backward compatibility)
        curr_type = canonical_data.get('curriculum_type', 'linear')
        print(f"  📚 Curriculum type: {curr_type.upper()}")
        
        # Base manifest structure
        manifest = {
            "curriculum_name": "cp_accelerator",
            "description": "Systematic competitive programming mastery via rating-based progression",
            "author": f"Curated from DSA-Taxonomies + CP Roadmap (v{canonical_data['version']})",
            "version": canonical_data['version'],
            "type": curr_type,  # LINEAR or LIBRARY
            "metadata": {
                "sources": canonical_data['sources'],
                "rating_system": "Codeforces",
                "target_audience": "0-1899 rating",
                "last_generated": canonical_data['last_updated']
            }
        }
        
        if curr_type == "library":
            # LIBRARY MODE: Generate patterns -> problems hierarchy
            patterns = []
            total_problems = 0
            
            for topic in self.topics:
                pattern_obj = {
                    "id": topic['id'],
                    "title": topic['name'],
                    "theory_path": f"patterns/{topic['id']}/theory",
                    "metadata": {
                        "rating_bracket": topic.get('rating_bracket', '0-999'),
                        "priority": topic.get('priority', 'Helpful'),
                        "taxonomy_source": topic.get('taxonomy_path', ''),
                        "estimated_hours": topic.get('estimated_hours', 5)
                    },
                    "problems": []
                }
                
                # Generate problem entries
                for prob in topic.get('problems', []):
                    # Sanitize problem ID (e.g., "LC-912" -> "lc_912")
                    raw_id = prob.get('id', '')
                    p_id = raw_id.lower().replace('-', '_').replace(' ', '_')
                    
                    pattern_obj["problems"].append({
                        "id": p_id,
                        "title": prob.get('title', 'Untitled Problem'),
                        "path": f"patterns/{topic['id']}/problems/{p_id}",
                        "difficulty": prob.get('difficulty', 'Medium'),
                        "metadata": {
                            "url": prob.get('url', ''),
                            "platform": prob.get('platform', 'LeetCode'),
                            "original_id": raw_id
                        }
                    })
                    total_problems += 1
                
                patterns.append(pattern_obj)
            
            manifest["patterns"] = patterns
            print(f"  ✓ Generated LIBRARY manifest with {len(patterns)} patterns and {total_problems} problems")
        
        else:
            # LINEAR MODE: Generate flat modules list (Legacy)
            modules = []
            for topic in self.topics:
                module = {
                    "id": topic['id'],
                    "name": topic['name'],
                    "path": f"modules/{topic['id']}",
                    "dependencies": topic.get('dependencies', []),
                    "metadata": {
                        "rating_bracket": topic.get('rating_bracket', '0-999'),
                        "priority": topic.get('priority', 'Helpful'),
                        "taxonomy_source": topic.get('taxonomy_path', ''),
                        "estimated_hours": topic.get('estimated_hours', 5)
                    }
                }
                modules.append(module)
            
            manifest["modules"] = modules
            print(f"  ✓ Generated LINEAR manifest with {len(modules)} modules")
        
        return manifest
    
    def write_manifest(self, manifest: dict):
        """Write manifest to file with pretty formatting."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Manifest written to {self.output_path}")
    
    def generate(self, validate_only: bool = False):
        """Main generation workflow."""
        print("="*60)
        print("MANIFEST GENERATOR")
        print("="*60)
        
        # Load canonical source
        canonical_data = self.load_canonical()
        
        # Validate dependencies
        if not self.validate_dependencies():
            print("\n❌ VALIDATION FAILED: Fix dependency errors before proceeding")
            sys.exit(1)
        
        if validate_only:
            print("\n✅ VALIDATION PASSED: Canonical curriculum is valid")
            return
        
        # Generate manifest
        manifest = self.generate_manifest(canonical_data)
        
        # Write to file
        self.write_manifest(manifest)
        
        print("\n" + "="*60)
        print("✅ GENERATION COMPLETE")
        print("="*60)
        print(f"\n📋 Next steps:")
        print(f"   1. Review the generated manifest: {self.output_path}")
        print(f"   2. Create module directories: curricula/cp_accelerator/modules/<module_id>/")
        print(f"   3. Run content ingestion: python scripts/ingest_cp_content.py")


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest.json from canonical curriculum source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate manifest
  python scripts/generate_manifest.py

  # Validate canonical source without generating
  python scripts/generate_manifest.py --validate-only

WARNING: Never edit manifest.json manually. Always update canonical_curriculum.json
and regenerate. Manual edits will cause CI failures.
        """
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate canonical source, do not generate manifest'
    )
    
    parser.add_argument(
        '--canonical',
        type=Path,
        default=Path('curricula/cp_accelerator/canonical_curriculum.json'),
        help='Path to canonical curriculum source (default: curricula/cp_accelerator/canonical_curriculum.json)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('curricula/cp_accelerator/manifest.json'),
        help='Path to output manifest (default: curricula/cp_accelerator/manifest.json)'
    )
    
    args = parser.parse_args()
    
    generator = ManifestGenerator(
        canonical_path=args.canonical,
        output_path=args.output
    )
    
    generator.generate(validate_only=args.validate_only)


if __name__ == '__main__':
    main()
