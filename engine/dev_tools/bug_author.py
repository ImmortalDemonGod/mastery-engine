"""
LLM-Powered Bug Authoring Tool

Automates the creation of JSON bug definitions from legacy .patch files
using few-shot learning with the golden dataset.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Optional

from engine.ast_harden.generic_injector import GenericBugInjector
from engine.services.llm_service import LLMService


class BugAuthor:
    """
    Generates JSON bug definitions from patch files using LLM with validation loop.
    """
    
    # Golden dataset for few-shot prompting
    GOLDEN_EXAMPLES = [
        {
            "name": "softmax-no-subtract-max",
            "path": "curricula/cs336_a1/modules/softmax/bugs/no_subtract_max.json",
            "description": "Multi-pass with context tracking",
            "complexity": "complex"
        },
        {
            "name": "silu-missing-multiply",
            "path": "curricula/cs336_a1/modules/silu/bugs/missing_multiply.json",
            "description": "Single-pass node replacement",
            "complexity": "simple"
        },
        {
            "name": "rmsnorm-missing-keepdim",
            "path": "curricula/cs336_a1/modules/rmsnorm/bugs/missing_keepdim.json",
            "description": "Keyword argument manipulation",
            "complexity": "medium"
        },
        {
            "name": "attention-missing-scale",
            "path": "curricula/cs336_a1/modules/attention/bugs/missing_scale.json",
            "description": "Statement deletion (using delete_statement)",
            "complexity": "simple"
        }
    ]
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize bug author with LLM service."""
        self.llm_service = llm_service or LLMService()
        self.golden_dataset = self._load_golden_dataset()
    
    def _load_golden_dataset(self) -> list[dict]:
        """Load golden dataset examples from disk."""
        examples = []
        
        # Resolve paths relative to project root
        # Assume this file is at engine/dev_tools/bug_author.py
        project_root = Path(__file__).parent.parent.parent
        
        for example_meta in self.GOLDEN_EXAMPLES:
            # Make path absolute relative to project root
            path = project_root / example_meta["path"]
            
            if path.exists():
                with open(path, 'r') as f:
                    bug_def = json.load(f)
                examples.append({
                    "meta": example_meta,
                    "definition": bug_def
                })
            else:
                # Log warning but continue
                print(f"⚠️  Warning: Golden example not found: {path}")
        
        return examples
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with schema and golden examples."""
        prompt = """You are an expert in Python AST manipulation and curriculum design. Your task is to generate JSON bug definitions following the v2.1 schema that can be interpreted by a generic AST transformation engine.

# JSON Schema v2.1

```json
{
  "id": "module-bug-type",
  "description": "Human-readable description of what the bug does",
  "injection_type": "ast",
  "engine_version": "2.1",
  "target_function": "function_name",
  "logic": [
    {
      "pass": 1,
      "type": "find_and_track | find_and_replace",
      "description": "What this pass does",
      "pattern": {
        "node_type": "Assign | BinOp | Call | Attribute | Name | ...",
        "attr": "attribute_name",
        "op": "Sub | Mult | Add | ...",
        "value": { /* nested pattern */ },
        "left": { /* nested pattern */ },
        "right": { /* nested pattern */ }
      },
      "conditions": [
        {
          "check": "targets_length_equals | target_is_name | has_keyword_arg",
          "value": 1,
          "name": "arg_name"
        }
      ],
      "track_as": {
        "context_var_name": "pattern.path.to.extract"
      },
      "replacement": {
        "type": "replace_value_with | replace_with | remove_keyword_arg",
        "source": "node.path.to.replacement",
        "name": "arg_name"
      }
    }
  ],
  "metadata": {
    "created": "YYYY-MM-DD",
    "version": "2.0",
    "author": "LLM-Generated",
    "tier": "simple | medium | complex"
  }
}
```

# Pattern Matching Rules

1. **node_type**: Must match AST node class name (Assign, BinOp, Call, Attribute, Name, etc.)
2. **Operators**: Use class names (Sub, Mult, Add, Div, etc.)
3. **Context references**: Use {"from_context": "var_name"} to reference tracked variables
4. **Paths**: Use dot notation (e.g., "node.value.left", "pattern.targets[0].id")

# Transformation Types

1. **replace_value_with**: Replace the value of an Assign node
2. **replace_with**: Replace entire node with another part
3. **remove_keyword_arg**: Remove a keyword argument from Call node
4. **delete_statement**: Delete the matched statement entirely (use for removing assignments, expressions, etc.)

# Multi-Pass Strategy

- Use **find_and_track** when you need to remember variable names for later passes
- Use **find_and_replace** to perform the actual transformation
- Context variables from track passes are available in replace passes via "from_context"

"""
        
        # Add golden examples
        prompt += "\n# Golden Dataset Examples\n\n"
        for i, example in enumerate(self.golden_dataset, 1):
            meta = example["meta"]
            definition = example["definition"]
            prompt += f"## Example {i}: {meta['name']} ({meta['complexity']})\n"
            prompt += f"{meta['description']}\n\n"
            prompt += "```json\n"
            prompt += json.dumps(definition, indent=2)
            prompt += "\n```\n\n"
        
        return prompt
    
    def _extract_patch_info(self, patch_path: Path) -> dict:
        """Extract before/after code from patch file."""
        patch_content = patch_path.read_text()
        
        # Simple patch parser (assumes unified diff format)
        lines = patch_content.split('\n')
        before_lines = []
        after_lines = []
        in_hunk = False
        
        for line in lines:
            if line.startswith('@@'):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            
            if line.startswith('-') and not line.startswith('---'):
                before_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                after_lines.append(line[1:])
            elif line.startswith(' '):
                before_lines.append(line[1:])
                after_lines.append(line[1:])
        
        return {
            "before": '\n'.join(before_lines),
            "after": '\n'.join(after_lines)
        }
    
    def _build_user_prompt(self, module_name: str, patch_info: dict, symptom: str) -> str:
        """Build user prompt with specific bug context."""
        prompt = f"""# Task: Generate Bug Definition for '{module_name}'

## Bug Description
{symptom}

## Code Transformation

**BEFORE (Correct):**
```python
{patch_info['before']}
```

**AFTER (Buggy):**
```python
{patch_info['after']}
```

## Your Task

Analyze the transformation from BEFORE to AFTER and generate a JSON bug definition that would produce this exact transformation when applied to ANY correct implementation (regardless of variable names).

**Critical Requirements:**
1. The pattern must match the semantic structure, not specific variable names
2. Use context tracking if you need to reference variables across multiple operations
3. Choose the appropriate pass type (find_and_track vs find_and_replace)
4. Validate that your pattern would match the BEFORE code's AST structure

**Output Format:**
Return ONLY valid JSON matching the v2.1 schema. No markdown, no explanations, just the JSON object.
"""
        return prompt
    
    def generate_bug_definition(
        self,
        module_name: str,
        patch_path: Path,
        symptom: str,
        max_retries: int = 3,
        debug: bool = True
    ) -> tuple[Optional[dict], bool]:
        """
        Generate bug definition with validation loop and comprehensive diagnostics.
        
        Returns:
            (bug_definition, success)
        """
        patch_info = self._extract_patch_info(patch_path)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(module_name, patch_info, symptom)
        
        for attempt in range(max_retries):
            print(f"\n🤖 LLM Attempt {attempt + 1}/{max_retries}...")
            
            # Generate JSON with Structured Outputs
            try:
                from engine.schemas.bug_definition import BugDefinition as BugDefSchema
                response = self.llm_service.generate_completion(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.3,
                    response_format=BugDefSchema
                )
            except ImportError:
                # Fall back to unstructured if schema not available
                response = self.llm_service.generate_completion(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.3
                )
            
            if debug:
                print(f"\n📄 LLM Response Preview:")
                print(response[:500] + "..." if len(response) > 500 else response)
            
            # Parse JSON
            try:
                bug_def = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"  ❌ Invalid JSON: {e}")
                if debug:
                    print(f"\n🔍 JSON Error Location:")
                    lines = response.split('\n')
                    error_line = e.lineno - 1 if hasattr(e, 'lineno') else 0
                    for i in range(max(0, error_line - 2), min(len(lines), error_line + 3)):
                        marker = "  >>>" if i == error_line else "     "
                        print(f"{marker} {i+1}: {lines[i]}")
                user_prompt += f"\n\n**Error in attempt {attempt + 1}:** Invalid JSON - {e}\nPlease try again with valid JSON."
                continue
            
            # Validate against schema
            if not self._validate_schema(bug_def):
                print(f"  ❌ Schema validation failed")
                if debug:
                    print(f"\n🔍 Schema Validation Issues:")
                    required = ["id", "description", "injection_type", "engine_version", "target_function", "logic"]
                    for field in required:
                        has_field = field in bug_def
                        print(f"  {'✅' if has_field else '❌'} {field}: {'present' if has_field else 'MISSING'}")
                user_prompt += f"\n\n**Error in attempt {attempt + 1}:** Schema validation failed. Ensure all required fields are present."
                continue
            
            if debug:
                print(f"\n📊 Generated Bug Definition:")
                print(f"  ID: {bug_def['id']}")
                print(f"  Target: {bug_def['target_function']}")
                print(f"  Passes: {len(bug_def['logic'])}")
                for i, pass_def in enumerate(bug_def['logic'], 1):
                    print(f"    Pass {i}: {pass_def['type']}")
                    if 'replacement' in pass_def:
                        print(f"      Replacement: {pass_def['replacement'].get('type', 'N/A')}")
            
            # Test with generic injector
            success, diagnostic = self._test_bug_definition_with_diagnostics(
                bug_def, 
                patch_info['before'], 
                patch_info['after'],
                debug=debug
            )
            
            if success:
                print(f"  ✅ Validation passed!")
                return bug_def, True
            else:
                print(f"  ❌ Injection test failed")
                if debug and diagnostic:
                    print(f"\n🔍 Injection Test Diagnostic:\n{diagnostic}")
                user_prompt += f"\n\n**Error in attempt {attempt + 1}:** {diagnostic}\nReview the AST structure and try again."
        
        print(f"\n❌ Failed after {max_retries} attempts")
        return None, False
    
    def _validate_schema(self, bug_def: dict) -> bool:
        """Validate bug definition against schema."""
        required_fields = ["id", "description", "injection_type", "engine_version", "target_function", "logic"]
        return all(field in bug_def for field in required_fields)
    
    def _test_bug_definition(self, bug_def: dict, correct_code: str, expected_buggy: str) -> bool:
        """Test bug definition by injecting and comparing."""
        success, _ = self._test_bug_definition_with_diagnostics(bug_def, correct_code, expected_buggy, debug=False)
        return success
    
    def _test_bug_definition_with_diagnostics(
        self, 
        bug_def: dict, 
        correct_code: str, 
        expected_buggy: str,
        debug: bool = True
    ) -> tuple[bool, str]:
        """
        Test bug definition with comprehensive diagnostics.
        
        Returns:
            (success, diagnostic_message)
        """
        diagnostic = []
        
        try:
            injector = GenericBugInjector(bug_def)
            buggy_code, success = injector.inject(correct_code)
            
            if not success:
                diagnostic.append("❌ Pattern matching failed - The pattern you specified was not found in the code")
                
                # Show what patterns were attempted
                diagnostic.append("\n🔍 Patterns you tried to match:")
                for i, pass_def in enumerate(bug_def['logic'], 1):
                    if 'pattern' in pass_def:
                        pattern = pass_def['pattern']
                        diagnostic.append(f"\n  Pass {i} ({pass_def['type']}):")
                        diagnostic.append(f"    Looking for: {pattern.get('node_type', 'N/A')}")
                        if 'targets' in pattern:
                            diagnostic.append(f"    Target variable: {pattern['targets']}")
                        if 'value' in pattern:
                            value_type = pattern['value'].get('node_type', 'N/A') if isinstance(pattern['value'], dict) else 'N/A'
                            diagnostic.append(f"    Value type: {value_type}")
                
                # Show actual AST structure
                diagnostic.append("\n📊 Actual AST structure of the BEFORE code:")
                try:
                    import ast as ast_module
                    # Try to parse as-is, or wrap in a function if it's a snippet
                    try:
                        tree = ast_module.parse(correct_code)
                    except SyntaxError:
                        # Likely a code snippet - wrap in function
                        wrapped = f"def dummy():\n" + "\n".join(f"    {line}" for line in correct_code.split("\n"))
                        tree = ast_module.parse(wrapped)
                    
                    # Show first few Assign statements with their structure IN JSON PATTERN FORMAT
                    statements = []
                    for node in ast_module.walk(tree):
                        if isinstance(node, ast_module.Assign):
                            targets = [t.id for t in node.targets if isinstance(t, ast_module.Name)]
                            value_type = type(node.value).__name__
                            
                            # Build the exact JSON pattern the LLM should use
                            if targets:
                                target_pattern = {"node_type": "Name", "id": targets[0]}
                                
                                # Show more detail about the value
                                if isinstance(node.value, ast_module.BinOp):
                                    op = type(node.value.op).__name__
                                    value_pattern = {"node_type": "BinOp", "op": op}
                                elif isinstance(node.value, ast_module.Call):
                                    value_pattern = {"node_type": "Call"}
                                else:
                                    value_pattern = {"node_type": value_type}
                                
                                # Show in JSON format the LLM should use
                                pattern_json = {
                                    "node_type": "Assign",
                                    "targets": [target_pattern],
                                    "value": value_pattern
                                }
                                statements.append(f"    Variable '{targets[0]}':")
                                statements.append(f"      Pattern: {json.dumps(pattern_json, indent=8)}")
                                
                            if len(statements) >= 10:  # More lines now due to JSON format
                                break
                    if statements:
                        diagnostic.append("\n  To match these statements, use these patterns:")
                        diagnostic.extend(statements)
                    else:
                        diagnostic.append("  No Assign statements found in code")
                except Exception as e:
                    diagnostic.append(f"  (Could not parse: {e})")
                    diagnostic.append(f"\n  Raw code (first 200 chars):")
                    diagnostic.append(f"  {correct_code[:200]}")
                
                diagnostic.append("\n💡 Hint: Check that your pattern's node_type and structure match the actual code AST")
                
                return False, "\n".join(diagnostic)
            
            # Normalize whitespace for comparison
            buggy_normalized = self._normalize_code(buggy_code)
            expected_normalized = self._normalize_code(expected_buggy)
            
            if buggy_normalized == expected_normalized:
                return True, "✅ Perfect match!"
            
            # Failed - provide detailed comparison
            diagnostic.append("❌ Injection succeeded but transformation was incorrect")
            
            # Show what was expected vs what we got
            buggy_lines = buggy_normalized.split('\n')
            expected_lines = expected_normalized.split('\n')
            
            # Find all differences
            differences = []
            max_len = max(len(buggy_lines), len(expected_lines))
            for i in range(max_len):
                expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
                buggy_line = buggy_lines[i] if i < len(buggy_lines) else "<missing>"
                if expected_line != buggy_line:
                    differences.append((i+1, expected_line, buggy_line))
            
            diagnostic.append(f"\n🔍 Found {len(differences)} differences:")
            
            # Show first 3 differences
            for line_num, expected, actual in differences[:3]:
                diagnostic.append(f"\n  Line {line_num}:")
                diagnostic.append(f"    Expected: {expected[:100]}")
                diagnostic.append(f"    Got:      {actual[:100]}")
            
            if len(differences) > 3:
                diagnostic.append(f"\n  ... and {len(differences) - 3} more differences")
            
            # Specific guidance
            diagnostic.append("\n💡 Hints:")
            if len(buggy_lines) < len(expected_lines):
                diagnostic.append("  - Your transformation removed TOO MUCH (fewer lines than expected)")
                diagnostic.append("  - Check if you're deleting the right statements")
            elif len(buggy_lines) > len(expected_lines):
                diagnostic.append("  - Your transformation didn't remove ENOUGH (more lines than expected)")
                diagnostic.append("  - Check if patterns are matching the statements you want to delete")
            else:
                diagnostic.append("  - Line count matches but content differs")
                diagnostic.append("  - Check if your replacement values are correct")
            
            return False, "\n".join(diagnostic)
            
        except Exception as e:
            diagnostic.append(f"❌ Exception during injection: {type(e).__name__}: {e}")
            
            if debug:
                import traceback
                diagnostic.append("\n🐛 Full traceback:")
                diagnostic.append(traceback.format_exc())
            
            return False, "\n".join(diagnostic)
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove extra whitespace, etc.)."""
        try:
            tree = ast.parse(code)
            return ast.unparse(tree)
        except:
            # Fallback: just normalize whitespace
            return ' '.join(code.split())
