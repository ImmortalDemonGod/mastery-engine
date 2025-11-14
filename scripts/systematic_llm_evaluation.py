"""
Systematic evaluation of LLM bug authoring tool.

Collect quantitative data to understand:
1. Success rate by bug complexity
2. Improvement across attempts (does feedback help?)
3. Failure modes by category
4. Impact of golden examples
"""
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.dev_tools.bug_author import BugAuthor
from engine.services.llm_service import LLMService


@dataclass
class AttemptResult:
    """Result of a single LLM attempt"""
    attempt_num: int
    success: bool
    failure_mode: str  # "json_parse", "schema_validation", "pattern_match", "output_mismatch"
    patterns_generated: int
    had_specific_var_names: bool  # Did patterns include specific variable names like "id"?
    response_length: int
    response_text: str = ""  # Store full response for comparison
    feedback_given: str = ""  # Store feedback that was provided
    similarity_to_golden: float = 0.0  # How similar patterns are to golden examples
    pattern_complexity: Dict[str, int] = None  # Track over-specification
    missing_operators: List[int] = None  # Which passes missing "op" field
    wrong_node_types: List[str] = None  # Track if wrong AST node type used
    node_type_accuracy: float = 0.0  # Percentage of correct node types vs expected
    node_type_details: Dict[str, Dict] = None  # Per-variable accuracy details
    injection_succeeded: bool = False  # Did pattern match and transform (even if comparison failed)
    is_false_negative: bool = False  # Success but marked as failure due to comparison
    
    def __post_init__(self):
        if self.pattern_complexity is None:
            self.pattern_complexity = {}
        if self.missing_operators is None:
            self.missing_operators = []
        if self.wrong_node_types is None:
            self.wrong_node_types = []
        if self.node_type_details is None:
            self.node_type_details = {}


@dataclass
class BugEvaluationResult:
    """Result of evaluating a single bug"""
    module: str
    bug_file: str
    complexity: str  # "simple", "medium", "complex"
    num_operations: int  # How many transformations needed
    attempts: List[AttemptResult]
    final_success: bool
    time_taken: float
    expected_patterns: Dict[str, tuple] = None  # Expected (node_type, op) per variable
    
    def __post_init__(self):
        if self.expected_patterns is None:
            self.expected_patterns = {}


class SystematicEvaluator:
    """Evaluate LLM bug authoring systematically"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize evaluator.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
                   Options: gpt-4o-mini (fast/cheap), gpt-4o (smart), o1-preview (reasoning)
        """
        llm_service = LLMService(model=model)
        self.author = BugAuthor(llm_service=llm_service)
        self.results: List[BugEvaluationResult] = []
        self.model = model
        
        # Ground truth: Expected patterns for each bug
        self.expected_patterns = {
            'silu': {
                # return in_features * torch.sigmoid(in_features)
                '__return__': ('Return', None)  # Special key for return statements
            },
            'attention': {
                # Bug deletes: d_k = Q.shape[-1] and scores = scores / sqrt(d_k)
                'd_k': ('Subscript', None),
                'scores': ('BinOp', 'Div')  # SECOND occurrence (the one being deleted)
            },
            'rmsnorm': {
                '__mean_call__': ('Call', None)  # .mean() call with keepdim
            },
            'adamw': {
                'bias_correction1': ('BinOp', 'Sub'),
                'bias_correction2': ('BinOp', 'Sub'),
                'step_size': ('BinOp', 'Div'),
                'denom': ('Call', None)
            }
        }
    
    def evaluate_bug(
        self, 
        module: str,
        patch_path: Path,
        symptom: str,
        complexity: str,
        num_operations: int,
        max_attempts: int = 3
    ) -> BugEvaluationResult:
        """Evaluate LLM performance on a single bug with detailed tracking"""
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {module}")
        print(f"Complexity: {complexity} ({num_operations} operations)")
        print(f"{'='*60}")
        
        attempts = []
        start_time = time.time()
        
        # Manually track attempts instead of using generate_bug_definition
        patch_info = self.author._extract_patch_info(patch_path)
        system_prompt = self.author._build_system_prompt()
        user_prompt = self.author._build_user_prompt(module, patch_info, symptom)
        
        final_success = False
        
        for attempt_num in range(1, max_attempts + 1):
            print(f"\n🔍 Attempt {attempt_num}/{max_attempts}")
            
            # Generate WITHOUT structured outputs (schema too loose for OpenAI requirements)
            try:
                response = self.author.llm_service.generate_completion(
                    prompt=user_prompt,
                    system=system_prompt,
                    temperature=0.3
                )
            except Exception as e:
                print(f"  ❌ LLM call failed: {e}")
                attempts.append(AttemptResult(
                    attempt_num=attempt_num,
                    success=False,
                    failure_mode="llm_error",
                    patterns_generated=0,
                    had_specific_var_names=False,
                    response_length=0,
                    response_text="",
                    feedback_given="",
                    similarity_to_golden=0.0
                ))
                continue
            
            # Strip markdown code fences if present (gpt-4o adds these)
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Remove opening fence (```json or ```)
                lines = response_clean.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove closing fence
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                response_clean = '\n'.join(lines)
            
            # Parse JSON
            try:
                bug_def = json.loads(response_clean)
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parse error")
                attempts.append(AttemptResult(
                    attempt_num=attempt_num,
                    success=False,
                    failure_mode="json_parse",
                    patterns_generated=0,
                    had_specific_var_names=False,
                    response_length=len(response),
                    response_text=response,
                    feedback_given="",
                    similarity_to_golden=0.0
                ))
                # Add feedback for next attempt
                user_prompt += f"\n\n**Attempt {attempt_num} failed:** Invalid JSON. Please ensure valid JSON syntax."
                continue
            
            # Validate schema
            if not self.author._validate_schema(bug_def):
                print(f"  ❌ Schema validation failed")
                attempts.append(AttemptResult(
                    attempt_num=attempt_num,
                    success=False,
                    failure_mode="schema_validation",
                    patterns_generated=len(bug_def.get('logic', [])),
                    had_specific_var_names=False,
                    response_length=len(response),
                    response_text=response,
                    feedback_given="",
                    similarity_to_golden=0.0
                ))
                user_prompt += f"\n\n**Attempt {attempt_num} failed:** Missing required fields."
                continue
            
            # Analyze patterns
            patterns_generated = len(bug_def.get('logic', []))
            has_specific_vars = self._check_for_specific_variable_names(bug_def)
            similarity_to_golden = self._measure_pattern_similarity_to_golden(bug_def)
            pattern_complexity = self._analyze_pattern_complexity(bug_def)
            missing_ops = self._find_missing_operators(bug_def, patch_info['before'])
            wrong_types = self._detect_wrong_node_types(bug_def)
            node_accuracy, node_details = self._calculate_node_type_accuracy(bug_def, module)
            
            # Test injection
            success, diagnostic = self.author._test_bug_definition_with_diagnostics(
                bug_def, 
                patch_info['before'],
                patch_info['after'],
                debug=False
            )
            
            if success:
                print(f"  ✅ SUCCESS!")
                attempts.append(AttemptResult(
                    attempt_num=attempt_num,
                    success=True,
                    failure_mode="none",
                    patterns_generated=patterns_generated,
                    had_specific_var_names=has_specific_vars,
                    response_length=len(response),
                    response_text=response,
                    feedback_given="",
                    similarity_to_golden=similarity_to_golden,
                    pattern_complexity=pattern_complexity,
                    missing_operators=missing_ops,
                    wrong_node_types=wrong_types,
                    node_type_accuracy=node_accuracy,
                    node_type_details=node_details
                ))
                final_success = True
                break
            else:
                # Determine failure mode
                if "pattern matching failed" in diagnostic.lower():
                    failure_mode = "pattern_match"
                elif "output doesn't match" in diagnostic.lower():
                    failure_mode = "output_mismatch"
                else:
                    failure_mode = "unknown"
                
                print(f"  ❌ Failed: {failure_mode}")
                print(f"     Patterns: {patterns_generated}, Specific vars: {has_specific_vars}")
                
                # For pattern_match failures on first attempt, get detailed debug info
                if failure_mode == "pattern_match" and attempt_num == 1:
                    print(f"\n  🔍 Collecting detailed pattern matching diagnostics...")
                    from engine.ast_harden.generic_injector import GenericBugInjector
                    import io
                    import sys
                    
                    # Capture debug output
                    old_stdout = sys.stdout
                    sys.stdout = captured = io.StringIO()
                    
                    try:
                        injector = GenericBugInjector(bug_def)
                        injector.inject(patch_info['before'], debug=True)
                    except:
                        pass
                    
                    sys.stdout = old_stdout
                    debug_output = captured.getvalue()
                    
                    if debug_output:
                        print(f"\n  📋 Pattern Matcher Debug Output:")
                        # Show just the key parts
                        for line in debug_output.split('\n'):
                            if 'Pass' in line or 'Target' in line or 'Replacements' in line or '❌' in line:
                                print(f"     {line}")
                
                # Store the diagnostic as feedback
                feedback_text = f"**Attempt {attempt_num} failed:**\n{diagnostic}\n"
                
                # Detect false negatives: injection succeeded but comparison failed
                injection_succeeded = 'Injection succeeded but transformation was incorrect' in diagnostic
                is_false_negative = injection_succeeded and 'def ' in diagnostic
                
                attempts.append(AttemptResult(
                    attempt_num=attempt_num,
                    success=False,
                    failure_mode=failure_mode,
                    patterns_generated=patterns_generated,
                    had_specific_var_names=has_specific_vars,
                    response_length=len(response),
                    response_text=response,
                    feedback_given=feedback_text,
                    similarity_to_golden=similarity_to_golden,
                    pattern_complexity=pattern_complexity,
                    missing_operators=missing_ops,
                    wrong_node_types=wrong_types,
                    node_type_accuracy=node_accuracy,
                    node_type_details=node_details,
                    injection_succeeded=injection_succeeded,
                    is_false_negative=is_false_negative
                ))
                
                # Add detailed feedback for next attempt
                user_prompt += f"\n\n{feedback_text}"
        
        time_taken = time.time() - start_time
        
        result = BugEvaluationResult(
            module=module,
            bug_file=patch_path.name,
            complexity=complexity,
            num_operations=num_operations,
            attempts=attempts,
            final_success=final_success,
            time_taken=time_taken
        )
        
        self.results.append(result)
        return result
    
    def _check_for_specific_variable_names(self, bug_def: dict) -> bool:
        """Check if patterns include specific variable names (e.g., 'id' field anywhere in pattern)"""
        def has_id_field(obj):
            """Recursively check if any dict in structure has 'id' field"""
            if isinstance(obj, dict):
                if 'id' in obj and not isinstance(obj['id'], dict):
                    return True
                for value in obj.values():
                    if has_id_field(value):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if has_id_field(item):
                        return True
            return False
        
        for pass_def in bug_def.get('logic', []):
            if 'pattern' in pass_def:
                if has_id_field(pass_def['pattern']):
                    return True
        return False
    
    def _analyze_pattern_complexity(self, bug_def: dict) -> Dict[str, int]:
        """
        Analyze if patterns are over-specified with unnecessary nested fields.
        Returns count of over-specified patterns by type.
        """
        complexity = {"over_specified": 0, "simple": 0}
        
        for pass_def in bug_def.get('logic', []):
            if 'pattern' in pass_def and 'value' in pass_def['pattern']:
                value = pass_def['pattern']['value']
                if isinstance(value, dict):
                    # Check if has deep nesting (left/right/args/keywords)
                    if any(k in value for k in ['left', 'right', 'args', 'keywords']):
                        complexity["over_specified"] += 1
                    else:
                        complexity["simple"] += 1
        
        return complexity
    
    def _find_missing_operators(self, bug_def: dict, before_code: str) -> List[int]:
        """
        Identify which passes are missing operator specifications when needed.
        Returns list of pass numbers that should have 'op' but don't.
        """
        missing = []
        
        # Parse before code to find variable assignments
        try:
            import ast
            tree = ast.parse(before_code)
            # Count assignments per variable
            var_assignments = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            var_assignments[var_name] = var_assignments.get(var_name, 0) + 1
        except:
            return missing
        
        # Check each pass
        for i, pass_def in enumerate(bug_def.get('logic', []), 1):
            if 'pattern' in pass_def:
                pattern = pass_def['pattern']
                # Check if targeting a variable with multiple assignments
                if 'targets' in pattern and isinstance(pattern['targets'], list):
                    for target in pattern['targets']:
                        if isinstance(target, dict) and 'id' in target:
                            var_id = target['id']
                            if isinstance(var_id, str):  # Direct id, not from_context
                                # If this variable has multiple assignments and pattern lacks op
                                if var_assignments.get(var_id, 0) > 1:
                                    if 'value' not in pattern or 'op' not in pattern.get('value', {}):
                                        missing.append(i)
        
        return missing
    
    def _detect_wrong_node_types(self, bug_def: dict) -> List[str]:
        """
        Detect if LLM is using wrong AST node types.
        Returns list of issues found.
        """
        issues = []
        
        for i, pass_def in enumerate(bug_def.get('logic', []), 1):
            if 'pattern' in pass_def:
                pattern = pass_def['pattern']
                node_type = pattern.get('node_type')
                
                # Common mistakes
                if node_type == 'BinOp' and 'targets' in pattern:
                    issues.append(f"Pass{i}: BinOp with targets (should be Assign?)")
                
                if node_type == 'Return' and 'targets' in pattern:
                    issues.append(f"Pass{i}: Return with targets (invalid)")
                    
                # Check if trying to match operations directly instead of statements
                if node_type in ['BinOp', 'Call', 'Attribute'] and pass_def.get('type') == 'find_and_replace':
                    # Likely needs to be wrapped in Assign or Return
                    issues.append(f"Pass{i}: {node_type} at statement level (wrap in Assign/Return?)")
        
        return issues
    
    def _calculate_node_type_accuracy(self, bug_def: dict, module: str) -> tuple[float, Dict]:
        """
        Calculate node type accuracy by comparing generated patterns to expected patterns.
        Returns: (accuracy_percentage, detailed_results_dict)
        """
        if module not in self.expected_patterns:
            return 0.0, {}
        
        expected = self.expected_patterns[module]
        details = {}
        correct = 0
        total = 0
        
        for pass_def in bug_def.get('logic', []):
            if 'pattern' not in pass_def:
                continue
            
            pattern = pass_def['pattern']
            
            # Get variable name from pattern
            var_id = 'unknown'
            if 'targets' in pattern and pattern['targets']:
                target = pattern['targets'][0]
                if isinstance(target, dict):
                    var_id = target.get('id', 'unknown')
            
            # Special case for return statements
            if pattern.get('node_type') == 'Return':
                var_id = '__return__'
            
            # Get generated pattern details
            if 'value' in pattern and isinstance(pattern['value'], dict):
                gen_type = pattern['value'].get('node_type', 'N/A')
                gen_op = pattern['value'].get('op', None)
            elif pattern.get('node_type') == 'Return':
                gen_type = 'Return'
                gen_op = None
            else:
                gen_type = pattern.get('node_type', 'N/A')
                gen_op = None
            
            # Compare with expected
            if var_id in expected:
                exp_type, exp_op = expected[var_id]
                type_match = gen_type == exp_type
                op_match = exp_op is None or gen_op == exp_op
                
                is_correct = type_match and op_match
                details[var_id] = {
                    'expected': (exp_type, exp_op),
                    'generated': (gen_type, gen_op),
                    'correct': is_correct
                }
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return accuracy, details
    
    def _measure_attempt_consistency(self, attempts: List[AttemptResult]) -> float:
        """
        Measure if LLM is generating identical responses across attempts.
        Returns: 1.0 if all identical, 0.0 if all different, 0.0-1.0 for partial similarity
        """
        if len(attempts) < 2:
            return 0.0
        
        # Simple approach: check if pattern structures are identical
        responses = [a.response_text for a in attempts if a.response_text]
        if len(responses) < 2:
            return 0.0
        
        # Count how many are identical to the first
        first = responses[0]
        identical_count = sum(1 for r in responses if r == first)
        
        return identical_count / len(responses)
    
    def _measure_feedback_incorporation(self, attempts: List[AttemptResult]) -> float:
        """
        Measure if LLM incorporates feedback terms in next attempt.
        Returns: average incorporation rate across all attempts
        """
        if len(attempts) < 2:
            return 0.0
        
        incorporation_scores = []
        
        for i in range(len(attempts) - 1):
            curr_attempt = attempts[i]
            next_attempt = attempts[i + 1]
            
            if not curr_attempt.feedback_given or not next_attempt.response_text:
                continue
            
            # Extract key terms from feedback (variable names, concepts)
            feedback = curr_attempt.feedback_given.lower()
            next_response = next_attempt.response_text.lower()
            
            # Look for specific variable names mentioned in feedback
            # e.g., "bias_correction1", "step_size", "denom"
            import re
            var_pattern = r'\b([a-z_][a-z0-9_]{2,})\b'
            feedback_terms = set(re.findall(var_pattern, feedback))
            
            # Filter to meaningful terms (not common words)
            common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'node', 'type', 'value', 'pattern', 'assign'}
            feedback_terms = feedback_terms - common_words
            
            if not feedback_terms:
                continue
            
            # Count how many feedback terms appear in next response
            incorporated = sum(1 for term in feedback_terms if term in next_response)
            score = incorporated / len(feedback_terms) if feedback_terms else 0.0
            incorporation_scores.append(score)
        
        return sum(incorporation_scores) / len(incorporation_scores) if incorporation_scores else 0.0
    
    def _measure_pattern_similarity_to_golden(self, bug_def: dict) -> float:
        """
        Measure structural similarity between generated patterns and golden examples.
        Returns: 0.0-1.0 score for how close patterns are to golden structure
        """
        if not bug_def or 'logic' not in bug_def:
            return 0.0
        
        # Load golden examples
        golden_patterns = []
        for example in self.author.golden_dataset:
            if 'definition' in example and 'logic' in example['definition']:
                for pass_def in example['definition']['logic']:
                    if 'pattern' in pass_def:
                        golden_patterns.append(pass_def['pattern'])
        
        if not golden_patterns:
            return 0.0
        
        # Compare each generated pattern to all golden patterns
        generated_patterns = []
        for pass_def in bug_def.get('logic', []):
            if 'pattern' in pass_def:
                generated_patterns.append(pass_def['pattern'])
        
        if not generated_patterns:
            return 0.0
        
        # For each generated pattern, find best match in golden
        similarity_scores = []
        for gen_pattern in generated_patterns:
            best_match = 0.0
            for gold_pattern in golden_patterns:
                sim = self._compare_pattern_structure(gen_pattern, gold_pattern)
                best_match = max(best_match, sim)
            similarity_scores.append(best_match)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _compare_pattern_structure(self, pattern1: dict, pattern2: dict) -> float:
        """Compare two pattern structures, return similarity score 0.0-1.0"""
        if not isinstance(pattern1, dict) or not isinstance(pattern2, dict):
            return 0.0
        
        score = 0.0
        total_fields = 0
        
        # Check common fields
        for field in ['node_type', 'targets', 'value', 'attr', 'op']:
            total_fields += 1
            if field in pattern1 and field in pattern2:
                # Both have this field
                if field == 'node_type':
                    # Direct comparison
                    score += 1.0 if pattern1[field] == pattern2[field] else 0.0
                elif field == 'targets':
                    # Check if both have targets with similar structure
                    if isinstance(pattern1[field], list) and isinstance(pattern2[field], list):
                        if len(pattern1[field]) > 0 and len(pattern2[field]) > 0:
                            # Both have targets
                            t1 = pattern1[field][0] if pattern1[field] else {}
                            t2 = pattern2[field][0] if pattern2[field] else {}
                            # Check if both have 'id' field (specific variable)
                            has_id_1 = isinstance(t1, dict) and 'id' in t1
                            has_id_2 = isinstance(t2, dict) and 'id' in t2
                            if has_id_1 == has_id_2:
                                score += 1.0  # Both have or both lack specific id
                            else:
                                score += 0.5  # Structure mismatch
                        else:
                            score += 1.0 if len(pattern1[field]) == len(pattern2[field]) else 0.5
                elif field == 'value':
                    # Check if both have value with similar node_type
                    if isinstance(pattern1[field], dict) and isinstance(pattern2[field], dict):
                        if pattern1[field].get('node_type') == pattern2[field].get('node_type'):
                            score += 1.0
                        else:
                            score += 0.5
                else:
                    score += 1.0 if pattern1[field] == pattern2[field] else 0.5
            elif field in pattern1 or field in pattern2:
                # Only one has this field
                score += 0.3
        
        return score / total_fields if total_fields > 0 else 0.0
    
    def manual_llm_output_analysis(self):
        """
        MANUAL ANALYSIS: What stats don't show us.
        Analyze actual LLM-generated text to understand failure modes.
        """
        print("\n" + "="*60)
        print("🔍 MANUAL LLM OUTPUT ANALYSIS")
        print("="*60)
        print("(What statistics fail to reveal)")
        
        for bug_result in self.results:
            print(f"\n{'='*60}")
            print(f"{bug_result.module.upper()}")
            print(f"{'='*60}")
            print(f"Overall: {'✅ SUCCESS' if bug_result.final_success else '❌ FAILED'}")
            
            for i, attempt in enumerate(bug_result.attempts, 1):
                print(f"\n  Attempt {i}: {'✅' if attempt.success else '❌'} {attempt.failure_mode}")
                
                if not attempt.response_text:
                    print("    (No response text captured)")
                    continue
                
                # Parse the response to analyze patterns
                try:
                    # Strip markdown code fences if present
                    response_clean = attempt.response_text.strip()
                    if response_clean.startswith('```'):
                        lines = response_clean.split('\n')
                        if lines[0].startswith('```'):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == '```':
                            lines = lines[:-1]
                        response_clean = '\n'.join(lines)
                    
                    bug_def = json.loads(response_clean)
                    logic = bug_def.get('logic', [])
                    
                    print(f"    Generated {len(logic)} passes")
                    
                    for pass_idx, pass_def in enumerate(logic, 1):
                        if 'pattern' not in pass_def:
                            continue
                        
                        pattern = pass_def['pattern']
                        replacement = pass_def.get('replacement', {})
                        
                        # Human-readable pattern description
                        desc = self._describe_pattern(pattern)
                        repl_desc = replacement.get('type', 'unknown')
                        
                        print(f"      Pass {pass_idx}: {desc} → {repl_desc}")
                        
                        # Check for common issues
                        issues = []
                        if self._is_over_specified(pattern):
                            issues.append("⚠️ OVER-SPECIFIED")
                        if not self._has_specific_variable(pattern):
                            issues.append("⚠️ NO SPECIFIC VAR")
                        if self._wrong_node_type(bug_result.module, pattern, bug_result.expected_patterns):
                            issues.append("❌ WRONG NODE TYPE")
                        
                        if issues:
                            print(f"        Issues: {', '.join(issues)}")
                    
                    # Check for scope mismatch in feedback (permanent diagnostic)
                    if attempt.feedback_given and not attempt.success:
                        if 'def ' in attempt.feedback_given and ('Expected:' in attempt.feedback_given or 'Got:' in attempt.feedback_given):
                            # Check if comparing different scopes
                            lines = attempt.feedback_given.split('\n')
                            expected_lines = [l for l in lines if 'Expected:' in l]
                            got_lines = [l for l in lines if 'Got:' in l]
                            
                            # If one has 'def' and other doesn't, it's a scope mismatch
                            if expected_lines and got_lines:
                                has_def_expected = any('def ' in l for l in expected_lines)
                                has_def_got = any('def ' in l for l in got_lines)
                                
                                if has_def_expected != has_def_got:
                                    print(f"\n    🚨 SCOPE MISMATCH DETECTED:")
                                    print(f"       Expected: {'function' if has_def_expected else 'snippet'}")
                                    print(f"       Got: {'function' if has_def_got else 'snippet'}")
                                    print(f"       → Comparison invalid (apples vs oranges)")
                        
                        # Check for injection success with comparison failure
                        if 'Injection succeeded but transformation was incorrect' in attempt.feedback_given:
                            # Check if this is actually a successful transformation
                            if 'Got:' in attempt.feedback_given and 'def ' in attempt.feedback_given:
                                print(f"\n    💡 LIKELY SUCCESSFUL TRANSFORMATION:")
                                print(f"       Pattern matched and code transformed")
                                print(f"       Failure is in comparison, not transformation")
                                print(f"       Check if transformation is functionally correct")
                    
                except Exception as e:
                    print(f"    Error parsing response: {e}")
                    # Show first 200 chars of response
                    preview = attempt.response_text[:200] + "..." if len(attempt.response_text) > 200 else attempt.response_text
                    print(f"    Response preview: {preview}")
    
    def _describe_pattern(self, pattern: dict) -> str:
        """Convert JSON pattern to human-readable description"""
        node_type = pattern.get('node_type', 'Unknown')
        
        if node_type == 'Assign':
            targets = pattern.get('targets', [])
            if targets and isinstance(targets[0], dict):
                var_name = targets[0].get('id', '?')
                value = pattern.get('value', {})
                value_type = value.get('node_type', '?')
                op = value.get('op', '')
                if op:
                    return f"Assign {var_name} = {value_type}({op})"
                return f"Assign {var_name} = {value_type}"
        elif node_type == 'Call':
            func = pattern.get('func', {})
            attr = func.get('attr', '?')
            return f"Call .{attr}()"
        elif node_type == 'BinOp':
            op = pattern.get('op', '?')
            return f"BinOp({op})"
        
        return node_type
    
    def _is_over_specified(self, pattern: dict) -> bool:
        """Check if pattern is over-specified (too deep nesting)"""
        def depth(obj, current=0):
            if not isinstance(obj, dict):
                return current
            max_d = current
            for v in obj.values():
                if isinstance(v, dict):
                    max_d = max(max_d, depth(v, current + 1))
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            max_d = max(max_d, depth(item, current + 1))
            return max_d
        
        return depth(pattern) > 3
    
    def _has_specific_variable(self, pattern: dict) -> bool:
        """Check if pattern includes specific variable name (has 'id' field)"""
        if 'targets' in pattern:
            targets = pattern['targets']
            if targets and isinstance(targets[0], dict):
                return 'id' in targets[0]
        return False
    
    def _wrong_node_type(self, module: str, pattern: dict, expected: dict) -> bool:
        """Check if pattern uses wrong AST node type"""
        if module not in expected:
            return False
        
        # Try to match pattern to expected patterns
        if 'targets' in pattern:
            targets = pattern.get('targets', [])
            if targets and isinstance(targets[0], dict):
                var_name = targets[0].get('id')
                if var_name in expected:
                    expected_node, expected_op = expected[var_name]
                    value = pattern.get('value', {})
                    actual_node = value.get('node_type')
                    if actual_node and actual_node != expected_node:
                        return True
        
        return False
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\n" + "="*60)
        print("SYSTEMATIC EVALUATION RESULTS")
        print("="*60)
        
        # Overall success rate
        total_bugs = len(self.results)
        successful_bugs = sum(1 for r in self.results if r.final_success)
        success_rate = (successful_bugs / total_bugs * 100) if total_bugs > 0 else 0
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total bugs tested: {total_bugs}")
        print(f"  Successful: {successful_bugs}")
        print(f"  Failed: {total_bugs - successful_bugs}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        # Check for false negatives (injection succeeded but marked as failure)
        false_negatives = []
        for bug_result in self.results:
            if not bug_result.final_success:
                for attempt in bug_result.attempts:
                    if attempt.is_false_negative:
                        false_negatives.append((bug_result.module, attempt.attempt_num))
        
        if false_negatives:
            print(f"\n⚠️  FALSE NEGATIVES DETECTED:")
            print(f"  {len(false_negatives)} attempt(s) where injection succeeded but comparison failed")
            for module, attempt_num in false_negatives:
                print(f"    - {module} attempt {attempt_num}: Transformation works, comparison issue")
            print(f"\n  💡 Actual success rate likely higher than reported!")
            print(f"     If these are true successes: ~{((successful_bugs + len(set(m for m, _ in false_negatives))) / total_bugs * 100):.0f}%")
        
        # Success by complexity
        print(f"\n📊 SUCCESS BY COMPLEXITY:")
        for complexity in ["simple", "medium", "complex"]:
            bugs_of_complexity = [r for r in self.results if r.complexity == complexity]
            if bugs_of_complexity:
                successes = sum(1 for r in bugs_of_complexity if r.final_success)
                rate = (successes / len(bugs_of_complexity) * 100)
                print(f"  {complexity.capitalize()}: {successes}/{len(bugs_of_complexity)} ({rate:.1f}%)")
        
        # Improvement across attempts
        print(f"\n📊 IMPROVEMENT ACROSS ATTEMPTS:")
        for attempt_num in range(1, 4):
            attempt_results = []
            for bug_result in self.results:
                if len(bug_result.attempts) >= attempt_num:
                    attempt_results.append(bug_result.attempts[attempt_num - 1])
            
            if attempt_results:
                successes = sum(1 for a in attempt_results if a.success)
                rate = (successes / len(attempt_results) * 100)
                avg_patterns = sum(a.patterns_generated for a in attempt_results) / len(attempt_results)
                with_specific_vars = sum(1 for a in attempt_results if a.had_specific_var_names)
                print(f"  Attempt {attempt_num}: {successes}/{len(attempt_results)} success ({rate:.1f}%), "
                      f"avg {avg_patterns:.1f} patterns, {with_specific_vars} with specific vars")
        
        # Failure modes
        print(f"\n📊 FAILURE MODES:")
        failure_counts = {}
        for bug_result in self.results:
            if not bug_result.final_success:
                last_attempt = bug_result.attempts[-1]
                mode = last_attempt.failure_mode
                failure_counts[mode] = failure_counts.get(mode, 0) + 1
        
        for mode, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count}")
        
        # Time statistics
        print(f"\n📊 TIME STATISTICS:")
        avg_time = sum(r.time_taken for r in self.results) / len(self.results)
        print(f"  Average time per bug: {avg_time:.1f}s")
        print(f"  Total time: {sum(r.time_taken for r in self.results):.1f}s")
        
        # NEW DIAGNOSTIC METRICS
        print(f"\n📊 DETAILED DIAGNOSTIC METRICS:")
        for bug_result in self.results:
            consistency = self._measure_attempt_consistency(bug_result.attempts)
            incorporation = self._measure_feedback_incorporation(bug_result.attempts)
            # Average similarity to golden across attempts
            avg_similarity = sum(a.similarity_to_golden for a in bug_result.attempts) / len(bug_result.attempts) if bug_result.attempts else 0.0
            
            print(f"  {bug_result.module}:")
            print(f"    Attempt consistency: {consistency:.2f} (1.0 = all identical, 0.0 = all different)")
            print(f"    Feedback incorporation: {incorporation:.2f} (how much feedback was used)")
            print(f"    Similarity to golden: {avg_similarity:.2f} (how close to golden examples)")
        
        # Overall averages
        avg_consistency = sum(self._measure_attempt_consistency(r.attempts) for r in self.results) / len(self.results)
        avg_incorporation = sum(self._measure_feedback_incorporation(r.attempts) for r in self.results) / len(self.results)
        all_similarities = [a.similarity_to_golden for r in self.results for a in r.attempts if a.similarity_to_golden > 0]
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
        
        print(f"\n  OVERALL AVERAGES:")
        print(f"    Attempt consistency: {avg_consistency:.2f}")
        print(f"    Feedback incorporation: {avg_incorporation:.2f}")
        print(f"    Similarity to golden: {avg_similarity:.2f}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if avg_consistency > 0.9:
            print(f"  ⚠️  HIGH CONSISTENCY ({avg_consistency:.2f}) - LLM is stuck generating same pattern")
        if avg_incorporation < 0.3:
            print(f"  ⚠️  LOW INCORPORATION ({avg_incorporation:.2f}) - LLM not using feedback")
        if avg_similarity < 0.4:
            print(f"  ⚠️  LOW SIMILARITY ({avg_similarity:.2f}) - Patterns very different from golden examples")
        elif avg_similarity > 0.7:
            print(f"  ✅  HIGH SIMILARITY ({avg_similarity:.2f}) - Patterns structurally close to golden")
        
        # NEW: Node Type Accuracy Statistics
        print(f"\n📊 AUTOMATED NODE TYPE ACCURACY:")
        print(f"="*60)
        
        # Overall average
        all_accuracies = [a.node_type_accuracy for r in self.results for a in r.attempts if a.node_type_accuracy > 0]
        avg_node_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        
        print(f"\nOverall Node Type Accuracy: {avg_node_accuracy:.1f}%")
        
        # CRITICAL: Detect pattern matcher bugs (100% accurate but fails)
        perfect_but_failing = []
        for bug_result in self.results:
            for attempt in bug_result.attempts:
                if attempt.node_type_accuracy == 100 and not attempt.success:
                    perfect_but_failing.append({
                        'bug': bug_result.module,
                        'attempt': attempt.attempt_num,
                        'failure_mode': attempt.failure_mode,
                        'over_specified': attempt.pattern_complexity.get('over_specified', 0) if attempt.pattern_complexity else 0
                    })
        
        if perfect_but_failing:
            print(f"\n🚨 PATTERN MATCHER BUG DETECTED:")
            print(f"  {len(perfect_but_failing)} attempts have 100% accurate patterns but FAIL")
            for item in perfect_but_failing:
                over_spec_note = f" (over-specified: {item['over_specified']} patterns)" if item['over_specified'] > 0 else ""
                print(f"    - {item['bug']} attempt {item['attempt']}: {item['failure_mode']}{over_spec_note}")
            print(f"  → This indicates a bug in the pattern matcher, not LLM generation")
        
        # Per-bug breakdown
        for bug_result in self.results:
            print(f"\n{bug_result.module}:")
            for attempt in bug_result.attempts:
                if attempt.node_type_accuracy > 0:
                    status = "✅" if attempt.node_type_accuracy == 100 else "⚠️"
                    fail_note = " BUT FAILS" if attempt.node_type_accuracy == 100 and not attempt.success else ""
                    print(f"  Attempt {attempt.attempt_num}: {status} {attempt.node_type_accuracy:.0f}% accurate{fail_note}")
                    
                    # Show per-variable details if not 100%
                    if attempt.node_type_accuracy < 100 and attempt.node_type_details:
                        for var, details in attempt.node_type_details.items():
                            if not details['correct']:
                                exp = details['expected']
                                gen = details['generated']
                                print(f"    ❌ {var}: expected {exp[0]}/{exp[1]}, got {gen[0]}/{gen[1]}")
        
        # NEW: Detailed Pattern Analysis for Manual Review
        print(f"\n📋 DETAILED PATTERN ANALYSIS (For Manual Review):")
        print(f"="*60)
        
        for bug_result in self.results:
            print(f"\n{bug_result.module} ({bug_result.complexity}):")
            
            for attempt in bug_result.attempts:
                print(f"\n  Attempt {attempt.attempt_num}: {attempt.failure_mode}")
                print(f"    Has specific vars: {'✅' if attempt.had_specific_var_names else '❌'}")
                print(f"    Node type accuracy: {attempt.node_type_accuracy:.0f}%")
                
                # Pattern complexity
                if attempt.pattern_complexity:
                    over_spec = attempt.pattern_complexity.get('over_specified', 0)
                    simple = attempt.pattern_complexity.get('simple', 0)
                    if over_spec > 0:
                        print(f"    ⚠️  Over-specified patterns: {over_spec}/{over_spec+simple}")
                    else:
                        print(f"    ✅ Appropriately simple patterns")
                
                # Missing operators
                if attempt.missing_operators:
                    print(f"    ⚠️  Missing operators in passes: {attempt.missing_operators}")
                
                # Wrong node types
                if attempt.wrong_node_types:
                    print(f"    ⚠️  Node type issues:")
                    for issue in attempt.wrong_node_types:
                        print(f"      - {issue}")
        
        print(f"\n{'='*60}")
    
    def save_results(self, output_path: Path):
        """Save detailed results to JSON"""
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_bugs": len(self.results),
                "successful": sum(1 for r in self.results if r.final_success),
                "success_rate": (sum(1 for r in self.results if r.final_success) / len(self.results) * 100) if self.results else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    """Run systematic evaluation"""
    
    # Default to gpt-4o-mini for cost efficiency
    # Set model="gpt-4o" for better first-try success (100% improvement vs gpt-4o-mini)
    print("🧠 Using model: gpt-4o (smarter model)")
    evaluator = SystematicEvaluator(model="gpt-4o")
    
    base_path = Path("/Volumes/Totallynotaharddrive/assignment1-basics/curricula/cs336_a1/modules")
    
    # Test suite: ALL 21 bugs from curriculum
    test_bugs = [
        # ===== SIMPLE (1-2 operations) =====
        {
            "module": "silu",
            "patch": base_path / "silu/bugs/missing_multiply.patch",
            "symptom": "Missing multiplication in SiLU",
            "complexity": "simple",
            "operations": 1
        },
        {
            "module": "attention",
            "patch": base_path / "attention/bugs/missing_scale.patch",
            "symptom": "Missing scaling in attention",
            "complexity": "simple",
            "operations": 2
        },
        {
            "module": "rmsnorm",
            "patch": base_path / "rmsnorm/bugs/missing_keepdim.patch",
            "symptom": "Missing keepdim in RMSNorm",
            "complexity": "simple",
            "operations": 1
        },
        {
            "module": "softmax",
            "patch": base_path / "softmax/bugs/no_subtract_max.patch",
            "symptom": "Missing max subtraction in softmax",
            "complexity": "simple",
            "operations": 1
        },
        {
            "module": "linear",
            "patch": base_path / "linear/bugs/missing_transpose.patch",
            "symptom": "Missing transpose in linear layer",
            "complexity": "simple",
            "operations": 1
        },
        {
            "module": "embedding",
            "patch": base_path / "embedding/bugs/wrong_dimension_order.patch",
            "symptom": "Wrong dimension order in embedding",
            "complexity": "simple",
            "operations": 1
        },
        {
            "module": "swiglu",
            "patch": base_path / "swiglu/bugs/missing_gate.patch",
            "symptom": "Missing gate in SwiGLU",
            "complexity": "simple",
            "operations": 1
        },
        
        # ===== MEDIUM (2-3 operations or keyword args) =====
        {
            "module": "cross_entropy",
            "patch": base_path / "cross_entropy/bugs/no_logsumexp.patch",
            "symptom": "Not using logsumexp in cross entropy",
            "complexity": "medium",
            "operations": 2
        },
        {
            "module": "multihead_attention",
            "patch": base_path / "multihead_attention/bugs/missing_transpose_back.patch",
            "symptom": "Missing transpose back in multihead attention",
            "complexity": "medium",
            "operations": 1
        },
        {
            "module": "rope",
            "patch": base_path / "rope/bugs/wrong_rotation.patch",
            "symptom": "Wrong rotation in RoPE",
            "complexity": "medium",
            "operations": 2
        },
        {
            "module": "cosine_schedule",
            "patch": base_path / "cosine_schedule/bugs/wrong_cosine_range.patch",
            "symptom": "Wrong cosine range in scheduler",
            "complexity": "medium",
            "operations": 2
        },
        {
            "module": "gradient_clipping",
            "patch": base_path / "gradient_clipping/bugs/per_parameter_clipping.patch",
            "symptom": "Per-parameter instead of global clipping",
            "complexity": "medium",
            "operations": 2
        },
        {
            "module": "text_generation",
            "patch": base_path / "text_generation/bugs/temperature_after_softmax.patch",
            "symptom": "Applying temperature after softmax",
            "complexity": "medium",
            "operations": 2
        },
        {
            "module": "transformer_block",
            "patch": base_path / "transformer_block/bugs/missing_residual.patch",
            "symptom": "Missing residual connection",
            "complexity": "medium",
            "operations": 2
        },
        
        # ===== COMPLEX (3+ operations or multi-statement) =====
        {
            "module": "adamw",
            "patch": base_path / "adamw/bugs/missing_bias_correction.patch",
            "symptom": "Missing bias correction in AdamW",
            "complexity": "complex",
            "operations": 4
        },
        {
            "module": "bpe_tokenizer",
            "patch": base_path / "bpe_tokenizer/bugs/wrong_merge_order.patch",
            "symptom": "Wrong merge order in BPE",
            "complexity": "complex",
            "operations": 3
        },
        {
            "module": "tokenizer_class",
            "patch": base_path / "tokenizer_class/bugs/wrong_merge_order.patch",
            "symptom": "Wrong merge order in tokenizer",
            "complexity": "complex",
            "operations": 3
        },
        {
            "module": "data_loader",
            "patch": base_path / "data_loader/bugs/wrong_sampling_range.patch",
            "symptom": "Wrong sampling range in data loader",
            "complexity": "complex",
            "operations": 2
        },
        {
            "module": "checkpointing",
            "patch": base_path / "checkpointing/bugs/missing_optimizer_state.patch",
            "symptom": "Missing optimizer state in checkpoint",
            "complexity": "complex",
            "operations": 2
        },
        {
            "module": "training_loop",
            "patch": base_path / "training_loop/bugs/missing_zero_grad.patch",
            "symptom": "Missing zero_grad in training loop",
            "complexity": "complex",
            "operations": 2
        },
        {
            "module": "transformer_lm",
            "patch": base_path / "transformer_lm/bugs/missing_final_norm.patch",
            "symptom": "Missing final normalization",
            "complexity": "complex",
            "operations": 2
        },
    ]
    
    print("="*60)
    print("SYSTEMATIC LLM EVALUATION")
    print("="*60)
    print(f"\nTesting {len(test_bugs)} bugs with 3 attempts each")
    print(f"Golden examples: {len(evaluator.author.golden_dataset)}")
    print(f"Collecting quantitative data on:")
    print("  - Success rates by complexity")
    print("  - Improvement across attempts")
    print("  - Failure mode distribution")
    print("  - Pattern generation quality")
    
    for bug_info in test_bugs:
        if bug_info["patch"].exists():
            evaluator.evaluate_bug(
                module=bug_info["module"],
                patch_path=bug_info["patch"],
                symptom=bug_info["symptom"],
                complexity=bug_info["complexity"],
                num_operations=bug_info["operations"],
                max_attempts=3
            )
        else:
            print(f"\n⚠️  Skipping {bug_info['module']}: patch not found")
    
    # Manual analysis FIRST (what stats don't show)
    evaluator.manual_llm_output_analysis()
    
    # Then print statistics
    evaluator.print_statistics()
    
    # Save results
    output_path = Path("/tmp/llm_evaluation_results.json")
    evaluator.save_results(output_path)


def check_regression(results_path: Path):
    """
    Check for performance regression against baseline.
    Baseline: 3/4 bugs (75%) - attention, rmsnorm, adamw
    """
    if not results_path.exists():
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    overall_success = sum(1 for bug in data['results'] if any(a['success'] for a in bug['attempts']))
    
    BASELINE_SUCCESS = 3
    BASELINE_BUGS = {"attention", "rmsnorm", "adamw"}
    
    current_bugs = {bug['module'] for bug in data['results'] if any(a['success'] for a in bug['attempts'])}
    
    print("\n" + "="*60)
    print("🔒 REGRESSION CHECK")
    print("="*60)
    print(f"Baseline: {BASELINE_SUCCESS}/4 bugs - {', '.join(sorted(BASELINE_BUGS))}")
    print(f"Current:  {overall_success}/4 bugs - {', '.join(sorted(current_bugs))}")
    
    if overall_success < BASELINE_SUCCESS:
        print(f"\n❌ REGRESSION DETECTED!")
        print(f"   Performance dropped from {BASELINE_SUCCESS}/4 to {overall_success}/4")
        print(f"   Lost: {BASELINE_BUGS - current_bugs}")
        print(f"\n   This should not happen! Revert recent changes.")
        return False
    elif overall_success == BASELINE_SUCCESS:
        if current_bugs == BASELINE_BUGS:
            print(f"\n✅ No regression - same bugs passing")
        else:
            print(f"\n⚠️  Same count but different bugs")
            print(f"   Lost: {BASELINE_BUGS - current_bugs}")
            print(f"   Gained: {current_bugs - BASELINE_BUGS}")
        return True
    else:
        print(f"\n🎉 IMPROVEMENT!")
        print(f"   Performance improved from {BASELINE_SUCCESS}/4 to {overall_success}/4")
        print(f"   Gained: {current_bugs - BASELINE_BUGS}")
        return True


if __name__ == "__main__":
    main()
    
    # Check for regression after evaluation
    results_path = Path("/tmp/llm_evaluation_results.json")
    check_regression(results_path)
