# Experiment Module Framework Design

## Purpose

Extend the Build-Justify-Harden (BJH) framework to support **experimental investigation** modules that teach the scientific method applied to ML systems.

## Motivation

The CS336 PDF §7 treats experiments as the culmination of the assignment, but the current curriculum:
- **Mentions experiments informally** (no structured guidance)
- **Lacks rigorous experimental design teaching** (hypothesis, ablation, interpretation)
- **Misses opportunity** to apply BJH framework to scientific process

Modern ML research requires:
1. **Hypothesis formation** - Clear, testable claims
2. **Experimental design** - Controlled ablations, baselines
3. **Result interpretation** - Statistical significance, confounders
4. **Debugging experiments** - Identifying flawed setups

These skills are as important as implementation, but currently not systematically taught.

## Design Philosophy

**Insight**: The scientific method IS a Build-Justify-Harden loop!

```
Build   → Design and run experiment
Justify → Explain why this design tests the hypothesis
Harden  → Debug flawed experimental setup
```

This naturally maps experimental investigation to the BJH framework.

## Experiment Module Structure

### Module Type Declaration

In `manifest.json`:

```json
{
  "id": "rope_ablation",
  "name": "RoPE Ablation Study",
  "path": "modules/rope_ablation",
  "baseline_perf_seconds": 5.0,
  "dependencies": ["training_loop", "transformer_lm"],
  "module_type": "experiment"
}
```

### Directory Structure

```
modules/rope_ablation/
├── README.md                    # Experiment overview
├── experiment_prompt.txt        # Build stage: experimental task
├── justify_questions.json       # Justify stage: design rationale
├── flawed_setups/               # Harden stage: buggy experiments
│   ├── bug1_no_baseline.patch
│   ├── bug2_insufficient_runs.patch
│   └── bug3_confounded_variables.patch
└── reference_results/           # Expected outcomes
    ├── baseline.json
    ├── ablation.json
    └── analysis_template.md
```

### File Purposes

1. **`experiment_prompt.txt`** (Build stage):
   - Hypothesis to test
   - Experimental design instructions
   - Required measurements
   - Baseline requirements
   - Analysis expectations

2. **`justify_questions.json`** (Justify stage):
   - Why this design tests the hypothesis
   - What confounders are controlled
   - How to interpret results
   - Statistical considerations

3. **`flawed_setups/`** (Harden stage):
   - Buggy experimental code
   - Common mistakes (no baseline, p-hacking, confounders)
   - Students debug and fix

4. **`reference_results/`**:
   - Expected results for validation
   - Analysis templates
   - Interpretation guidelines

## Stage Design

### Build Stage: Design and Run Experiment

**Input**: `experiment_prompt.txt` specifies hypothesis and task

**Example**: RoPE Ablation Study

```markdown
# Experiment: Does RoPE improve long-sequence perplexity?

## Hypothesis
RoPE (Rotary Position Embeddings) improves perplexity on sequences 
longer than training length, compared to fixed sinusoidal embeddings.

## Task
Design and execute an ablation study:

1. **Baseline**: Train model with sinusoidal position embeddings
   - Train on seq_len=512
   - Evaluate on seq_len=512, 1024, 2048

2. **Ablation**: Train model with RoPE
   - Train on seq_len=512 (same as baseline)
   - Evaluate on seq_len=512, 1024, 2048

3. **Requirements**:
   - ONLY change position embedding (keep all other hyperparameters identical)
   - Run 3 seeds for each condition (statistical reliability)
   - Measure: perplexity, wall-clock time, memory usage
   - Plot: perplexity vs sequence length (with error bars)

4. **Deliverables**:
   - Training script (baseline and ablation)
   - Evaluation script
   - Results JSON files
   - Analysis notebook with plots

## Success Criteria
- Both models train to similar perplexity on train length (512)
- Clear comparison of extrapolation performance (1024, 2048)
- Statistical analysis (mean, std, significance tests)
- Interpretation: Does RoPE help? By how much? Why?
```

**Student Output**:
- Experimental code (`experiment.py`)
- Results files (`baseline.json`, `rope.json`)
- Analysis notebook (`analysis.ipynb`)

**Validation**: `validator.sh` checks:
- Results files exist
- Required measurements present
- Multiple seeds run
- Plots generated

### Justify Stage: Experimental Design Rationale

**Input**: `justify_questions.json` with experimental design questions

**Example Questions**:

```json
[
  {
    "id": "rope_ablation_q1",
    "question": "Why is it critical to use the SAME training sequence length (512) for both baseline and RoPE models? What would be confounded if we trained baseline on 512 but RoPE on 1024?",
    "model_answer": "Using different training lengths confounds two variables: (1) Position encoding method (sinusoidal vs RoPE), and (2) Training data amount (512 vs 1024 tokens). If RoPE trained on 1024 performs better, we can't isolate whether it's because RoPE is superior OR because it saw more training data. Controlled experiment: Change ONE variable (position encoding), hold all others constant (train length, data, hyperparams, architecture). This isolates the causal effect of RoPE specifically.",
    "required_concepts": [
      "Ablation studies change ONE variable",
      "Confounders: multiple variables changing simultaneously",
      "Controlled experiment: isolate causal factors",
      "Training length affects data amount seen",
      "Can't attribute results to RoPE vs data amount"
    ]
  },
  {
    "id": "rope_ablation_q2",
    "question": "Why run 3 seeds instead of 1? Calculate: If baseline gets 45.2 perplexity (1 run) and RoPE gets 43.8 (1 run), can you conclude RoPE is better? What about 45.2±0.3 (3 runs) vs 43.8±2.5 (3 runs)?",
    "model_answer": "Single runs are unreliable due to random initialization, data order, GPU non-determinism. Scenario 1 (single run): Baseline 45.2, RoPE 43.8. Difference: 1.4 points. Looks like RoPE wins! BUT: No variance estimate. Could be random luck. Can't compute statistical significance. Scenario 2 (multiple runs): Baseline: 45.2±0.3 (std=0.3, N=3). RoPE: 43.8±2.5 (std=2.5, N=3). Analysis: Baseline is STABLE (low variance). RoPE is NOISY (high variance). Error bars OVERLAP! (43.8-2.5=41.3, 45.2+0.3=45.5 - ranges overlap). Statistical significance: Can't claim RoPE is better (might be noise). Correct approach: Run more seeds (N=10), compute t-test, check p<0.05. Only then can we claim significance! Key insight: Point estimates (single runs) are meaningless. Need variance estimates to make claims!",
    "required_concepts": [
      "Random variation: initialization, data order, GPU",
      "Single runs: no variance estimate, unreliable",
      "Multiple runs: compute mean and std",
      "Error bars: visualize uncertainty",
      "Overlapping error bars: results not significant",
      "Statistical tests: t-test, p-values",
      "N=3 is minimum, N=10+ is better"
    ]
  },
  {
    "id": "rope_ablation_q3",
    "question": "Your results show RoPE improves 512-token perplexity (trained length) by 10%. Is this evidence that RoPE helps with LENGTH EXTRAPOLATION? Why or why not? What result would actually support the extrapolation hypothesis?",
    "model_answer": "NO! Improvement at training length does NOT test extrapolation. Extrapolation = generalization BEYOND training length. Evidence for extrapolation: Compare performance at UNSEEN lengths (1024, 2048 tokens). Scenario 1 (NOT extrapolation): Train length 512: RoPE 40.0, Baseline 44.4 (RoPE better!). Eval length 1024: RoPE 55.0, Baseline 54.8 (RoPE slightly better). Eval length 2048: RoPE 75.2, Baseline 75.0 (basically same). Conclusion: RoPE better at training length, but NO extrapolation advantage! Scenario 2 (TRUE extrapolation): Train length 512: RoPE 40.0, Baseline 40.2 (same). Eval length 1024: RoPE 48.5, Baseline 54.8 (RoPE better!). Eval length 2048: RoPE 62.1, Baseline 75.0 (RoPE MUCH better!). Conclusion: RoPE EXTRAPOLATES! Gap widens at longer lengths. This tests the hypothesis! Key insight: Extrapolation is about OUT-OF-DISTRIBUTION generalization. Must evaluate beyond training length to make claims about extrapolation!",
    "required_concepts": [
      "Extrapolation = generalization beyond training distribution",
      "Training length performance ≠ extrapolation",
      "Must evaluate at unseen lengths (1024, 2048)",
      "Extrapolation hypothesis: gap widens at longer lengths",
      "In-distribution vs out-of-distribution evaluation",
      "Hypothesis-specific measurements required"
    ]
  }
]
```

**Student Task**: Answer questions demonstrating understanding of:
- Confounders and controlled experiments
- Statistical significance
- Hypothesis-specific evaluation

### Harden Stage: Debug Flawed Experiments

**Input**: `flawed_setups/*.patch` with buggy experimental code

**Example Bugs**:

#### Bug 1: No Baseline
```python
# flawed_setups/bug1_no_baseline.patch
# BUG: Only trains RoPE, no baseline comparison!

def run_experiment():
    model_rope = train_with_rope(seq_len=512)
    results = evaluate(model_rope, [512, 1024, 2048])
    print(f"RoPE perplexity: {results}")
    print("Conclusion: RoPE works!")
    # BUG: Can't conclude anything without baseline!
```

**Student Task**: Identify the bug (no baseline), fix it (add sinusoidal baseline).

#### Bug 2: Insufficient Runs
```python
# flawed_setups/bug2_insufficient_runs.patch
# BUG: Single run, no variance estimate!

def run_experiment():
    model_baseline = train_with_sinusoidal(seq_len=512, seed=42)
    model_rope = train_with_rope(seq_len=512, seed=42)
    # BUG: Only 1 seed! Can't measure variance!
    results = {"baseline": eval(model_baseline), "rope": eval(model_rope)}
    if results["rope"] < results["baseline"]:
        print("RoPE is better!")  # BUG: Can't claim significance!
```

**Student Task**: Add multiple seeds, compute mean/std, check significance.

#### Bug 3: Confounded Variables
```python
# flawed_setups/bug3_confounded_variables.patch
# BUG: Changes multiple variables simultaneously!

def run_experiment():
    # BUG: Baseline uses d_model=512, RoPE uses d_model=1024!
    model_baseline = train_with_sinusoidal(d_model=512, seq_len=512)
    model_rope = train_with_rope(d_model=1024, seq_len=512)
    # BUG: RoPE has 4× more parameters! Unfair comparison!
```

**Student Task**: Identify confounded variables, fix to isolate position encoding only.

**Validation**: Students explain what was wrong and provide fixed code.

## Integration with BJH Loop

### Build → Justify → Harden Flow

```
Build Stage:
User: Designs experiment, runs ablation, generates results
Engine: Validates results files exist, plots generated
→ Advance to Justify

Justify Stage:
User: Answers questions about experimental design
LLM: Evaluates understanding of confounders, statistics
→ Advance to Harden if correct

Harden Stage:
User: Receives buggy experimental code
User: Identifies bug, explains flaw, provides fix
LLM: Validates explanation and fix
→ Advance to next module if correct
```

## Example Experiment Modules

### 1. RoPE Ablation Study
**Hypothesis**: RoPE improves length extrapolation  
**Design**: Train baseline (sinusoidal) vs RoPE, evaluate at multiple lengths  
**Bugs**: No baseline, single seed, confounded architecture

### 2. Batch Size Scaling
**Hypothesis**: Larger batch size requires learning rate adjustment  
**Design**: Train with batch sizes [32, 64, 128, 256], sweep LR for each  
**Bugs**: Fixed LR across batch sizes, no learning rate warmup, insufficient training steps

### 3. Tokenizer Vocabulary Size
**Hypothesis**: Larger vocab reduces sequence length but increases embedding cost  
**Design**: Train with vocab [8K, 16K, 32K, 50K], measure memory and perplexity  
**Bugs**: Different training data amount, no memory profiling, unfair perplexity comparison

### 4. Optimizer Comparison
**Hypothesis**: AdamW outperforms SGD for Transformer training  
**Design**: Train with SGD vs AdamW, control for LR schedule and hyperparameters  
**Bugs**: Different LR schedules, no momentum for SGD, single seed

### 5. Attention Head Ablation
**Hypothesis**: Not all attention heads are equally important  
**Design**: Train full model, then ablate individual heads, measure perplexity  
**Bugs**: Ablation changes architecture size, no fine-tuning after ablation, confounded capacity

## Implementation Requirements

### Schema Updates

```python
class ExperimentMetadata(BaseModel):
    """Metadata for experiment modules."""
    module_type: str = "experiment"
    hypothesis: str
    required_measurements: list[str]
    statistical_requirements: dict  # e.g., {"min_seeds": 3, "significance_level": 0.05}
```

### Validator Extensions

`validator.sh` for experiments checks:
- Results files exist (JSON, plots)
- Required measurements present
- Multiple seeds run
- Statistical analysis performed

```bash
#!/bin/bash
# validator.sh for experiment modules

# Check results files
test -f results/baseline.json || { echo "Missing baseline results"; exit 1; }
test -f results/ablation.json || { echo "Missing ablation results"; exit 1; }

# Check plots
test -f plots/perplexity_vs_length.png || { echo "Missing plot"; exit 1; }

# Validate measurements
python validate_experiment.py --results results/ --requirements requirements.json
```

### Justify Questions Structure

Experiment modules use standard `justify_questions.json` focused on:
- Experimental design rationale
- Confounder identification
- Statistical interpretation
- Hypothesis-specific evaluation

### Harden Bug Categories

**Common Experimental Flaws**:
1. **No Baseline**: Only ablation, no control
2. **Confounded Variables**: Multiple changes simultaneously
3. **Insufficient Statistical Power**: Single run, no variance
4. **P-Hacking**: Selective reporting, multiple hypothesis testing
5. **Incorrect Evaluation**: Wrong metric, wrong distribution
6. **Implementation Bugs**: Off-by-one, wrong hyperparams, data leakage

## Pedagogical Benefits

1. **Rigorous Scientific Training**:
   - Hypothesis formation
   - Controlled experiments
   - Statistical thinking

2. **Practical Skills**:
   - Ablation study design
   - Baseline selection
   - Result interpretation

3. **Error Detection**:
   - Identifying flawed experiments
   - Debugging methodology
   - Critical thinking

4. **Research Preparation**:
   - Skills for ML research papers
   - Reproducibility practices
   - Honest reporting

## Module Sequencing

Experiments come AFTER implementation mastery:

```
Early Curriculum:
- Foundation components (softmax, attention, etc.)
- Architecture building (transformer, training loop)
- Tokenization pipeline

Late Curriculum:
- Experiment 1: RoPE ablation (position encoding)
- Experiment 2: Batch size scaling (optimization)
- Experiment 3: Vocabulary size (tokenization)
- Experiment 4: Optimizer comparison (training)
- Experiment 5: Head ablation (architecture)
```

**Rationale**: Students must understand implementation before designing experiments.

## Comparison with Other Module Types

| Module Type | Build Stage | Justify Stage | Harden Stage |
|-------------|-------------|---------------|--------------|
| **Standard** | Implement algorithm | Explain design | Fix bugs |
| **Justify-Only** | (skip) | Answer theory | (skip) |
| **Experiment** | Design/run experiment | Explain methodology | Debug flawed setup |

Experiments are closest to standard modules but focus on scientific method rather than implementation.

## Migration Path

**Phase 1**: Design framework (this document)

**Phase 2**: Create 2-3 example experiment modules
- RoPE ablation (position encoding)
- Batch size scaling (optimization)
- Vocabulary size (tokenization)

**Phase 3**: Engine support
- Add `module_type: "experiment"` handling
- Extend validator for experiment-specific checks
- Update justify stage for experimental design questions

**Phase 4**: Curriculum integration
- Add experiments after training_loop module
- Update manifest dependencies
- Test end-to-end flow

## Assessment Criteria

Students pass experiment modules by demonstrating:

**Build Stage**:
- ✅ Correct experimental design (baseline + ablation)
- ✅ Required measurements collected
- ✅ Multiple seeds run
- ✅ Results visualized appropriately

**Justify Stage**:
- ✅ Understanding of confounders
- ✅ Statistical significance awareness
- ✅ Hypothesis-specific evaluation
- ✅ Interpretation of results

**Harden Stage**:
- ✅ Identification of experimental flaws
- ✅ Explanation of why setup is wrong
- ✅ Corrected methodology

## Future Extensions

### Advanced Experiment Types

1. **Scaling Laws**: Compute optimal model size for given budget
2. **Hyperparameter Search**: Design efficient search strategies
3. **Architecture Search**: Systematic component comparison
4. **Robustness Studies**: Evaluate under distribution shift
5. **Efficiency Analysis**: Wall-clock time vs accuracy trade-offs

### Multi-Module Experiments

Experiments spanning multiple concepts:
- Position encoding + context length + memory
- Tokenization + vocabulary + multilingual performance
- Architecture + optimization + generalization

## Example: Complete RoPE Ablation Module

### Directory Structure
```
modules/rope_ablation/
├── README.md
├── experiment_prompt.txt          # Hypothesis and experimental design
├── justify_questions.json         # 5 questions on methodology
├── flawed_setups/
│   ├── bug1_no_baseline.patch
│   ├── bug2_single_seed.patch
│   └── bug3_confounded_architecture.patch
├── reference_results/
│   ├── baseline.json              # Expected baseline results
│   ├── rope.json                  # Expected RoPE results
│   └── analysis_template.md      # Interpretation guide
└── validator.sh                   # Checks results and plots exist
```

### Success Flow
1. **Build**: Student designs experiment, trains models, generates plots
2. **Justify**: Student explains why this design tests extrapolation hypothesis
3. **Harden**: Student debugs 3 flawed experimental setups
4. **Complete**: Module marked done, student advances

---

**Status**: Framework design complete, awaiting implementation.

**Next Steps**:
1. Create 2-3 example experiment modules
2. Implement engine support for experiment module type
3. Write validator extensions for experiments
4. Test end-to-end experimental workflow
5. Document experimental best practices for students
