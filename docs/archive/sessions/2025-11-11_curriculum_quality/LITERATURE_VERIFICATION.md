# CS336 Curriculum - Literature Verification Guide

## Purpose

This document maps each curriculum module to its ground-truth literature sources, enabling systematic verification of pedagogical claims against primary research.

## Base Literature Path

All analysis files referenced below are located at:

```
/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/docs/5_domain_knowledge_and_curricula/computer_science/
```

---

## New Modules Created (Priority 2 Work)

### 1. **embedding** (Fixed for From-Scratch)

**Literature Source:**
- `transformer_paradigm/Attention_Is_All_You_Need_Analysis.md`

**What to Verify:**
- Learned embeddings as Transformer input
- Weight tying (tied embeddings) concept
- Embedding dimension and model dimension relationship

**Key Claims in Build Prompt:**
- Uses `nn.Parameter` directly (no `nn.Embedding`)
- Truncated normal initialization
- Forward pass via table lookup

---

### 2. **linear** (New Module)

**Literature Sources:**
- `transformer_paradigm/Attention_Is_All_You_Need_Analysis.md`
- `core_architectural_components/GLU_Variants_Improve_Transformer_Analysis.md`

**What to Verify:**
- Linear layer as building block of position-wise FFN (original Transformer)
- Usage in three-matrix SwiGLU design (W1, W2, W3)
- Kaiming initialization rationale

**Key Claims in Build Prompt:**
- y = x @ W^T + b (weight transpose)
- Kaiming uniform initialization for ReLU-family activations
- Parameter counting: out_features × in_features

---

### 3. **tokenizer_class** (New Module)

**Literature Sources:**
- `subword_tokenization/Neural_Machine_Translation_of_Rare_Words_with_Subword_Units_Analysis.md`
- `subword_tokenization/Formalizing_BPE_Tokenization_Analysis.md`

**What to Verify:**
- BPE for open-vocabulary NMT (Sennrich et al. 2016)
- Formal semantics of merge application order (Berglund & van der Merwe 2023)
- Greedy algorithm and rule priority

**Key Claims in Build Prompt:**
- Sequential merge application (order matters)
- Byte-level BPE for universal coverage
- Reverse vocabulary for O(1) encoding lookup

---

### 4. **transformer_lm** (New Module)

**Literature Source:**
- `transformer_paradigm/Attention_Is_All_You_Need_Analysis.md`

**What to Verify:**
- Layer stacking (N encoder/decoder blocks)
- Pre-norm vs post-norm architecture
- Final normalization before output projection
- Autoregressive language modeling setup

**Key Claims in Build Prompt:**
- Token embeddings → N blocks → final norm → LM head
- Pre-norm architecture for stability
- Tied embeddings save parameters

---

### 5. **data_loader** (New Module)

**Literature Sources:**
- `practical_implementation_guides/CS336_Assignment_1_Analysis.md` (primary)
- `training_mechanisms_understanding/Training_Dynamics_Underlying_Language_Model_Scaling_Laws_Analysis.md` (conceptual)

**What to Verify:**
- Random sampling vs sequential processing
- Context window sampling strategy
- This is the enabling technology for scaling laws experiments

**Key Claims in Build Prompt:**
- Random sampling prevents positional overfitting
- Input/target shifted by 1 for next-token prediction
- Sampling range: [0, N - context_length - 1]

---

### 6. **checkpointing** (New Module)

**Literature Source:**
- `practical_implementation_guides/CS336_Assignment_1_Analysis.md`

**What to Verify:**
- MLOps best practice (not research contribution)
- Must save: model state, optimizer state, iteration count
- Device-agnostic loading (map_location='cpu')

**Key Claims in Build Prompt:**
- Optimizer state 2× model size (AdamW stores m and v)
- In-place restoration preserves references
- Optimal checkpoint frequency: ~2-3 hours for spot instances

---

## Existing Modules (Verification Reference)

### Foundation Components

| Module | Literature Source | Key Verification Points |
|:---|:---|:---|
| **softmax** | Attention_Is_All_You_Need | Core attention component, numerical stability via max subtraction |
| **cross_entropy** | CS336_Assignment_1 | Log-sum-exp trick for numerical stability |
| **gradient_clipping** | CS336_Assignment_1 | Global norm clipping preserves direction |

### Architecture Components

| Module | Literature Source | Key Verification Points |
|:---|:---|:---|
| **rmsnorm** | Root_Mean_Square_Layer_Normalization | "Re-centering is dispensable" - key innovation |
| **silu** | GLU_Variants_Improve_Transformer | SiLU (Swish) activation for SwiGLU |
| **swiglu** | GLU_Variants_Improve_Transformer | Gating mechanism, d_ff ≈ (8/3)d for parameter parity |
| **attention** | Attention_Is_All_You_Need | Q,K,V analogy, 1/√d_k scaling |
| **multihead_attention** | Attention_Is_All_You_Need | Multiple representation subspaces |
| **transformer_block** | Attention_Is_All_You_Need + RMSNorm | Residual connections, pre-norm architecture |

### Training Infrastructure

| Module | Literature Source | Key Verification Points |
|:---|:---|:---|
| **adamw** | CS336_Assignment_1 | Decoupled weight decay innovation (requires AdamW paper verification) |
| **cosine_schedule** | Training_Dynamics_Scaling_Laws | Industry standard (warmup + cosine decay) |
| **training_loop** | Training_Dynamics_Scaling_Laws | Forward → backward → step mechanics |

### Application

| Module | Literature Source | Key Verification Points |
|:---|:---|:---|
| **bpe_tokenizer** | Neural_Machine_Translation + Formalizing_BPE | Greedy merge selection, OOV solution |
| **text_generation** | CS336_Assignment_1 | Sampling strategies: greedy, temp, top-k, top-p |

---

## Verification Checklist for New Modules

Use this checklist to verify each new module systematically:

### Per Module:
- [ ] Read corresponding literature analysis file(s)
- [ ] Verify mathematical formulations match source papers
- [ ] Check historical context is accurate
- [ ] Validate implementation constraints align with literature
- [ ] Confirm pedagogical claims are grounded in research
- [ ] Verify code examples are correct
- [ ] Check justify questions test actual concepts from papers

### Priority 2 Modules Status:

| Module | Build Prompt | Justify Questions | Literature Verified | Status |
|:---|:---:|:---:|:---:|:---:|
| embedding | ✅ | ✅ | ⏳ | Pending verification |
| linear | ✅ | ✅ | ⏳ | Pending verification |
| tokenizer_class | ✅ | ✅ | ⏳ | Pending verification |
| transformer_lm | ✅ | ✅ | ⏳ | Pending verification |
| data_loader | ✅ | ✅ | ⏳ | Pending verification |
| checkpointing | ✅ | ✅ | ⏳ | Pending verification |

---

## Notes for Verification

### Critical Areas Requiring Primary Literature

The following modules reference papers not yet analyzed in the literature base:

1. **adamw** - Requires verification against Loshchilov & Hutter (2019)
   - "Decoupled weight decay" as key innovation
   - Difference from Adam + L2 regularization

### Recently Confirmed Sources

- **rope** – Verified analysis available at:
  `/Users/tomriddle1/Holistic-Performance-Enhancement/cultivation/docs/5_domain_knowledge_and_curricula/computer_science/architectures_and_models/transformer_paradigm/RoFormer_Analysis.md`
  - Covers Su et al. (2021) "RoFormer" foundations, including rotational relative position encoding and extrapolation behavior.

### Modules with Strong Literature Coverage

These modules have direct 1:1 mappings to comprehensive analyses:

- ✅ **rmsnorm** - Complete RMSNorm paper analysis available
- ✅ **swiglu** - Complete GLU Variants paper analysis available  
- ✅ **attention** - Complete Attention Is All You Need analysis available
- ✅ **bpe_tokenizer** - Dual coverage (original + formal semantics)

---

## Verification Workflow

For systematic verification:

1. **Select module** to verify
2. **Read literature source(s)** from the mapping above
3. **Open build prompt**: `curricula/cs336_a1/modules/{module_id}/build_prompt.txt`
4. **Cross-reference**:
   - Mathematical formulations
   - Historical claims
   - Implementation rationale
   - Performance characteristics
5. **Check justify questions**: Verify model answers cite correct concepts
6. **Document findings**: Note any discrepancies or needed corrections
7. **Mark verified**: Update checklist above

---

## Quality Standard

All curriculum content must be:
- **Grounded** in primary literature
- **Accurate** in mathematical detail
- **Contextualized** historically
- **Pedagogically sound** in explanation

This verification ensures the curriculum maintains research integrity while being accessible to students.

---

**Last Updated**: 2025-11-12 (after Priority 2 completion)  
**Modules Requiring Verification**: 6 new modules  
**Status**: Ready for systematic literature cross-reference
