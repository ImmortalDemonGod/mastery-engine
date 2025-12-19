# Mastery Engine: Pedagogical Operating System

**A Python-based platform for interactive technical mastery via AST mutation and shadow-worktree isolation.**

[![Tests](https://github.com/ImmortalDemonGod/mastery-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/ImmortalDemonGod/mastery-engine/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Test Coverage](https://img.shields.io/badge/coverage-78%25-green.svg)](https://github.com/ImmortalDemonGod/mastery-engine)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL2-lightgrey.svg)](https://github.com/ImmortalDemonGod/mastery-engine)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start (Demo Mode)

Experience the engine immediately without writing code. This demo uses **Developer Mode** (pre-loaded reference implementations) to simulate a user solving the curriculum.

```bash
# 1. Clone and Enter Repository
git clone https://github.com/ImmortalDemonGod/mastery-engine.git
cd mastery-engine

# 2. Activate Developer Mode (REQUIRED: Creates cs336_basics symlink)
./scripts/mode switch developer

# 3. Install Dependencies
pip install uv
uv sync
uv pip install -e .

# 4. Initialize a Curriculum
uv run mastery init cs336_a1

# 5. View Current Module
uv run mastery show
# Output: Displays build prompt with problem description

# 6. Run Build Validation (Watch the engine validate the reference implementation)
uv run mastery submit
# Output: ✅ Validation Passed!
#         Performance: <seconds>

# 7. Answer Justify Questions (Auto-passes in mock mode without OpenAI key)
uv run mastery submit
# Output: 🎭 MOCK MODE: Auto-passing justify stage
#         (Panel explains mock mode and next steps)

# 8. Inject Runtime Bug (Watch the engine mutate the code)
uv run mastery start-challenge
# Output: 💉 Injecting semantic bug '<bug_id>'...
#         (Bug is selected randomly from the module's bugs directory)

# 9. See the Bug Fail
uv run mastery submit
# Output: ❌ Fix Incomplete
#         (Pytest output shows the failing assertion for the injected bug)
```

**What Just Happened?**  
You witnessed the complete Build-Justify-Harden loop:
1. **Build**: Validated a working implementation  
2. **Justify**: Evaluated conceptual understanding (mocked without API key)  
3. **Harden**: Injected a semantic bug and detected the failure

No OpenAI API key required for demo - the engine operates in mock mode automatically.

---

## 🏗️ Key Engineering Features

| Feature | Description | Implementation |
| :--- | :--- | :--- |
| **Runtime AST Mutation** | Parses Python code into an Abstract Syntax Tree, identifies semantic patterns (e.g., loop termination conditions), and surgically injects logic bugs—not just syntax errors. Students debug realistic mistakes. | [`engine/ast_harden/generic_injector.py`](engine/ast_harden/generic_injector.py) |
| **Process Isolation via Shadow Worktrees** | Solves the "dirty state" problem by executing user code in ephemeral Git shadow worktrees. Each harden challenge runs in an isolated filesystem copy, preventing data corruption and enabling safe rollback. | [`engine/workspace.py`](engine/workspace.py) |
| **Socratic LLM Evaluation** | Uses GPT-4o with Chain-of-Thought prompting to evaluate natural language justifications against technical rubrics. Detects surface-level answers via keyword filtering before calling the LLM. | [`engine/services/llm_service.py`](engine/services/llm_service.py) |
| **Automated Content Pipeline** | Generated 38+ competitive programming problems by parsing unstructured LeetCode data into structured JSON schemas. Demonstrates data engineering at scale. | [`scripts/generate_module.py`](scripts/generate_module.py) |
| **Curriculum-Agnostic Architecture** | Supports multiple curricula (Deep Learning, Competitive Programming, Job Prep) with both LINEAR (sequential) and LIBRARY (freeform) modes. Curricula are hot-swappable via manifest files. | [`engine/curriculum.py`](engine/curriculum.py) |

---

## 📚 Included Curricula

### 1. **cs336_a1** - [Stanford CS336](https://stanford-cs336.github.io/spring2025/): Language Modeling (LINEAR)
- **Target**: Deep learning practitioners building transformers from scratch
- **Modules**: 21 modules (BPE Tokenizer → Full Training Loop)
- **Tech Stack**: PyTorch, einops, tiktoken
- **Completion Time**: ~60 hours

### 2. **cp_accelerator** - Competitive Programming (LIBRARY)
- **Target**: Software engineers preparing for algorithmic interviews  
- **Problems**: 38 LeetCode problems across 19 patterns
- **Demonstration**: Showcases automated content generation pipeline  
- **Status**: Breadth-first scaffolding complete

### 3. **job_prep_data_annotation** - DataAnnotation Assessment Prep (LINEAR)
- **Target**: Candidates preparing for coding assessments  
- **Modules**: 3 modules (HTTP, HTML Parsing, 2D Grids)
- **Focus**: Forensic pedagogy - each module targets a documented failure mode  
- **Completion Time**: ~3 hours

### 4. **python_for_cp** - Python Standard Library for CP (LINEAR, Partial)
- **Target**: Competitive programmers learning Python idioms  
- **Module**: `std_lib_augmentation` (deque, heapq, bisect)  
- **Status**: Priority module complete

---

## 🎓 The Build-Justify-Harden Loop

The engine's pedagogical core is a three-stage loop designed for deep, resilient mastery:

### 1. **BUILD** - Implementation
Write code that passes functional tests.

```bash
uv run mastery submit
# → Runs validator.sh, checks correctness and performance
```

### 2. **JUSTIFY** - Conceptual Understanding
Answer Socratic questions to prove you understand *why* your code works.

```bash
uv run mastery submit
# → Opens $EDITOR for natural language responses
# → LLM evaluates depth of understanding (or auto-passes in mock mode)
```

### 3. **HARDEN** - Debugging Resilience
Debug a bug that was surgically injected into *your* correct code via AST transformation.

```bash
uv run mastery start-challenge
# → Creates shadow worktree with mutated code
# → Common mistakes: off-by-one, missing base case, type confusion

uv run mastery submit
# → Validates the fix in isolated environment
```

**Why This Works:**  
- **Build** ensures functional competence  
- **Justify** prevents "interview syndrome" (can code but can't explain)  
- **Harden** builds debugging intuition for production systems

---

## 🔬 Technical Architecture

### Shadow Worktree System
```
Main Repository
├── modes/student/      ← Your workspace (stubs)
└── modes/developer/    ← Reference implementations

Shadow Worktree (ephemeral)
└── .mastery_engine_worktree/
    └── modes/student/  ← Isolated copy with injected bug
```

**Workflow:**
1. User completes build + justify in main repo  
2. `start-challenge` creates shadow worktree from main repo  
3. AST injector mutates code in shadow copy  
4. User debugs in shadow worktree  
5. On success, shadow is deleted (clean slate for next bug)

### AST Mutation Example

**Original Code:**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:  # ✅ Correct termination
        ...
```

**After Injection (off_by_one bug):**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:  # ❌ Misses single-element case
        ...
```

The AST injector:
1. Parses code into syntax tree  
2. Locates `Compare` nodes with `<=` operator  
3. Replaces with `<` operator  
4. Regenerates Python code

Result: **Syntactically valid but semantically broken code.**

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.11+**
- **uv** (fast Python package manager)
- **Git**

### ⚠️ System Requirements

**Platform:** Linux, macOS, or Windows WSL2  
**Why?** This engine uses POSIX features (`bash` scripts, symlinks via `ln -s`, `git worktree`).

<details>
<summary><b>Windows Users:</b> Install WSL2 (click to expand)</summary>

```powershell
# PowerShell (as Administrator)
wsl --install
wsl --set-default-version 2
```

Then continue installation inside WSL.
</details>

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/ImmortalDemonGod/mastery-engine.git
cd mastery-engine

# Install dependencies
pip install uv
uv sync

# Install in editable mode
uv pip install -e .

# Verify installation
uv run mastery --version
```

### Optional: OpenAI API Key (for LLM evaluation)
```bash
# Set in environment
export OPENAI_API_KEY="sk-..."

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

**Note:** The engine operates in mock mode without an API key (auto-passes justify stage).

---

## 🎮 Usage

### Basic Workflow

```bash
# 1. Choose a mode
./scripts/mode switch student      # Start with stubs (typical user)
./scripts/mode switch developer    # Use reference implementations (demo)

# 2. Initialize a curriculum
uv run mastery init cs336_a1

# 3. View current assignment
uv run mastery show

# 4. Work through the BJH loop
uv run mastery submit     # Build stage
uv run mastery submit     # Justify stage
uv run mastery start-challenge  # Harden stage (injects bug)
uv run mastery submit     # Fix the bug
```

### Advanced Commands

```bash
# List all modules in curriculum
uv run mastery curriculum-list

# Reset a specific module
uv run mastery progress-reset <module_id>

# Check current status
uv run mastery status

# Cleanup artifacts
uv run mastery cleanup
```

---

## 🧪 Running Tests

The engine has 78% test coverage (145 passing tests):

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=engine --cov-report=html

# Run specific test file
uv run pytest tests/engine/test_curriculum.py -v
```

### Test Categories
- **Unit Tests**: Engine components (`test_curriculum.py`, `test_workspace.py`)  
- **Integration Tests**: Full workflows (`test_main_workflows_real.py`)  
- **E2E Tests**: Error handling (`test_error_handling.py`)

---

## 📂 Project Structure

```
mastery-engine/
├── engine/                   # Core engine
│   ├── curriculum.py         # Curriculum loading and validation
│   ├── workspace.py          # Shadow worktree management
│   ├── main.py               # CLI entry point
│   ├── ast_harden/           # AST mutation engine
│   │   └── generic_injector.py
│   └── services/
│       └── llm_service.py    # LLM evaluation (with mock mode)
├── curricula/                # Curriculum content
│   ├── cs336_a1/             # Deep learning curriculum
│   ├── cp_accelerator/       # Competitive programming (LIBRARY mode)
│   ├── job_prep_data_annotation/  # Job prep curriculum
│   └── python_for_cp/        # Python for CP (partial)
├── modes/
│   ├── student/              # Stub implementations
│   └── developer/            # Reference solutions
├── scripts/
│   ├── mode                  # Mode switcher script
│   ├── generate_module.py    # Content generation pipeline
│   └── generate_manifest.py  # Manifest generator
├── tests/                    # Test suite (145 tests, 78% coverage)
└── docs/                     # Documentation
```

---

## 🎯 Design Philosophy

### 1. **No Hand-Waving**
Every claim is executable. "You'll learn X" → "Run this validator to prove you learned X."

### 2. **Fail Fast, Learn Deep**
The engine aggressively validates at each stage. Passing justify ≠ memorization; it requires explaining trade-offs.

### 3. **Production-Realistic Bugs**
Harden bugs simulate real mistakes (not contrived errors). Example: using `list.pop(0)` in BFS instead of `deque.popleft()` - syntactically correct, O(n²) performance trap.

### 4. **Curriculum Agnostic**
The engine doesn't care if you're learning transformers or competitive programming. Curricula are data, not code.

### 5. **Forensic Pedagogy**
`job_prep_data_annotation` was designed by analyzing actual assessment failures and creating modules that surgically address each failure mode.

---

## 🤝 Contributing

See [`docs/current/BUG_INJECTION_GUIDE.md`](docs/current/BUG_INJECTION_GUIDE.md) for curriculum authoring guidelines.

**Adding a New Curriculum:**
1. Create `curricula/<name>/manifest.json`  
2. Define modules with build prompts, validators, justify questions  
3. Create harden bugs as `.patch` files  
4. Test with `uv run mastery init <name>`

---

## 📜 License & Attribution

This project is a dual-composition of original systems engineering and adapted educational content.

### The Mastery Engine (Original Work)

The core architecture is **original engineering work** licensed under the **MIT License**:
- **CLI Orchestration:** Command-line interface and workflow management (`engine/main.py`)
- **AST Mutation Engine:** Runtime code transformation system (`engine/ast_harden/`)
- **Shadow Worktree System:** Process isolation via git internals (`engine/workspace.py`)
- **Validation Pipeline:** Test harness and LLM evaluation (`engine/validator.py`, `engine/services/`)
- **Curriculum Generation:** Automated content pipeline (`scripts/generate_module.py`)

### Curriculum Content (Adapted)

Educational materials are adapted from the following sources:

- **Deep Learning (cs336_a1):** Problem statements and mathematical specifications adapted from **[Stanford CS336 (Spring 2024)](https://stanford-cs336.github.io/spring2024/)**. Reference implementations derive from course assignments. The "Justify" questions and "Harden" bug patterns were engineered specifically for this platform.

- **Competitive Programming (cp_accelerator):** Problem taxonomies derived from **[LeetCode](https://leetcode.com)** and **[DSA Taxonomies](https://github.com/Yassir-aykhlf/DSA-Taxonomies)**. Problem descriptions remain property of respective authors.

- **Python Basics (python_for_cp, job_prep_data_annotation):** Concepts adapted from **[30 Days of Python](https://github.com/Asabeneh/30-Days-Of-Python)** by Asabeneh Yetayeh (MIT License).

**See individual curriculum READMEs for detailed attribution.**

### What This Means

**Our Contribution:** The engineering achievement is the **system architecture** that enables runtime AST mutation, shadow worktree isolation, and cost-optimized LLM validation—not the specific curriculum problems.

**License:** See [LICENSE](LICENSE) and [NOTICE](NOTICE) files for complete terms.

---

## 📞 Contact

**GitHub:** [@ImmortalDemonGod](https://github.com/ImmortalDemonGod)  
**Project:** [mastery-engine](https://github.com/ImmortalDemonGod/mastery-engine)  
**Issues:** [Report bugs or request features](https://github.com/ImmortalDemonGod/mastery-engine/issues)

---

**Built with Python 🐍 | Powered by uv ⚡ | Engineered for Mastery 🎯**
