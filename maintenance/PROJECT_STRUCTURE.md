# Project Structure

This document describes the organization of the Mastery Engine repository.

## Dual-Mode Architecture

This repository supports **two modes** for different workflows:

- **Student Mode**: TODO stubs, tests fail until implemented
- **Developer Mode**: Complete implementations, tests pass

Switch modes with: `./scripts/mode switch student|developer`

## Root Directory

```
assignment1-basics/
├── README.md                      # Project overview and quick start
├── LICENSE                        # MIT License
├── pyproject.toml                # Python project configuration
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
│
├── modes/                        # Source of truth for each mode (git tracked)
│   ├── student/                  # Student mode (TODO stubs)
│   │   └── cs336_basics/
│   │       └── utils.py          # TODO: Implement softmax, cross_entropy, gradient_clipping
│   └── developer/                # Developer mode (complete implementations)
│       └── cs336_basics/
│           └── utils.py          # Complete implementations
│
├── cs336_basics/                 # Active workspace (symlink to modes/{mode}/cs336_basics/)
│   │                             # This directory is gitignored and derived
│   ├── __init__.py
│   ├── utils.py                  # Content depends on active mode
│   └── layers.py                 # Pre-implemented layers (same in both modes)
│
├── engine/                       # Mastery Engine core
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── state.py                  # Progress tracking
│   ├── curriculum.py             # Curriculum management
│   ├── workspace.py              # Shadow worktree isolation
│   ├── validator.py              # Test execution
│   ├── schemas.py                # Data models
│   ├── stages/                   # BJH loop stages
│   │   ├── build.py
│   │   ├── justify.py
│   │   └── harden.py
│   └── services/                 # External services
│       └── llm_service.py        # LLM integration
│
├── curricula/                    # Curriculum content
│   └── cs336_a1/                 # CS336 Assignment 1 curriculum
│       ├── manifest.json         # Curriculum metadata
│       ├── modules/              # Individual modules
│       │   ├── softmax/
│       │   │   ├── build_prompt.txt
│       │   │   ├── justify_questions.json
│       │   │   ├── validator.sh
│       │   │   └── bugs/
│       │   ├── cross_entropy/
│       │   └── gradient_clipping/
│       └── reference/            # Complete implementations (archived)
│           └── utils_complete.py
│
├── tests/                        # Test suite
│   ├── test_nn_utils.py          # Unit tests for cs336_basics
│   ├── adapters.py               # Test adapters
│   ├── engine/                   # Engine unit tests
│   ├── integration/              # LLM integration tests
│   └── e2e/                      # End-to-end tests
│
├── docs/                         # Documentation
│   ├── architecture/             # System design documents
│   │   ├── MASTERY_ENGINE.md    # Full architecture specification
│   │   └── REPO_ANALYSIS.md     # Codebase analysis
│   ├── development/              # Development logs
│   │   ├── MASTERY_WORKLOG.md   # Engine development log
│   │   ├── WORKLOG.md           # Original assignment worklog
│   │   ├── IMPLEMENTATION_PLAN.md
│   │   ├── MVP_COMPLETION_STATUS.md
│   │   └── CHANGELOG.md
│   ├── assignment/               # Original assignment materials
│   │   └── cs336_spring2025_assignment1_basics.pdf
│   └── LLM_PROMPT_REVIEW.md     # LLM prompt analysis
│
├── scripts/                      # Utility scripts
│   └── make_submission.sh
│
└── .mastery_engine_worktree/     # Isolated testing environment (gitignored)
    ├── cs336_basics/             # Copy of student code under test
    ├── tests/                    # Copy of test suite
    └── workspace/
        └── harden/               # Bug-injected code for debugging
```

## Key Directories Explained

### `cs336_basics/` - Student Workspace
- **Purpose**: Where students implement their code
- **Current State**: TODO stubs (raise NotImplementedError)
- **Do NOT edit**: Helper functions already implemented

### `engine/` - Mastery Engine Core
- **Purpose**: Powers the Build-Justify-Harden learning loop
- **Architecture**: CLI → State Manager → Curriculum → Validator → LLM
- **Key Files**:
  - `main.py`: Typer CLI application
  - `schemas.py`: Pydantic models for data validation
  - `stages/`: BJH pedagogical workflow

### `curricula/` - Curriculum Content
- **Purpose**: Modular curriculum definitions
- **Structure**: Each module has prompts, questions, validator, and bugs
- **Format**: JSON manifests + Markdown prompts + bash validators

### `tests/` - Test Suite
- **Unit Tests**: `test_nn_utils.py` validates student implementations
- **Integration Tests**: `integration/test_llm_service.py` validates LLM
- **E2E Tests**: `e2e/test_complete_bjh_loop.py` validates full workflow

### `docs/` - Documentation
- **architecture/**: System design and analysis
- **development/**: Development logs and planning
- **assignment/**: Original CS336 materials

### `.mastery_engine_worktree/` - Shadow Worktree
- **Purpose**: Isolated environment for safe testing
- **Lifecycle**: Created by `engine init`, cleaned by `engine cleanup`
- **Do NOT edit**: Automatically managed by the engine

## Mode Management

### Check Current Mode
```bash
./scripts/mode status
```

### Switch Modes
```bash
# Switch to student mode (TODO stubs)
./scripts/mode switch student

# Switch to developer mode (complete implementations)
./scripts/mode switch developer
```

### Test in Specific Mode
```bash
# Test in student mode without permanently switching
./scripts/mode test student "uv run pytest tests/test_nn_utils.py"

# Test student experience temporarily
./scripts/mode test student "uv run python -m engine.main next"
```

### Important Mode Rules

- **cs336_basics/ is derived**: Never commit this directory - it's a symlink
- **Edit in modes/**: Source of truth is `modes/student/` or `modes/developer/`
- **Tests reflect mode**: Student mode tests FAIL (TODO stubs), Developer mode tests PASS

## Development Workflow

### As a Developer (Testing Engine Features)
1. **Switch to developer mode**: `./scripts/mode switch developer`
2. **Initialize**: `engine init cs336_a1`
3. **Run tests**: `uv run pytest tests/engine/`
4. **Develop features**: Edit engine code
5. **Verify**: Tests pass with complete implementations

### As a Student (Testing Pedagogy)
1. **Switch to student mode**: `./scripts/mode switch student`
2. **View Prompt**: `engine next`
3. **Implement**: Edit `modes/student/cs336_basics/utils.py` (or just `cs336_basics/utils.py`)
4. **Submit**: `engine submit-build`
5. **Justify**: `engine submit-justification "<answer>"`
6. **Debug**: `engine submit-fix`
7. **Progress**: `engine status`

### Hybrid Workflow (Recommended for Development)
```bash
# Stay in developer mode for engine work
./scripts/mode switch developer

# Test student experience temporarily when needed
./scripts/mode test student "uv run python -m engine.main submit-build"

# No need to switch back - already in developer mode
```

## Configuration

- **Environment Variables**: Copy `.env.example` to `.env` and configure
  - `OPENAI_API_KEY`: Required for LLM-powered justify stage
  - `MASTERY_DISABLE_FAST_FILTER`: Debug flag (default: false)

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/test_nn_utils.py      # Unit tests
uv run pytest tests/engine/               # Engine tests
uv run pytest tests/integration/ -m integration  # LLM tests (requires API key)
uv run pytest tests/e2e/                  # E2E tests
```

## Git Workflow

**Main Branch**: Stable, production-ready code
**Feature Branches**: All development happens here

**Commit Convention**:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation
- `refactor:` Code restructuring
- `test:` Test additions/changes

## Notes

- **Reference Implementations**: Complete solutions archived in `curricula/cs336_a1/reference/`
- **Shadow Worktree**: Provides isolation - changes never affect main workspace
- **State File**: `~/.mastery_progress.json` tracks user progress (gitignored)
