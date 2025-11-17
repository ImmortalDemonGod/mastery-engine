# Mastery Engine Documentation

**Production-ready interactive learning system** implementing Build-Justify-Harden pedagogy.

## Quick Start

**Choose your path:**

### 👨‍🎓 I'm a Student
Read the main [`/README.md`](../README.md) for setup and workflow.

**TL;DR**:
```bash
engine init           # Start curriculum
engine status         # Check progress
engine submit         # Submit work (auto-detects stage)
```

Full command reference: [`current/CLI_GUIDE.md`](./current/CLI_GUIDE.md)

### 📝 I'm Creating Curriculum Content
Read [`current/BUG_INJECTION_GUIDE.md`](./current/BUG_INJECTION_GUIDE.md) for the bug system.

**TL;DR**: Bugs are defined as:
1. `.patch` file (source of truth, durable)
2. `.json` file (compiled for runtime, regenerable)
3. `_symptom.txt` file (student-facing description)

Create with: `engine create-bug module --patch bugs/my_bug.patch`

### 💻 I'm Contributing Code
Start with these:

1. **Architecture**: [`architecture/MASTERY_ENGINE.md`](./architecture/MASTERY_ENGINE.md)
2. **Project Status**: [`development/MVP_COMPLETION_STATUS.md`](./development/MVP_COMPLETION_STATUS.md)
3. **Test Coverage**: [`current/TEST_COVERAGE_REPORT.md`](./current/TEST_COVERAGE_REPORT.md)

**System Health**:
- ✅ 78% test coverage (production-ready)
- ✅ 145/145 tests passing
- ✅ 2 curricula (22 modules total)
- ✅ Zero known critical bugs

### 🔧 I'm Maintaining the Project
Check these regularly:

1. **Curriculum Status**: [`current/CURRICULUM_STATUS.md`](./current/CURRICULUM_STATUS.md)
2. **Coverage Report**: [`current/TEST_COVERAGE_REPORT.md`](./current/TEST_COVERAGE_REPORT.md)
3. **Changelog**: [`development/CHANGELOG.md`](./development/CHANGELOG.md)
4. **Work Log**: [`development/WORKLOG.md`](./development/WORKLOG.md)

## Documentation Structure

```
docs/
├── INDEX.md                    # Full navigation (START HERE)
├── README.md                   # This file
│
├── current/                    # Canonical documentation
│   ├── CLI_GUIDE.md            # Command reference
│   ├── BUG_INJECTION_GUIDE.md  # Bug creation guide
│   ├── CURRICULUM_STATUS.md    # Module status
│   └── TEST_COVERAGE_REPORT.md # Coverage metrics
│
├── architecture/               # System design
│   ├── MASTERY_ENGINE.md       # Core architecture
│   └── REPO_ANALYSIS.md        # Codebase structure
│
├── development/                # For maintainers
│   ├── CHANGELOG.md            # Version history
│   ├── MVP_COMPLETION_STATUS.md# Project status
│   └── WORKLOG.md              # Development log
│
├── coverage/                   # Test coverage
│   └── html/                   # Interactive reports
│
└── archive/                    # Historical sessions
    └── sessions/               # Organized by date
```

## Core Concepts

### The Build-Justify-Harden Loop

**Build**: Implement the solution (validated by tests)  
**Justify**: Explain your understanding (evaluated by LLM)  
**Harden**: Debug a bug injected into YOUR correct code

This pedagogical cycle ensures:
- ✅ Working implementation (build)
- ✅ Deep understanding (justify)
- ✅ Debugging skill (harden)

### Runtime Bug Injection

**Critical**: Bugs are injected into YOUR code at runtime, not pre-written buggy files.

```
Your correct code → GenericBugInjector.inject(bug.json) → Buggy version → You debug
```

**Why?**
- More realistic (debug code YOU wrote)
- Works on ANY correct implementation
- Teaches debugging YOUR own mistakes

See [`current/BUG_INJECTION_GUIDE.md`](./current/BUG_INJECTION_GUIDE.md) for details.

### Shadow Worktree

The harden stage creates a shadow git worktree:
- Isolates buggy code from your main workspace
- Preserves your correct implementation
- Safe experimentation environment

## Key Features

✅ **Context-aware CLI** - `submit` auto-detects your stage  
✅ **LLM-powered evaluation** - Deep understanding verification  
✅ **AST-based bug injection** - Robust, implementation-agnostic  
✅ **Shadow worktree safety** - Isolates debugging environment  
✅ **78% test coverage** - Production-ready quality

## System Status

**Version**: 1.0 (Production MVP)  
**Status**: ✅ Ready for production deployment  
**Last Updated**: 2025-11-12

**Current Curricula**:
- cs336_a1: 21 modules (Stanford CS336 Deep Learning)
- cp_accelerator: 1 module (Competitive Programming pilot)

**Quality Metrics**:
- Test coverage: 78% (excellent)
- Test pass rate: 100% (145/145)
- Curricula quality: 98/100 (cs336_a1), 95/100 (cp_accelerator)

## Getting Help

**Questions about**:
- **Using the system**: See [`current/CLI_GUIDE.md`](./current/CLI_GUIDE.md)
- **Creating bugs**: See [`current/BUG_INJECTION_GUIDE.md`](./current/BUG_INJECTION_GUIDE.md)
- **Architecture**: See [`architecture/MASTERY_ENGINE.md`](./architecture/MASTERY_ENGINE.md)
- **Everything else**: See [`INDEX.md`](./INDEX.md) for full navigation

**Found a bug?** Open an issue on GitHub.

## Contributing

See [`development/WORKLOG.md`](./development/WORKLOG.md) for current work and roadmap.

**Key areas for contribution**:
- Curriculum expansion (more modules)
- Additional bug patterns
- Improved LLM evaluation
- UI/UX enhancements

## License

See main repository LICENSE file.
