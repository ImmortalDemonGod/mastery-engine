# Mastery Engine Documentation

Welcome to the Mastery Engine documentation. This is a **pedagogical operating system** that combines runtime AST mutation, shadow worktree isolation, and LLM-powered evaluation to create an interactive learning environment.

## 📚 Documentation Map

### For Users & Portfolio Reviewers

**Start here:**
- **[Main README](../README.md)** - Quick start, features, and installation
- **[Architecture](./architecture/)** - System design and technical deep-dive
- **[User Guide](./user-guide/)** - Command reference and workflows

### For Contributors & Maintainers

**Development documentation:**
- **[Internal Docs](./internal/)** - Development logs, historical documentation
- **[INDEX.md](./INDEX.md)** - Complete file navigation

## 🏗️ Core Architecture

### The Three Engineering Pillars

1. **Runtime AST Mutation**
   - Parses correct Python code into Abstract Syntax Trees
   - Surgically injects semantic bugs (not syntax errors)
   - Implementation-agnostic: works on ANY correct solution
   - Evidence: `engine/ast_harden/generic_injector.py`

2. **Shadow Worktree Isolation**
   - Creates ephemeral git worktrees for bug validation
   - Ensures idempotency and prevents dirty state
   - System programming + git internals
   - Evidence: `engine/workspace.py`

3. **Cost-Optimized Validation**
   - Fast keyword filter ($0 cost) catches shallow answers
   - LLM escalation ($0.003 cost) for deep semantic verification
   - System design + AI engineering
   - Evidence: `engine/services/llm_service.py`

## 🎓 The Build-Justify-Harden Loop

**Build** → Implement the solution (test validation)
**Justify** → Explain your understanding (LLM evaluation)  
**Harden** → Debug a bug in YOUR code (real-world scenario)

This cycle ensures:
- ✅ Functional competence (you can code)
- ✅ Conceptual understanding (you can explain)
- ✅ Debugging intuition (you can fix production issues)

## 📖 Quick Navigation

### I want to...

**...use the Mastery Engine**
→ Read the [Main README](../README.md) and [User Guide](./user-guide/)

**...understand the architecture**
→ Read [Architecture Docs](./architecture/)

**...contribute code**
→ Check [Internal Development Docs](./internal/development/)

**...create curriculum content**
→ See [Internal Bug Injection Guide](./internal/current/BUG_INJECTION_GUIDE.md)

## 🎯 System Status

**Version:** 1.0 Production  
**Test Coverage:** 78% (145/145 passing)  
**Curricula:** 4 (cs336_a1, cp_accelerator, python_for_cp, job_prep_data_annotation)

**Platform Requirements:** Linux, macOS, or Windows WSL2
- Requires: `bash`, `git`, symlink support
- See main README for installation

## 📦 Repository Structure

```
docs/
├── README.md           ← You are here
├── INDEX.md            ← Complete file index
├── architecture/       ← System design (START for technical review)
├── user-guide/         ← CLI reference and workflows
└── internal/           ← Development logs (optional reading)
```

## 🔗 External Resources

- **GitHub:** https://github.com/ImmortalDemonGod/mastery-engine
- **License:** MIT (see root LICENSE file)
- **Attribution:** Engine is original work; curricula adapted from educational sources

---

**For a comprehensive technical review, start with [Architecture](./architecture/).**
