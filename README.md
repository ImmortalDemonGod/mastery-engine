# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Mastery Engine Workflow

This assignment includes an interactive learning engine. Use the unified `engine submit` command for all stages:

```bash
# Initialize and start
engine init
engine status
engine next

# Build-Justify-Harden loop - single command!
engine submit  # Auto-detects your current stage (build/justify/harden)

# Check progress anytime
engine status
```

The `submit` command automatically:
- Validates your implementation in the **build** stage
- Opens your editor for answers in the **justify** stage
- Tests your bug fix in the **harden** stage

**Note**: Legacy commands (`submit-build`, `submit-justification`, `submit-fix`) still work but `submit` is recommended.

## Understanding the Harden Stage: Bug Injection Architecture

**CRITICAL**: The Harden stage injects bugs into YOUR correct code at runtime, not pre-written buggy files.

### The Two-Tier Bug System

Every bug has TWO files:

1. **`.patch` file** (SOURCE OF TRUTH)
   - Human-readable diff showing the code transformation
   - Version-controlled and maintained long-term
   - Created with standard `diff -u correct.py buggy.py > bug.patch`
   - **Schema-independent**: Survives engine upgrades
   - **Primary artifact**: The durable definition of what the bug does

2. **`.json` file** (COMPILED RUNTIME ARTIFACT)
   - AST-based declarative pattern for `GenericBugInjector`
   - Auto-generated from `.patch` using LLM tool: `engine create-bug module --patch bug.patch`
   - Can be regenerated anytime (not sacred)
   - **Schema-dependent**: Tied to current engine version
   - **Runtime consumption**: Used by engine to inject bug into your code

### How It Works

```
Your correct code → GenericBugInjector reads .json → Injects bug → You debug YOUR buggy code
```

**Why this matters:**
- You're debugging code YOU wrote, making the experience realistic
- Bugs are defined once (`.patch`) but work on any correct implementation
- Engine upgrades just regenerate `.json` from durable `.patch` files
- No AST knowledge required to author bugs (just write buggy code and diff it)

### Creating a Bug (For Curriculum Authors)

```bash
# 1. Write correct and buggy versions
vim correct_solution.py
vim buggy_solution.py

# 2. Create the .patch (source artifact)
diff -u correct_solution.py buggy_solution.py > bugs/my_bug.patch

# 3. Generate the .json (compiled artifact) 
engine create-bug module_name --patch bugs/my_bug.patch --output bugs/my_bug.json

# 4. Write symptom description
echo "Runtime Error: Output has missing elements" > bugs/my_bug_symptom.txt

# Done! Engine will inject this bug into student's correct code during Harden stage
```

The LLM tool may fail for complex patterns (control flow, statement deletion). In those cases, manually write the `.json` by studying examples in `curricula/cs336_a1/modules/*/bugs/*.json`.

### Example: Statement Deletion Bug

```python
# .patch shows: Delete result.extend(right[j:])

# .json contains AST pattern:
{
  "pattern": {
    "node_type": "Expr",
    "value": {
      "node_type": "Call",
      "func": {"node_type": "Attribute", "attr": "extend"}
    }
  },
  "replacement": {"type": "delete_statement"}
}
```

The engine finds this pattern in YOUR correct code and deletes it, creating the buggy version you'll debug.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

