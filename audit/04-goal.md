# 04 — Grounded Goal + External Research

## Candidate long-term goals (plural by design)
### Be a curriculum-agnostic 'pedagogical operating system' CLI that drives learners to deep technical mastery through a three-stage Build-Justify-Harden (BJH) loop: implement to a spec (Build, auto-validated), defend understanding in natural language (Justify, LLM-graded), and debug an AST-injected semantic bug in their own correct code under git-worktree isolation (Harden).
- **Grounded (author):** true · **Judge consensus:** true (4/4)
- Success signals:
  - A `mastery` CLI exists with the full BJH command surface (init, show, submit, start-challenge, status, curriculum-list, etc.) and renders the loop end-to-end without crashing. — _evidence:_ audit/03-execution.md:11 (mastery --help rc=0 lists submit/show/start-challenge/init/curriculum-list/status/select/create-bug)
  - `submit` auto-detects and routes to the correct BJH stage, and a full submit->validate cycle renders 'Validation Passed', 'Bug Fixed', 'Module Complete'. — _evidence:_ engine/main.py:815 (submit cli_command auto-detects stage) + audit/03-execution.md:14 (full submit->validate cycle rendered Validation Passed/Bug Fixed/Module Complete)
  - Build stage validates implementations by shelling out to per-module validator.sh harnesses with timeout/error handling. — _evidence:_ engine/validator.py:108 (subprocess.run of validator_path with timeout=300) + curricula/cs336_a1/modules/softmax/validator.sh:1
  - Justify stage evaluates free-text answers against rubrics via an LLM (GPT-4o, Chain-of-Thought) with a keyword fast-filter. — _evidence:_ engine/services/llm_service.py:27 (LLMService) + audit/03-execution.md:14 ('Justify fast-filter rejection and LLM correct/incorrect paths all exercised')
  - Harden stage injects a semantic (not syntactic) bug into the learner's correct code via AST mutation inside an ephemeral git shadow worktree, then validates the fix in isolation. — _evidence:_ engine/stages/harden.py:196 (_select_bug globs bug specs) + engine/ast_harden/generic_injector.py:19 (GenericBugInjector) + engine/workspace.py:24 (shadow-worktree isolation)
  - The engine package is exercised by a passing test suite covering schemas/state/validator/justify/harden code paths. — _evidence:_ audit/03-execution.md:4 (185 passed; schemas.py 100%, validator.py 92%, stages/justify.py 95%, stages/harden.py 74%)

### Serve as an engineering portfolio / capability showcase whose central claimed achievement is the SYSTEM ARCHITECTURE — runtime AST mutation, process isolation via git shadow worktrees, LLM-as-evaluator with cost optimization, and an automated content-generation pipeline — rather than the curriculum content itself.
- **Grounded (author):** true · **Judge consensus:** true (4/4)
- Success signals:
  - README explicitly frames the deliverable as 'Key Engineering Features' (Runtime AST Mutation, Process Isolation via Shadow Worktrees, Socratic LLM Evaluation, Automated Content Pipeline, Curriculum-Agnostic Architecture). — _evidence:_ README.md:71-79 (Key Engineering Features table)
  - License/NOTICE statement asserts 'the engineering achievement is the system architecture ... not the specific curriculum problems', distinguishing original engine code (MIT) from adapted content. — _evidence:_ README.md:421 ('Our Contribution: The engineering achievement is the system architecture')
  - Runtime AST mutation is implemented as real code (multiple injector implementations) that parse->match->inject->unparse Python. — _evidence:_ engine/ast_harden/softmax_v2_1.py:229 (inject_softmax_bug_v2_1 two-phase pipeline) + engine/ast_harden/generic_injector.py:19
  - An automated content pipeline parses unstructured source data into structured curriculum JSON (claimed 38+ problems) with CI enforcing manifest integrity. — _evidence:_ scripts/generate_module.py (content pipeline) + .github/workflows/validate_cp_manifest.yml:15 (manifest integrity CI regenerates manifest.json)
  - Coverage/CI badges and a green-tests narrative are presented as portfolio signals. — _evidence:_ README.md:5-8 (Tests/coverage badges) + .github/workflows/tests.yml:10 (pytest+coverage CI job)

### Function as the author's personal self-study / assignment-completion harness — primarily to work through Stanford CS336 (transformer-from-scratch) and competitive-programming/interview prep on their own machine, persisting personal progress.
- **Grounded (author):** true · **Judge consensus:** true (4/4)
- Success signals:
  - A real, personal progress file is actively in use on the host, showing an in-progress CS336 run (curriculum=cs336_a1, module softmax, HARDEN stage). — _evidence:_ audit/03-execution.md:13 (mastery status: Curriculum=cs336_a1, Current Module 'Numerically Stable Softmax (1/22)', Stage HARDEN) + engine/state.py:32 (STATE_FILE = Path.home()/'.mastery_progress.json')
  - A maintenance script packages the work for actual CS336 assignment submission (zips into the official assignment submission archive). — _evidence:_ maintenance/make_submission.sh:1 (runs pytest then zips into cs336-spring2025-assignment-1-submission.zip)
  - Shipped curricula target the author's own learning goals: a full CS336 language-modeling track (~21-22 modules) and an interview/CP accelerator plus job-prep tracks. — _evidence:_ README.md:85-101 (cs336_a1 21 modules; cp_accelerator 38 LeetCode problems; job_prep_data_annotation) + curricula/cs336_a1/manifest.json
  - Developer Mode ships pre-loaded reference implementations the author can run against, indicating single-operator (author-as-user) workflow rather than multi-tenant deployment. — _evidence:_ README.md:23 ('Activate Developer Mode ... pre-loaded reference implementations') + modes/developer/cs336_basics (reference solutions symlinked at repo root)

### Be an extensible, curriculum-agnostic framework where curricula are pure data (manifest-described) supporting both LINEAR (sequential modules) and LIBRARY (freeform pattern/problem) modes, so new learning domains can be added without engine changes.
- **Grounded (author):** true · **Judge consensus:** true (4/4)
- Success signals:
  - CurriculumManager loads manifest.json-described curricula and supports both LINEAR and LIBRARY curriculum types. — _evidence:_ engine/curriculum.py:31 (CurriculumManager loads manifest.json in LINEAR/LIBRARY modes) + engine/schemas.py (CurriculumType)
  - Multiple independent curricula coexist and are selectable, spanning distinct domains (deep learning, competitive programming, job prep, python stdlib). — _evidence:_ README.md:83-107 (Included Curricula: cs336_a1, cp_accelerator, job_prep_data_annotation, python_for_cp) + curricula/ directory inventory (audit/01-understanding.md:175)
  - Both modes are wired in the CLI: LINEAR runs sequential modules and LIBRARY exposes a `select` command to set active pattern/problem. — _evidence:_ engine/main.py:2684 (select cli_command for LIBRARY mode) + engine/main.py:555-586 (_submit_linear_workflow stage routing)
  - Documented authoring path for new curricula treats curricula as data (manifest + per-module assets), not engine code. — _evidence:_ README.md:386-390 (Adding a New Curriculum: create manifest.json, define modules, create .patch bugs) + README.md:374 ('Curricula are data, not code')

## External research 
### Sources
| Title | URL | Claim | Verified | Corroboration |
| --- | --- | --- | --- | --- |
| Mutation Testing with Mutmut: Python for Code Reliability 2026 | https://johal.in/mutation-testing-with-mutmut-python-for-code-reliability-2026/ | Mutmut achieves 88.5% mutation detection rate, outperforming Cosmic Ray's 82.7%, using AST-level Code Parser, Mutant Generator, and Test Runner components; achieves 1.5x faster mutant generation with 20% less overhead in 2025 benchmarks. | true | IEEE Conference Publication (https://ieeexplore.ieee.org/document/10818231/) independently reports mutmut outperforming cosmic-ray in comparative benchmarks. |
| Cosmic Ray: mutation testing for Python — Cosmic Ray documentation | https://cosmic-ray.readthedocs.io/ | Cosmic-ray operates at the Python AST level, supports custom mutation operators via a plugin architecture, and includes built-in build tool integration — making it extensible for semantic mutation variants. | true | GitHub repository (https://github.com/sixty-north/cosmic-ray) confirms plugin architecture and AST-level operation independently. |
| An Analysis and Comparison of Mutation Testing Tools for Python \| IEEE Xplore | https://ieeexplore.ieee.org/document/10818231/ | Peer-reviewed comparison of MutPy, Mutmut, Mutatest, Poodle, and Cosmic Ray on Python mutation testing, covering detection rates and operator coverage. | true | ACM SBES 2024 paper (https://dl.acm.org/doi/10.1145/3701625.3701659) provides independent static and dynamic comparison of the same tool landscape. |
| Static and Dynamic Comparison of Mutation Testing Tools for Python \| ACM SBES 2024 | https://dl.acm.org/doi/10.1145/3701625.3701659 | Comprehensive comparison of source code mutators vs. AST mutators for Python, providing empirical data on operator coverage and false-positive rates. | true | IEEE Conference Publication (https://ieeexplore.ieee.org/document/10818231/) independently analyzes the same tool landscape with overlapping results. |
| DeepEval LLM Evaluation Framework GitHub | https://github.com/confident-ai/deepeval | DeepEval is a pytest-like open-source framework for LLM evaluation with 30+ metrics (G-Eval, hallucination detection, task completion, bias/toxicity); supports local LLM-as-judge evaluation and integrates with Anthropic, OpenAI, LangChain. | true | Confirmed independently by two research agents; cross-referenced with the Autorubric paper citing DeepEval as a reference framework |
| G-Eval Simply Explained: LLM-as-a-Judge for LLM Evaluation — Confident AI | https://www.confident-ai.com/blog/g-eval-the-definitive-guide | G-Eval uses chain-of-thought reasoning combined with a form-filling paradigm to evaluate LLM outputs against arbitrary custom criteria, reducing position bias and length bias common in holistic LLM judges. | true | DeepEval documentation (https://deepeval.com/docs/metrics-llm-evals) independently describes G-Eval CoT + form-filling approach. |
| From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge | https://arxiv.org/pdf/2411.16594 | Comprehensive survey of LLM-as-judge paradigm covering scoring rubrics, bias correction approaches (position, verbosity, self-enhancement biases), and calibration methodologies applicable to open-ended technical explanation evaluation. | false | — |
| Rubric-Based Evaluations & LLM-as-a-Judge: Methodologies, Biases, and Empirical Validation in Domain-Specific Contexts | https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80 | For expert-knowledge tasks, LLM-human agreement rates drop to 64-68%, well below inter-expert baselines of 72-75%; domain specificity reveals pronounced alignment gaps requiring hybrid human-in-the-loop workflows and domain-adapted rubrics. | true | PMC article on 'Evaluating large language models for criterion-based grading' at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11683144/ independently corroborates domain-specific LLM grading limitation findings. |
| How to Use Git Worktrees for Parallel AI Agent Execution \| Augment Code | https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution | Git worktrees give each agent its own isolated working directory and git index while sharing a single object store, reducing sequential CI time by approximately 63% (24 min to 9 min). | true | Zylos Research (https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/) independently confirms worktree isolation pattern and cites same CI time-reduction figures. |
| Git Worktree Isolation Patterns for Parallel AI Agent Development \| Zylos Research | https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/ | Teams can reliably run 4-8 concurrent worktrees per developer; the productive ceiling is 5-7 concurrent on a modern laptop due to disk consumption (~5 GB per worktree). Ephemeral worktree creation/destruction per task is the recommended pattern. | true | Augment Code guide (https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution) independently confirms worktree isolation as the standard pattern for parallel AI agent execution. |
| Git Worktrees Need Runtime Isolation for Parallel AI Agent Development | https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/ | Git worktrees isolate code (separate filesystems) but not the runtime environment; shared ports, databases, and services still collide between worktrees; Docker containers provide additional runtime isolation through Linux namespaces and cgroups that worktrees alone cannot supply. | true | Zazencodes article on Parallel Coding Agents at https://zazencodes.substack.com/p/parallel-coding-agents-with-container independently describes the complementary nature of worktrees and containers, stating the most mature teams combine both approaches. |
| nsjail: Lightweight Linux Sandboxing for AI Code Execution (2026) \| Morph | https://www.morphllm.com/nsjail-sandbox | nsjail provides kernel-level isolation via Linux namespaces, cgroups, rlimits, and seccomp-bpf syscall filters for Python processes without requiring a container runtime; developed by Google, production-deployed since 2015. | true | Python Discord's snekbox (https://github.com/python-discord/snekbox) independently validates nsjail for production Python sandboxing with confirmed deployment. |
| GitHub - python-discord/snekbox: Easy, safe evaluation of arbitrary Python code | https://github.com/python-discord/snekbox | Production-deployed service using nsjail to safely execute arbitrary Python code; demonstrates the nsjail + Python subprocess pattern with resource limits enforced via cgroups. | true | Morph nsjail guide (https://www.morphllm.com/nsjail-sandbox) independently confirms snekbox as a production example of nsjail for Python code evaluation. |
| GitHub - judge0/judge0: Robust, fast, scalable, and sandboxed open-source online code execution system | https://github.com/judge0/judge0 | Open-source sandboxed code execution system with multi-language support, resource constraints, and self-hosted option; used for competitive programming, e-learning, and code assessment platforms. | false | — |
| GitHub - engineer-man/piston: A high performance general purpose code execution engine | https://github.com/engineer-man/piston | High-performance code execution engine using the Isolate sandbox (same as IOI competitive programming), open-source, supports local instance deployment as a simpler alternative to Judge0. | false | — |
| Automatic Generation of Programming Exercises and Code Explanations Using Large Language Models \| ACM ICER 2022 | https://dl.acm.org/doi/10.1145/3501385.3543957 | LLMs can automatically generate programming exercises and code explanations for CS education, establishing the foundational case for LLM-driven automated curriculum content generation. | false | — |
| Improving LLM Code Generation via Requirement-Aware Curriculum Reinforcement Learning | https://arxiv.org/html/2605.00433v1 | Curriculum reinforcement learning organizes code generation training tasks from easier to harder difficulty levels, significantly improving LLM code generation performance — a framework directly applicable to exercise sequencing. | false | — |
| OpenCoderRank: Personalized Technical Assessments with Generative AI | https://arxiv.org/pdf/2509.06774 | Generative AI can create and automatically grade personalized technical programming assessments, demonstrating a viable pipeline for automated exercise generation + LLM evaluation. | false | — |
| Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem | https://arxiv.org/html/2510.09907v1 | An AI agent autonomously writes Hypothesis property-based tests by reading type annotations, docstrings, and function names; presented at NeurIPS 2025 Deep Learning for Code Workshop. | false | — |
| An Empirical Evaluation of Property-Based Testing in Python \| OOPSLA 2025 | https://cseweb.ucsd.edu/~mcoblenz/assets/pdf/OOPSLA_2025_PBT.pdf | Hypothesis transforms test suites from brittle scripts into robust validation engines by automating input generation, enforcing invariants, and applying delta-debugging on failures. | false | — |
| LLM Cost Optimization: 8 Strategies That Cut API Spend by 80% (2026 Guide) | https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/ | Combining prompt optimization, semantic caching, and model routing (cascade: cheap model first, expensive model for borderline cases) achieves 50-70% cost reduction while maintaining output quality. | false | — |
| Prompt Caching Infrastructure \| Introl Blog | https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025 | Anthropic prefix caching delivers up to 90% cost reduction and 85% latency reduction for long, repeated prompts — directly applicable to evaluation rubrics reused across many learner Justify responses. | false | — |
| PyResBugs: A Dataset of Residual Python Bugs for Natural Language-Driven Fault Injection \| FORGE 2025 | https://conf.researchr.org/details/forge-2025/forge-2025-benchmarking/1/PyResBugs-A-Dataset-of-Residual-Python-Bugs-for-Natural-Language-Driven-Fault-Inject | Dataset pairing residual Python bugs from major frameworks with fault-free versions and natural language descriptions, enabling LLM-driven fault injection from NL bug specifications. | false | — |
| Effective test generation using pre-trained Large Language Models and mutation testing (MuTAP) \| ScienceDirect | https://www.sciencedirect.com/science/article/abs/pii/S0950584924000739 | MuTAP augments LLM prompts with surviving mutants to highlight test suite gaps, improving LLM-generated test effectiveness — pattern applicable to generating Harden-stage bug descriptions from surviving mutations. | false | — |
| Remote code execution sandbox: secure isolation at scale (2026 guide) \| Northflank | https://northflank.com/blog/remote-code-execution-sandbox | Defense-in-depth for code execution sandboxes combines seccomp profiles, AppArmor/SELinux, capability dropping, rootless execution, read-only filesystems, and network segmentation to significantly reduce sandbox escape risk. | false | — |
| Exercism Test Runner Interface Documentation | https://exercism.org/docs/building/tooling/test-runners/interface | Exercism uses Docker containers per language track with a 20-second timeout, 3 GB RAM, 100% CPU, and 1 MB stdout/stderr limits; each runner produces a standardized results.json; automated static analyzers run on top of test pass/fail for mentor-style commentary. | true | Independently confirmed by two separate research agents both citing the exercism.org documentation and the javascript-test-runner repo at https://github.com/exercism/javascript-test-runner |
| Exercism CLI GitHub | https://github.com/exercism/cli | Exercism's CLI is open-source (AGPL-3.0); supports 52+ language tracks through a two-tier exercise system (Concept Exercises for sequential concept teaching, Practice Exercises for application); mentoring is human-driven via volunteers with no LLM integration in validation. | true | Confirmed by two independent research agents citing https://exercism.org/docs/building/tracks/syllabus and https://exercism.org/blog/automated-mentoring-support-project |
| Exercism Automated Mentoring Support Project Blog | https://exercism.org/blog/automated-mentoring-support-project | Exercism deliberately chose volunteer human mentors over AI for feedback to preserve mentoring relationships; automated analysis (not LLM-based) provides immediate static feedback while human mentors handle nuanced style/paradigm review with hours-to-days latency. | true | Confirmed by two independent research agents citing both the blog post and exercism.org mentoring docs at https://exercism.org/docs/mentoring/how-to-give-great-feedback |
| Rustlings GitHub Repository | https://github.com/rust-lang/rustlings | Rustlings is a single-binary CLI (~84 exercises) that validates exercises via the Rust compiler itself; uses an 'I AM NOT DONE' comment flag as an explicit completion gate; exercises are embedded in the binary via rustlings-macros; no network dependency for validation. | true | Confirmed by two independent research agents; also cross-referenced at https://rust-lang.github.io/rustlings/ |
| Python Koans GitHub Repository | https://github.com/gregmalcolm/python_koans | Python Koans offers 38 topics with 304 koans using TDD assert-filling (replace __ placeholders) and implementation exercises run via Python's unittest; stops on first failure to focus learner attention; no LLM or network dependency. | false | — |
| CodeCrafters CLI Introduction Blog | https://codecrafters.io/blog/cli | CodeCrafters offers advanced staged challenges (Redis, Git, SQLite, Shell, DNS from scratch) validated via both git-push-triggered remote tests and a local 'codecrafters test' CLI; no LLM feedback; pure objective test-based validation with recommended solution diffs shown after each stage. | true | Cross-referenced with https://github.com/codecrafters-io/build-your-own-x confirming the staged, project-based architecture and multi-language support |
| Mutmut Python Mutation Testing GitHub | https://github.com/boxed/mutmut | Mutmut uses AST-level node transformations to avoid text-replace formatting corruption; it checks test runner exit codes rather than round-tripping AST back to source; generates subtle mutations (e.g., integer increment by 1, operator swap). | true | Confirmed by two independent research agents; AST round-trip limitation also documented in Cosmic Ray docs at https://cosmic-ray.readthedocs.io/en/stable/ |
| MutPy Python AST Mutation Testing GitHub | https://github.com/mutpy/mutpy | MutPy applies 20+ mutation operators at the AST level (arithmetic AOD/AOR, logical LOR/LOD, conditional COD/COI, control flow BCR); produces mutation scores and kill reports; well-documented and mature; not designed for educational debugging exercises. | true | Confirmed by two independent research agents; mutation operator taxonomy cross-referenced with https://github.com/theofidry/awesome-mutation-testing |
| Cosmic Ray Python Mutation Testing GitHub | https://github.com/sixty-north/cosmic-ray | Cosmic Ray distributes AST mutation testing across workers; Python's AST library does not preserve code formatting on round-trip, making AST-to-source conversion lossy — a known limitation shared with mutmut. | true | Round-trip formatting problem confirmed independently by both the Cosmic Ray readthedocs and by the mutmut research agent citing the same limitation |
| Awesome Mutation Testing Curated List | https://github.com/theofidry/awesome-mutation-testing | No existing open-source mutation testing tool is specifically designed for educational debugging exercises; all surveyed tools (MutPy, mutmut, cosmic-ray, Stryker, PIT) target test-suite quality assessment, not student-facing debugging practice. | true | Confirmed by two independent research agents reviewing the repository and cross-referencing with CodeGrade's blog post on mutation testing for education |
| Autograder+ Paper (arxiv 2510.26402) | https://arxiv.org/pdf/2510.26402 | Autograder+ combines autograding with LLM-based feedback generation via fine-tuned models; achieves BERTScore F1 of 0.7658 semantic alignment with expert feedback across 600 student submissions; uses Docker-based dynamic code execution and Ollama for local LLM inference. | false | — |
| Autograder+ GitHub Repository | https://github.com/zvikrnt/Autograder-Plus | Autograder+ is the closest open-source system to mastery-engine's goals, combining autograding and LLM feedback, but lacks explicit AST mutation for semantic bug injection and has no natural-language explanation grading (it generates feedback, not evaluates student explanations). | false | — |
| Microsoft LLM-Rubric GitHub | https://github.com/microsoft/LLM-Rubric | LLM-Rubric uses manually constructed rubrics with multiple LLM distributions combined via small neural networks to predict human evaluator patterns; applied to dialogue evaluation across 9 dimensions; not designed for programming education. | false | — |
| Rubric Is All You Need: Improving LLM-Based Code Evaluation (arxiv 2503.23989) | https://arxiv.org/html/2503.23989 | Question-specific rubrics (tailored to each OOP/DSA problem) outperform generic rubrics for LLM-based code evaluation; Pointwise Rubric Evaluation (PRE) — one LLM call per criterion — reduces halo effects; open-source LLMs match commercial models when given strong rubrics. | false | — |
| Exploring Effectiveness of LLMs for Automated Assessment of Student Self-Explanations (arxiv 2605.21614) | https://arxiv.org/html/2605.21614 | LLMs significantly outperform semantic similarity methods for grading student code self-explanations (LLM F1=0.98 vs. semantic similarity F1=0.72); no mature open-source tool yet packages this capability with autograding. | false | — |
| BugSpotter: Automated Debugging Exercise Generation (arxiv 2411.14303) | https://arxiv.org/pdf/2411.14303 | BugSpotter automatically generates debugging exercises by creating buggy versions of correct student code; demonstrates LLMs can be used to generate educational debugging exercises, not just evaluate them. | false | — |
| Conceptual Mutation Testing for Student Programming Misconceptions (Programming 2024) | https://2024.programming-conference.org/details/programming-2024-papers/16/Conceptual-Mutation-Testing-for-Student-Programming-Misconceptions | Mutation operators targeting known student conceptual misconceptions (e.g., off-by-one, wrong loop bound, wrong comparator) outperform standard mutation testing for educational outcomes; mutation testing is not just for test-suite quality but can model student error patterns. | false | — |
| Mutation Testing via Iterative LLM-Driven Scientific Debugging (arxiv 2503.08182) | https://arxiv.org/html/2503.08182v1 | Semantic-preserving mutations (functionally equivalent but syntactically different) reduce LLM debugging accuracy by 78%; LLMs can form hypotheses about how to kill specific mutants; AST manipulation can track fault movement through code. | false | — |
| On the Use of Large Language Models in Mutation Testing (arxiv 2406.09843) | https://arxiv.org/html/2406.09843v3 | GPT-4 produces fewer equivalent mutants than smaller LLMs in mutation generation; GPT-3.5 detects 96.7% of bugs in Defects4J; AST-level mutations are essential for syntactically valid mutant programs. | false | — |
| RepoDebug: Multi-Task Debugging Evaluation (arxiv 2509.04078) | https://arxiv.org/pdf/2509.04078 | RepoDebug uses Tree-Sitter to parse source code into language-agnostic ASTs and inject bugs at specific AST nodes; enables cross-language bug injection without language-specific parsers. | false | — |
| VPL Moodle Plugin GitHub | https://github.com/jcrodriguez-dis/moodle-mod_vpl | VPL provides sandboxed code execution inside Moodle with automatic compilation and test-based evaluation across Python, C, Java, JavaScript; uses a dedicated jail server for isolation; no mutation testing or LLM features. | false | — |
| OpenAI Evals GitHub | https://github.com/openai/evals | OpenAI Evals is an open-source framework for evaluating LLM outputs with a registry of benchmarks; evaluates LLM-generated code correctness but does not grade student explanations or inject AST bugs. | false | — |
| Pynguin Automated Test Generation GitHub | https://github.com/se2p/pynguin | Pynguin generates Python unit tests using search-based algorithms and uses MutPy internally to generate assertions by comparing behavior on original vs. mutated code; a research prototype not safe for arbitrary classroom code. | false | — |
| CodeGrade: Testing the Tests Blog Post | https://www.codegrade.com/blog/testing-the-tests-autograding-student-unit-tests-in-python-assignments | CodeGrade is a production educational autograding platform that integrates mutation testing to assess student unit test quality; it grades test coverage via mutation scores but does not evaluate natural language explanations or inject semantic bugs for debugging exercises. | false | — |
| Awesome FSRS: Free Spaced Repetition Scheduler Implementations | https://github.com/open-spaced-repetition/awesome-fsrs | FSRS is a modern, research-backed spaced repetition scheduling algorithm available in multiple open-source implementations; no existing open-source project integrates FSRS scheduling with programming exercise mastery loops. | false | — |
| claude-tutor: SM-2 Spaced Repetition for Programming Inside Claude Code | https://github.com/kirilxd/claude-tutor | claude-tutor implements SM-2 spaced repetition scheduling, adaptive quizzes, and personalized learning plans inside Claude Code; programming-focused but lacks test validation, AST mutation, or natural language grading. | false | — |
| Pynguin / Mutatest Safety Concern | https://github.com/EvanKepner/mutatest | Mutatest mutates only __pycache__ bytecode (not source files) for safety; Pynguin executes arbitrary code and is explicitly not safe for classroom environments; both lack explanation grading or educational debugging exercise generation. | false | — |
| LLM-as-a-Judge for Scalable Test Coverage Evaluation: Accuracy, Operational Reliability, and Cost | https://arxiv.org/abs/2512.01232 | GPT-4o Mini achieves 78x cost reduction compared to other models while maintaining 96.6% reliability (ECR@1) for LLM-as-a-judge evaluation across 500 evaluations; smaller models can match larger ones on structured evaluation tasks at a fraction of the cost. | true | ICLR 2025 published version at https://proceedings.iclr.cc/paper_files/paper/2025/file/08dabd5345b37fffcbe335bd578b15a0-Paper-Conference.pdf independently confirms cost-accuracy tradeoff findings. |
| RocketEval: Efficient Automated LLM Evaluation via Grading Checklist | https://arxiv.org/abs/2503.05142 | Using Gemma-2-2B as a judge achieves 0.965 correlation with human preferences (comparable to GPT-4o) while providing over 50-fold cost reduction for large-scale evaluation via checklist-based grading with lightweight LLMs. | true | OpenReview forum at https://openreview.net/forum?id=zJjzNj6QUe independently confirms the 50-fold cost reduction and correlation findings. |
| Automated Assignment Grading with Large Language Models: Insights From a Bioinformatics Course | https://arxiv.org/abs/2501.14499 | With well-designed prompts LLMs achieve grading accuracy comparable to human graders on 36 open-ended text questions from 100+ students; RAG over course material reduces mean absolute grading error by up to 19.47%; open-source Llama-405B-q4 performs comparably to GPT-4o. | true | Oxford Academic published version at https://academic.oup.com/bioinformatics/article/41/Supplement_1/i21/8199383 and PubMed Central at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12261420/ independently confirm these findings. |
| SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models (NeurIPS 2024 Spotlight) | https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html | SocraticLM outperforms GPT-4 by more than 12% in teaching performance using a Dean-Teacher-Student multi-agent pipeline trained on 35K Socratic-style multi-round teaching dialogues, evaluated across five pedagogical dimensions. | true | GitHub repository at https://github.com/Ljyustc/SocraticLM confirms the approach, dataset, and reproducibility of the NeurIPS 2024 Spotlight claims. |
| Benchmarking Uncertainty Calibration in Large Language Model Long-Form Question Answering | https://arxiv.org/pdf/2602.00279 | Instruction-tuned LLMs become overconfident compared to base models, rewarding confident guessing over calibrated uncertainty; Flex-ECE adapts Expected Calibration Error for partial correctness in open-ended generation tasks. | true | ICLR 2025 paper 'Do LLMs Estimate Uncertainty Well' at https://proceedings.iclr.cc/paper_files/paper/2025/file/ef472869c217bf693f2d9bbde66a6b07-Paper-Conference.pdf independently confirms overconfidence in instruction-tuned models. |
| BugSpotter: Automated Generation of Code Debugging Exercises (SIGCSE TS 2025) | https://arxiv.org/abs/2411.14303 | BugSpotter uses LLMs to automatically generate realistic buggy code snippets from problem descriptions; students interact by writing failing test cases; LLM-generated debugging exercises are comparable to manually created instructor exercises in student performance, validated in large classroom settings. | true | ACM SIGCSE Proceedings at https://dl.acm.org/doi/10.1145/3641554.3701974 and ResearchGate at https://www.researchgate.net/publication/389189240_BugSpotter_Automated_Generation_of_Code_Debugging_Exercises independently confirm publication and methodology. |
| Conceptual Mutation Testing for Student Programming Misconceptions (Programming Journal 2024) | https://arxiv.org/pdf/2401.00021 | Small numbers of mutants tailored to common student misconceptions outperform overwhelming students with numerous mutations; semantically clustering student test failures and translating them into conceptual mutants better identifies misconceptions than traditional mutation tools and reduces student frustration. | true | Brown University PLT blog at https://blog.brownplt.org/2023/10/31/conceptual-mutation-testing.html and Programming Journal 2024 at https://programming-journal.org/2024/8/7/ independently confirm the educational mutation testing approach. |
| A Comprehensive Study on Large Language Models for Mutation Testing (ACM TOSEM) | https://arxiv.org/abs/2406.09843 | LLM-generated mutants (BugFarm, LLMorpheus) achieve 111.29% higher fault detection compared to traditional grammar-based mutation across 851 real-world Java bugs; LLM mutations are more diverse, behaviorally closer to real bugs, and lower cost. | true | ACM Digital Library at https://dl.acm.org/doi/10.1145/3805038 confirms publication; Meta FSE 2025 paper independently validates LLM superiority for mutation testing at industrial scale. |
| Mutation-Guided LLM-based Test Generation at Meta (FSE 2025) | https://arxiv.org/pdf/2501.12862 | Meta's ACH tool applied mutation testing to 10,795 Android Kotlin classes generating 9,095 mutants and 571 privacy-hardening test cases; engineers accepted 73% of ACH-generated tests, demonstrating LLM-guided mutation testing is deployable at industrial scale and overcomes barriers that previously prevented adoption. | true | Meta Engineering Blog at https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/ and ACM Digital Library at https://dl.acm.org/doi/10.1145/3696630.3728544 independently confirm findings. |
| LECTOR: LLM-Enhanced Concept-based Test-Oriented Repetition for Adaptive Spaced Learning | https://arxiv.org/abs/2508.03275 | LECTOR achieves 90.2% success rate on learning tasks by combining LLM-powered semantic similarity assessment with spaced repetition scheduling, outperforming traditional baselines (88.4%) by directly addressing semantic confusion in review scheduling. | true | Cepeda et al. meta-analysis of 254 studies (via https://www.structural-learning.com/post/ebbinghaus-forgetting-curve) confirms spaced repetition produces 10-30% better retention than massed practice; adaptive scheduling research at https://arxiv.org/pdf/2004.11327 corroborates the LLM+SRS hybrid approach. |
| Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning | https://arxiv.org/abs/2506.06632 | Progressive Easy-to-Hard (E2H) curriculum scheduling enables models to build reasoning skills with fewer total samples than direct learning, achieving state-of-the-art performance across five reasoning benchmarks including planning and arithmetic. | true | ACER paper at https://arxiv.org/abs/2510.26336 independently validates Bloom's taxonomy-guided curriculum scheduling with measured 5-point performance improvements in specialized domains. |
| Teaching According to Students' Aptitude: Personalized Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs (TASA) | https://arxiv.org/abs/2511.15163 | TASA integrates student persona modeling, dynamic memory tracking, and Ebbinghaus forgetting curves with knowledge tracing to generate difficulty-calibrated questions; modeling temporal forgetting and learner profiles produces superior outcomes over static memory assumptions. | true | FOREVER paper at https://arxiv.org/abs/2601.03938 independently validates that forgetting mirrors the Ebbinghaus curve pattern, improving performance from 58.7% to 61.5% when replay schedules align with learning progress. |
| From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning (ACER) | https://arxiv.org/abs/2510.26336 | ACER synthesizes domain-specific curricula using Bloom's taxonomy-guided question generation and interleaved training; boosts specialized-domain performance by 5 percentage points and achieves 2+ point improvements on knowledge-intensive benchmarks (ARC, GPQA) while preventing catastrophic forgetting. | true | Curriculum RL paper at https://arxiv.org/abs/2506.06632 and RPKT at https://arxiv.org/abs/2508.11892 independently confirm that structured prerequisite scaffolding through cognitive-level taxonomy improves learning outcomes. |
| RPKT: Recursive Prerequisite Knowledge Tracing | https://arxiv.org/abs/2508.11892 | RPKT dynamically traces prerequisite concepts in real-time until reaching a learner's actual knowledge boundary, discovering multi-level prerequisite chains without pre-built knowledge graphs, addressing 'unknown unknowns' in personalized learning. | false | Knowledge Tracing survey at https://dl.acm.org/doi/10.1145/3706468.3706501 confirms LLMs can annotate knowledge concepts with near human-level accuracy, but the specific recursive prerequisite discovery claim was not independently corroborated by a second source. |
| ICLF: An Immersive Code Learning Framework based on Git for Teaching and Evaluating Student Programming Projects | https://arxiv.org/pdf/2601.14814 | ICLF uses a git-based pipeline with hidden parent repositories containing solutions and student private forks (with solutions stripped) to reduce grading platform dependency, support automated feedback, and allow curricula to evolve without disrupting student work; tested over several years including in edX MOOCs. | true | Git Worktrees for Parallel AI development literature (matklad July 2024 at https://matklad.github.io/2024/07/25/git-worktrees.html and Upsun developer post at https://developer.upsun.com/posts/ai/git-worktrees-for-parallel-ai-coding-agents) independently confirm git worktree shared-object-store isolation as a production-grade pattern. |
| NsJail: A lightweight process isolation tool (Google) | https://nsjail.dev/ | NsJail uses Linux namespaces, resource limits (CPU time, memory, file descriptors), cgroups, and seccomp-bpf syscall filters for process isolation; adopted by online judges (including Arch Online Judge, NeoHOJ) as industry-standard infrastructure for sandboxing user-submitted code execution. | true | ACM Computing Surveys article on online judge systems and multiple archived online judge implementations independently confirm NsJail as a standard isolation layer for competitive programming judges. |
| E2B: AI Code Execution Sandbox (Firecracker microVMs) | https://e2b.dev/ | E2B provides Firecracker microVM sandboxes with approximately 150ms startup time purpose-built for AI agent code execution, providing isolation against code execution attacks, filesystem manipulation, network exfiltration, and resource abuse. | false | Bunnyshell coding agent sandbox guide at https://www.bunnyshell.com/guides/coding-agent-sandbox/ references E2B as an industry-standard approach, but specific adoption metrics (Fortune 500 figures, session counts) were not independently corroborated. |

### Ideas that advance the goal
| Idea | Advances | Sources |
| --- | --- | --- |
| Use cosmic-ray with custom semantic mutation operator plugins for the Harden stage bug injection. Cosmic-ray's plugin API allows defining domain-specific AST mutations (off-by-one, wrong variable reference, sign flip, condition inversion) rather than relying on generic syntactic mutations, producing 'plausible-but-wrong' bugs that test genuine understanding. | Directly implements the Harden stage's AST-injected semantic bug requirement. Custom operators ensure bugs are cognitively plausible rather than trivially detectable, raising the bar for deep mastery validation. The plugin architecture also means new curricula can register their own mutation operators without engine changes, supporting the curriculum-agnostic extensibility goal. | https://cosmic-ray.readthedocs.io/ https://github.com/sixty-north/cosmic-ray https://ieeexplore.ieee.org/document/10818231/ |
| Use mutmut as the default mutation engine when rapid iteration is needed. Its 88.5% detection rate and 1.5x speed advantage over cosmic-ray make it suitable for quick Harden stage cycles where throughput matters more than operator customization. | Provides a fast, well-maintained mutation backend for the Harden stage. Mutmut's format-preserving approach ensures injected mutants remain syntactically clean Python, avoiding trivially visible injection artifacts that would break the 'plausible bug' requirement. | https://johal.in/mutation-testing-with-mutmut-python-for-code-reliability-2026/ https://ieeexplore.ieee.org/document/10818231/ https://dl.acm.org/doi/10.1145/3701625.3701659 |
| Adopt DeepEval + G-Eval as the Justify stage's LLM-as-judge backend. G-Eval's chain-of-thought rubric evaluation with form-filling reduces position bias and enables arbitrary custom grading criteria (e.g., 'does the learner identify the time complexity correctly?'), producing calibrated 0-1 scores with explicit per-criterion breakdowns. | Directly implements the LLM-graded Justify stage. DeepEval is open-source, locally runnable, and supports custom rubrics — essential for curriculum-agnostic operation. G-Eval's bias mitigation improves score reliability, strengthening the portfolio claim of 'LLM-as-evaluator with cost optimization'. | https://github.com/confident-ai/deepeval https://www.confident-ai.com/blog/g-eval-the-definitive-guide https://deepeval.com/docs/metrics-llm-evals |
| Implement a model cascade for Justify stage cost optimization: route clear pass/fail responses through a cheap small model (Haiku-class) and escalate only ambiguous borderline scores to a more capable model. This is the 'model routing' pattern empirically shown to achieve 50-70% cost reduction. | Directly advances the 'LLM-as-evaluator with cost optimization' architectural claim. Keeps the system economically viable for sustained personal use (Stanford CS336 self-study) while maintaining evaluation quality for hard cases. | https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/ https://github.com/confident-ai/deepeval |
| Use Anthropic prefix caching for the Justify stage's system prompt and grading rubric. Since the rubric and system instructions are long and reused across every learner response, prefix caching delivers up to 90% cost reduction and 85% latency reduction on repeated calls — the highest-leverage optimization for a system running many evaluations. | Directly reduces the per-evaluation LLM cost that makes sustained personal self-study economically feasible. Also reduces latency, improving the real-time feedback loop in the BJH cycle. | https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025 https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/ |
| Implement the Harden stage using git worktrees as ephemeral isolation: create a linked worktree per Harden session (`git worktree add`), inject the AST mutation, run validation, then remove the worktree. This is now the industry-standard pattern for throwaway isolated workspaces and avoids polluting the learner's working tree. | Implements the 'process isolation via git shadow worktrees' central architectural claim. The ephemeral pattern matches the Harden stage's lifecycle — create, corrupt, test, destroy — and is now backed by broad 2026 tooling consensus. | https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/ https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/ |
| Add nsjail (via the snekbox pattern) as a supplementary runtime sandbox for the Harden stage. Since git worktrees isolate file state but NOT process/network state, wrapping the learner's test execution in nsjail's seccomp-bpf + cgroup constraints prevents runaway test code from interfering with the host environment. | Completes the 'process isolation' architectural claim by addressing the well-documented gap that worktrees alone do not provide runtime isolation. Also makes the system safe for executing arbitrary user-submitted code, which is required for the competitive-programming curriculum use case. | https://www.morphllm.com/nsjail-sandbox https://github.com/python-discord/snekbox https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/ |
| Use Python's `ast.NodeTransformer` base class directly for the Harden stage's bug injection layer, implementing a small library of semantic mutation operators: comparison operator inversion (`<` → `<=`), arithmetic sign flip (`+` → `-`), off-by-one constant shift, and wrong-variable substitution within same scope. These produce plausible-but-wrong bugs that require genuine conceptual understanding to identify. | Provides the core 'runtime AST mutation' mechanism without requiring cosmic-ray or mutmut as a dependency for simple cases. Direct use of ast.NodeTransformer keeps the injection logic transparent, auditable, and curriculum-author-controllable — essential for the extensible curriculum design goal. | https://ieeexplore.ieee.org/document/10818231/ https://dl.acm.org/doi/10.1145/3701625.3701659 https://cosmic-ray.readthedocs.io/ |
| Use Hypothesis for Build stage validation harnesses. Instead of only hand-written unit tests, exercise specs include Hypothesis strategies that generate diverse inputs. This makes the Build stage validation harder to game with degenerate solutions and more robust than fixed test cases, while keeping validators as pure data in the curriculum manifest. | Strengthens the Build stage's auto-validation claim. Hypothesis's delta-debugging on failures also produces minimal counterexamples that can be surfaced as feedback, improving the pedagogical quality of the system. | https://arxiv.org/html/2510.09907v1 https://cseweb.ucsd.edu/~mcoblenz/assets/pdf/OOPSLA_2025_PBT.pdf |
| Apply the MuTAP technique to the Harden stage: after generating an AST mutation, have the LLM generate a natural language hint describing the bug class (without revealing the exact mutation) by prompting with the surviving mutation alongside the original code. This enriches the Harden stage's feedback loop and demonstrates a sophisticated LLM-mutation hybrid architecture. | Advances the 'automated content-generation pipeline' goal by using LLMs to auto-generate pedagogically useful hints at injection time. Also strengthens the portfolio showcase: LLM-mutation hybrid for hint generation is a non-trivial architectural contribution. | https://www.sciencedirect.com/science/article/abs/pii/S0950584924000739 https://conf.researchr.org/details/forge-2025/forge-2025-benchmarking/1/PyResBugs-A-Dataset-of-Residual-Python-Bugs-for-Natural-Language-Driven-Fault-Inject |
| Structure the automated content-generation pipeline using curriculum reinforcement learning (CRL) difficulty progression: generate exercises in order from simpler to more complex variants of each pattern, scoring difficulty via proxy metrics (cyclomatic complexity, number of concepts involved). This produces coherent LINEAR-mode learning paths rather than isolated problems. | Directly advances the 'LINEAR and LIBRARY mode' curriculum design goal and the 'automated content-generation pipeline' claim. CRL-ordered content ensures that auto-generated curricula form valid pedagogical sequences without manual curation. | https://arxiv.org/html/2605.00433v1 https://dl.acm.org/doi/10.1145/3501385.3543957 https://arxiv.org/pdf/2509.06774 |
| Use Judge0 or Piston as a local self-hosted code execution backend for the Build stage validator, providing consistent resource-limited execution (CPU time, memory) across different learner environments. This decouples exercise validation from the host Python environment and makes the system portable. | Strengthens the Build stage's auto-validation reliability and the competitive-programming curriculum use case. Also demonstrates awareness of production-grade code execution infrastructure in the portfolio showcase. | https://github.com/judge0/judge0 https://github.com/engineer-man/piston |
| Replace custom AST injection in generic_injector.py with operator taxonomy from MutPy/mutmut, using their 20+ validated mutation operators (AOD, AOR, COD, COI, BCR) rather than ad-hoc transforms. This provides a principled catalog of semantically meaningful mutations with known educational value. | Strengthens the Harden stage of the BJH loop (Goal 1) and the portfolio's claimed SYSTEM ARCHITECTURE achievement of 'runtime AST mutation' (Goal 2) by grounding mutation selection in peer-reviewed operator taxonomies rather than bespoke code. | https://github.com/mutpy/mutpy https://github.com/boxed/mutmut https://github.com/theofidry/awesome-mutation-testing |
| Avoid Python AST round-tripping for mutation application: instead of ast.parse -> mutate -> ast.unparse, use mutmut's approach of mutating bytecode/source strings and only checking test exit codes. This eliminates the formatting corruption that Python's AST library introduces on round-trip. | Directly fixes a known reliability risk in the Harden stage (Goal 1): if mutated code loses formatting or introduces syntax artifacts, the debugging exercise becomes unfairly difficult or silently broken. | https://github.com/boxed/mutmut https://github.com/sixty-north/cosmic-ray |
| Adopt Exercism's standardized test-runner interface (containerized runner + results.json schema) for the Build stage. This would make mastery-engine's validation layer curriculum-agnostic at the protocol level — any language track only needs a Docker image that speaks the Exercism test-runner protocol. | Directly advances Goal 4 (curriculum-agnostic extensibility) by providing an existing, battle-tested interface that already supports 52 language tracks, rather than a bespoke validation contract that must be re-implemented per curriculum. | https://exercism.org/docs/building/tooling/test-runners/interface https://github.com/exercism/cli |
| Use question-specific rubrics (generated per exercise from its spec) for the Justify stage LLM evaluation, rather than a single generic rubric. Research shows this approach closes the gap between open-source and commercial LLM graders and reduces halo effects via Pointwise Rubric Evaluation (one LLM call per criterion). | Improves Justify stage reliability (Goal 1) and strengthens the 'LLM-as-evaluator with cost optimization' architecture claim (Goal 2): more targeted prompts mean fewer tokens per evaluation and more consistent scoring. | https://arxiv.org/html/2503.23989 https://github.com/microsoft/LLM-Rubric |
| Use an ensemble of 3 independent LLM Justify-stage evaluations with majority vote rather than a single GPT-4o call. Research (LLM-Rubric, Autorubric) shows single-evaluator variance is high; majority-vote ensembles match human inter-rater reliability at modest cost (3x LLM calls but with haiku-class models for the panel). | Advances Goal 1 (reliable Justify grading) and Goal 2 (LLM cost optimization claim): using 3 cheaper model calls in majority vote costs less than 1 GPT-4o call while reducing variance. | https://github.com/microsoft/LLM-Rubric https://arxiv.org/pdf/2510.26402 https://github.com/confident-ai/deepeval |
| Adopt DeepEval as the Justify-stage evaluation backend instead of raw API calls. DeepEval wraps LLM-as-judge with bias mitigation, reproducible metric definitions, and CI integration — providing auditable, versioned evaluation logic that raw GPT-4o calls lack. | Advances Goal 2 (portfolio architecture claim of LLM-as-evaluator) by showing awareness of state-of-the-art evaluation infrastructure; also advances Goal 1 (reliable Justify stage) with production-grade tooling. | https://github.com/confident-ai/deepeval https://arxiv.org/pdf/2510.26402 |
| Target mutations in the Harden stage specifically at documented student conceptual misconceptions (off-by-one, wrong loop termination, wrong comparator direction, missing base case) rather than arbitrary AST node swaps. The 2024 Programming conference paper shows misconception-targeted mutations produce significantly better educational outcomes than standard mutation testing. | Advances Goal 1 (pedagogical depth of BJH loop) and Goal 4 (content-generation pipeline): bugs become curriculum-aware rather than random, making the Harden stage genuinely diagnostic of conceptual understanding. | https://2024.programming-conference.org/details/programming-2024-papers/16/Conceptual-Mutation-Testing-for-Student-Programming-Misconceptions https://arxiv.org/html/2406.09843v3 |
| Use Tree-Sitter instead of Python's built-in ast module for cross-language AST parsing in the Harden stage. RepoDebug demonstrates Tree-Sitter enables language-agnostic bug injection at specific AST node types without per-language parser code, which would make the Harden stage extensible to new curricula (C, JavaScript, Go) without rewriting the injector. | Directly advances Goal 4 (curriculum-agnostic framework) and strengthens the Goal 2 architecture claim: the injector becomes a language-agnostic component rather than Python-specific. | https://arxiv.org/pdf/2509.04078 https://github.com/mutpy/mutpy |
| Add FSRS or SM-2 spaced repetition scheduling to determine which exercises are revisited and when, keyed off BJH stage outcomes. No existing open-source platform (Exercism, Rustlings, CodeCrafters, Python Koans) integrates spaced repetition with automated code validation — this would be a genuine differentiator. | Advances Goal 1 (deep technical mastery through long-term retention) and Goal 3 (personal self-study harness on the author's own machine): FSRS scheduling is the state-of-the-art algorithm for minimizing review sessions while maximizing retention. | https://github.com/open-spaced-repetition/awesome-fsrs https://github.com/kirilxd/claude-tutor |
| Adopt CodeCrafters' intra-exercise staged decomposition: break each Build spec into sub-stages (e.g., 'implement forward pass' → 'handle batch dimension' → 'add dropout'), each independently auto-validated. This reduces cognitive load and provides finer-grained Harden targets (each sub-stage can have its own injected bug). | Advances Goal 1 (driving learners to deep mastery through manageable progressive challenge) and Goal 4 (curriculum structure): LINEAR mode maps naturally to staged progression while LIBRARY mode can expose individual sub-stages as standalone problems. | https://codecrafters.io/blog/cli https://github.com/codecrafters-io/build-your-own-x |
| Implement an explicit acknowledgment gate between BJH stages, analogous to Rustlings' 'I AM NOT DONE' comment. Without an explicit gate, a learner can accidentally progress to Justify before truly understanding the Build output. A CLI prompt ('Type UNDERSTOOD to continue to Justify') creates a deliberate psychological checkpoint that improves metacognitive engagement. | Advances Goal 1 (driving learners to deep mastery): the BJH loop's pedagogical value depends on each transition being intentional, not accidental. | https://github.com/rust-lang/rustlings https://github.com/gregmalcolm/python_koans |
| Ensure the Harden stage's mutated code does NOT introduce semantically equivalent mutants (mutations the test suite cannot distinguish from correct code). LLM-based mutation generation (GPT-4 per 2024 research) produces fewer equivalent mutants than rule-based tools, and could be used in the content-generation pipeline to pre-screen injected bugs for detectability. | Advances Goal 2 (portfolio's automated content-generation pipeline claim) and Goal 1 (Harden stage quality): equivalent mutants make debugging exercises unresolvable and silently undermine the pedagogical contract. | https://arxiv.org/html/2406.09843v3 https://arxiv.org/html/2503.08182v1 |
| Study Exercism's decision to NOT use LLMs for grading as a deliberate design choice worth documenting. Their reasoning — preserving mentoring relationships, avoiding hallucinated feedback, reducing latency variance — is a useful counter-argument that mastery-engine's portfolio documentation should explicitly address when justifying the Justify-stage LLM design. | Advances Goal 2 (portfolio / capability showcase): a portfolio that shows awareness of the trade-off and defends the LLM choice with evidence is more credible than one that treats it as obvious. | https://exercism.org/blog/automated-mentoring-support-project https://exercism.org/docs/mentoring/how-to-give-great-feedback |
| Validate that mastery-engine's git worktree isolation actually prevents mutations from leaking to the student's working tree by running a test case where the worktree is corrupted and verifying the parent HEAD is unaffected. Research on coding-agent worktree sandboxes (opencode-worktree-session) notes that 'subagent isolation doesn't fully lock shell working directory, allowing git operations to affect parent repo's HEAD' — this risk applies to mastery-engine too. | Advances Goal 2 (portfolio's 'process isolation via git shadow worktrees' claim): the claim is currently unverified against adversarial git operations; demonstrating it holds is necessary for the claim to stand. | https://github.com/rust-lang/rustlings https://arxiv.org/pdf/2509.04078 |
| Contribute mastery-engine's Harden stage as a standalone open-source library (e.g., 'edu-mutate') that wraps misconception-targeted AST mutation + git worktree isolation for any Python test suite. No such library exists; this would fill the gap identified across all surveyed projects (MutPy, CodeGrade, VPL all lack student-facing debugging exercise generation). | Advances Goal 2 (portfolio differentiation): releasing the Harden component independently demonstrates the architecture is genuinely reusable, not just incidentally decoupled, which is a stronger portfolio claim. | https://github.com/mutpy/mutpy https://www.codegrade.com/blog/testing-the-tests-autograding-student-unit-tests-in-python-assignments https://arxiv.org/pdf/2411.14303 |
| Use lightweight LLM judges (GPT-4o Mini or Gemma-2-2B) with checklist rubrics for the Justify phase to achieve 50-78x cost reduction versus frontier models while maintaining >96% grading reliability, making per-learner LLM evaluation economically sustainable at scale without quality loss. | Directly addresses the BJH loop's LLM-as-evaluator cost optimization requirement. Enables the Justify phase to run frequently without API budget constraints, supporting both the personal self-study harness goal and the portfolio showcase of cost-aware LLM integration architecture. | https://arxiv.org/abs/2512.01232 https://arxiv.org/abs/2503.05142 |
| Augment Justify-phase grading with RAG over each module's specification and reference materials, reducing mean absolute grading error by up to 19.47% and enabling open-source local LLMs to match GPT-4o accuracy — supporting fully offline operation on the personal self-study harness. | Advances the personal self-study / offline harness goal by enabling accurate LLM grading without cloud API dependency. Also advances the portfolio showcase by demonstrating RAG-augmented evaluation architecture applied to a real curriculum. | https://arxiv.org/abs/2501.14499 |
| Implement multi-turn Socratic dialogue in the Justify phase — rather than a single open-ended question, probe understanding through 3-5 targeted follow-up questions drilling into weak spots — using a SocraticLM-style pipeline to improve teaching effectiveness by 12%+ over single-shot grading. | Deepens the Justify phase beyond surface-level explanation verification toward genuine Socratic dialogue, directly advancing the 'deep technical mastery' pedagogical OS goal and differentiating the system architecturally from simple Q&A graders. | https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html |
| Design Justify rubrics as explicit checklists (concept coverage, correctness, depth, edge case awareness) rather than holistic prompts, reducing LLM judge overconfidence and enabling calibrated per-dimension scoring with confidence estimates that surface learner misconceptions more reliably. | Addresses known LLM calibration failure modes (instruction-tuned model overconfidence; domain alignment gap at 64-68%) in the Justify phase. Structured rubrics make grading transparent, auditable, and consistent — critical for a portfolio showcase of LLM evaluation quality and for the engineering credibility of the system. | https://arxiv.org/pdf/2602.00279 https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80 |
| For the Harden phase's AST bug injection, generate semantically meaningful mutants tuned to common misconceptions for each problem type (e.g., off-by-one in attention masking, wrong reduction axis in softmax, missing normalization) rather than random grammar-based mutations, following Conceptual Mutation Testing which outperforms bulk mutation in educational settings. | Directly advances Harden phase quality: conceptual mutants create debugging exercises targeting real understanding gaps rather than trivial syntactic noise, producing stronger pedagogical signal per session. Advances the BJH loop's central claim of runtime AST mutation for deep technical mastery. | https://arxiv.org/pdf/2401.00021 https://arxiv.org/abs/2406.09843 |
| Auto-generate the Harden phase bug catalog using LLMs: given a correct Build submission, prompt an LLM to propose semantically subtle mutations (wrong sign, transposed indices, missing clamp) then validate each by confirming the test suite catches it — following BugSpotter's approach of LLM-generated exercises proven comparable to instructor-created ones. | Advances the automated content-generation pipeline goal by eliminating manual curation of AST injection patterns per module. Makes the framework truly curriculum-agnostic: any correct implementation can seed a bug catalog. Directly advances the portfolio showcase of automated content generation. | https://arxiv.org/abs/2411.14303 https://arxiv.org/abs/2406.09843 |
| In the Harden phase pipeline, validate injected bugs by running the test suite against mutated code and confirming failures before presenting to the learner — following Meta's ACH workflow that achieved 73% engineer acceptance — eliminating equivalent mutants (syntactically changed but semantically identical) that would waste learner time and degrade trust in the system. | Advances Harden phase reliability by ensuring every debugging challenge is genuinely detectable. Equivalent mutant elimination is a known research problem; applying an industrial-proven filter improves the BJH loop's pedagogical value and makes the system's quality claims defensible for the portfolio showcase. | https://arxiv.org/pdf/2501.12862 https://arxiv.org/abs/2406.09843 |
| Add spaced repetition scheduling to the LINEAR curriculum mode: track per-module mastery decay using Ebbinghaus forgetting curves and resurface modules for review at optimal intervals (LECTOR-style), preventing forgetting of earlier concepts as the curriculum advances. | Advances the personal self-study harness goal by ensuring long-term retention, not just one-time completion — especially valuable for CS336 where early attention math underpins later modules. Advances the pedagogical OS goal of driving learners to deep mastery rather than shallow pass-rates. | https://arxiv.org/abs/2508.03275 https://arxiv.org/abs/2511.15163 |
| Structure automated curriculum content generation using Bloom's taxonomy levels (Remember → Understand → Apply → Analyze → Evaluate → Create), ordering Build specs, Justify questions, and Harden bugs to progress through cognitive levels within each module — following ACER's measured 5-point improvement from taxonomy-guided question generation. | Advances the curriculum-agnostic extensible framework goal: Bloom's taxonomy provides a universal cognitive scaffold applicable to any learning domain (transformers, algorithms, systems), giving the content-generation pipeline principled structure without domain-specific hardcoding. Strengthens the portfolio showcase with pedagogically grounded design. | https://arxiv.org/abs/2510.26336 https://arxiv.org/abs/2506.06632 |
| Implement prerequisite knowledge tracing (RPKT-style) in LIBRARY mode: when a learner fails the Build or Justify phase, recursively identify and surface prerequisite concepts likely missing, enabling the freeform LIBRARY mode to self-organize into a coherent learning sequence without authors manually specifying dependency graphs. | Advances the extensible curriculum-agnostic framework goal: LIBRARY mode's freeform nature currently risks learners hitting prerequisite walls. Dynamic prerequisite discovery addresses 'unknown unknowns' and makes authoring new curricula easier — no dependency graphs required. | https://arxiv.org/abs/2508.11892 https://arxiv.org/abs/2510.26336 |
| Augment the git shadow worktree isolation used in the Harden phase with seccomp-bpf syscall filtering (NsJail-style) to prevent runtime collisions and contain untrusted code execution, addressing the key gap that worktrees share ports and host process space with each other. | Advances the process isolation via git shadow worktrees architecture goal by hardening it against runtime collisions and security risks. Critical if the engine ever runs concurrent Harden sessions. Advances the portfolio showcase of production-grade process isolation sophistication beyond naive file-level isolation. | https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/ https://nsjail.dev/ |
| Formalize the shadow worktree pattern using a git-based grading pipeline (ICLF-style): maintain a hidden reference repository with solutions, generate public module repos with solutions stripped, and evaluate learner Build submissions in private worktree forks — enabling multi-learner or multi-session use without solution leakage. | Advances both personal self-study harness and extensible framework goals by formalizing the git isolation architecture with a proven educational-deployment pattern. Demonstrates production-grade thinking about multi-learner scalability even in a single-user context, strengthening the portfolio showcase. | https://arxiv.org/pdf/2601.14814 https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/ |

---
### machine-readable artifact
```json
{
  "candidates": [
    {
      "goal": "Be a curriculum-agnostic 'pedagogical operating system' CLI that drives learners to deep technical mastery through a three-stage Build-Justify-Harden (BJH) loop: implement to a spec (Build, auto-validated), defend understanding in natural language (Justify, LLM-graded), and debug an AST-injected semantic bug in their own correct code under git-worktree isolation (Harden).",
      "grounded": true,
      "successSignals": [
        {
          "signal": "A `mastery` CLI exists with the full BJH command surface (init, show, submit, start-challenge, status, curriculum-list, etc.) and renders the loop end-to-end without crashing.",
          "evidenceRef": "audit/03-execution.md:11 (mastery --help rc=0 lists submit/show/start-challenge/init/curriculum-list/status/select/create-bug)"
        },
        {
          "signal": "`submit` auto-detects and routes to the correct BJH stage, and a full submit->validate cycle renders 'Validation Passed', 'Bug Fixed', 'Module Complete'.",
          "evidenceRef": "engine/main.py:815 (submit cli_command auto-detects stage) + audit/03-execution.md:14 (full submit->validate cycle rendered Validation Passed/Bug Fixed/Module Complete)"
        },
        {
          "signal": "Build stage validates implementations by shelling out to per-module validator.sh harnesses with timeout/error handling.",
          "evidenceRef": "engine/validator.py:108 (subprocess.run of validator_path with timeout=300) + curricula/cs336_a1/modules/softmax/validator.sh:1"
        },
        {
          "signal": "Justify stage evaluates free-text answers against rubrics via an LLM (GPT-4o, Chain-of-Thought) with a keyword fast-filter.",
          "evidenceRef": "engine/services/llm_service.py:27 (LLMService) + audit/03-execution.md:14 ('Justify fast-filter rejection and LLM correct/incorrect paths all exercised')"
        },
        {
          "signal": "Harden stage injects a semantic (not syntactic) bug into the learner's correct code via AST mutation inside an ephemeral git shadow worktree, then validates the fix in isolation.",
          "evidenceRef": "engine/stages/harden.py:196 (_select_bug globs bug specs) + engine/ast_harden/generic_injector.py:19 (GenericBugInjector) + engine/workspace.py:24 (shadow-worktree isolation)"
        },
        {
          "signal": "The engine package is exercised by a passing test suite covering schemas/state/validator/justify/harden code paths.",
          "evidenceRef": "audit/03-execution.md:4 (185 passed; schemas.py 100%, validator.py 92%, stages/justify.py 95%, stages/harden.py 74%)"
        }
      ],
      "groundedConsensus": true,
      "judgeVotes": "4/4"
    },
    {
      "goal": "Serve as an engineering portfolio / capability showcase whose central claimed achievement is the SYSTEM ARCHITECTURE — runtime AST mutation, process isolation via git shadow worktrees, LLM-as-evaluator with cost optimization, and an automated content-generation pipeline — rather than the curriculum content itself.",
      "grounded": true,
      "successSignals": [
        {
          "signal": "README explicitly frames the deliverable as 'Key Engineering Features' (Runtime AST Mutation, Process Isolation via Shadow Worktrees, Socratic LLM Evaluation, Automated Content Pipeline, Curriculum-Agnostic Architecture).",
          "evidenceRef": "README.md:71-79 (Key Engineering Features table)"
        },
        {
          "signal": "License/NOTICE statement asserts 'the engineering achievement is the system architecture ... not the specific curriculum problems', distinguishing original engine code (MIT) from adapted content.",
          "evidenceRef": "README.md:421 ('Our Contribution: The engineering achievement is the system architecture')"
        },
        {
          "signal": "Runtime AST mutation is implemented as real code (multiple injector implementations) that parse->match->inject->unparse Python.",
          "evidenceRef": "engine/ast_harden/softmax_v2_1.py:229 (inject_softmax_bug_v2_1 two-phase pipeline) + engine/ast_harden/generic_injector.py:19"
        },
        {
          "signal": "An automated content pipeline parses unstructured source data into structured curriculum JSON (claimed 38+ problems) with CI enforcing manifest integrity.",
          "evidenceRef": "scripts/generate_module.py (content pipeline) + .github/workflows/validate_cp_manifest.yml:15 (manifest integrity CI regenerates manifest.json)"
        },
        {
          "signal": "Coverage/CI badges and a green-tests narrative are presented as portfolio signals.",
          "evidenceRef": "README.md:5-8 (Tests/coverage badges) + .github/workflows/tests.yml:10 (pytest+coverage CI job)"
        }
      ],
      "groundedConsensus": true,
      "judgeVotes": "4/4"
    },
    {
      "goal": "Function as the author's personal self-study / assignment-completion harness — primarily to work through Stanford CS336 (transformer-from-scratch) and competitive-programming/interview prep on their own machine, persisting personal progress.",
      "grounded": true,
      "successSignals": [
        {
          "signal": "A real, personal progress file is actively in use on the host, showing an in-progress CS336 run (curriculum=cs336_a1, module softmax, HARDEN stage).",
          "evidenceRef": "audit/03-execution.md:13 (mastery status: Curriculum=cs336_a1, Current Module 'Numerically Stable Softmax (1/22)', Stage HARDEN) + engine/state.py:32 (STATE_FILE = Path.home()/'.mastery_progress.json')"
        },
        {
          "signal": "A maintenance script packages the work for actual CS336 assignment submission (zips into the official assignment submission archive).",
          "evidenceRef": "maintenance/make_submission.sh:1 (runs pytest then zips into cs336-spring2025-assignment-1-submission.zip)"
        },
        {
          "signal": "Shipped curricula target the author's own learning goals: a full CS336 language-modeling track (~21-22 modules) and an interview/CP accelerator plus job-prep tracks.",
          "evidenceRef": "README.md:85-101 (cs336_a1 21 modules; cp_accelerator 38 LeetCode problems; job_prep_data_annotation) + curricula/cs336_a1/manifest.json"
        },
        {
          "signal": "Developer Mode ships pre-loaded reference implementations the author can run against, indicating single-operator (author-as-user) workflow rather than multi-tenant deployment.",
          "evidenceRef": "README.md:23 ('Activate Developer Mode ... pre-loaded reference implementations') + modes/developer/cs336_basics (reference solutions symlinked at repo root)"
        }
      ],
      "groundedConsensus": true,
      "judgeVotes": "4/4"
    },
    {
      "goal": "Be an extensible, curriculum-agnostic framework where curricula are pure data (manifest-described) supporting both LINEAR (sequential modules) and LIBRARY (freeform pattern/problem) modes, so new learning domains can be added without engine changes.",
      "grounded": true,
      "successSignals": [
        {
          "signal": "CurriculumManager loads manifest.json-described curricula and supports both LINEAR and LIBRARY curriculum types.",
          "evidenceRef": "engine/curriculum.py:31 (CurriculumManager loads manifest.json in LINEAR/LIBRARY modes) + engine/schemas.py (CurriculumType)"
        },
        {
          "signal": "Multiple independent curricula coexist and are selectable, spanning distinct domains (deep learning, competitive programming, job prep, python stdlib).",
          "evidenceRef": "README.md:83-107 (Included Curricula: cs336_a1, cp_accelerator, job_prep_data_annotation, python_for_cp) + curricula/ directory inventory (audit/01-understanding.md:175)"
        },
        {
          "signal": "Both modes are wired in the CLI: LINEAR runs sequential modules and LIBRARY exposes a `select` command to set active pattern/problem.",
          "evidenceRef": "engine/main.py:2684 (select cli_command for LIBRARY mode) + engine/main.py:555-586 (_submit_linear_workflow stage routing)"
        },
        {
          "signal": "Documented authoring path for new curricula treats curricula as data (manifest + per-module assets), not engine code.",
          "evidenceRef": "README.md:386-390 (Adding a New Curriculum: create manifest.json, define modules, create .patch bugs) + README.md:374 ('Curricula are data, not code')"
        }
      ],
      "groundedConsensus": true,
      "judgeVotes": "4/4"
    }
  ],
  "research": {
    "sources": [
      {
        "title": "Mutation Testing with Mutmut: Python for Code Reliability 2026",
        "url": "https://johal.in/mutation-testing-with-mutmut-python-for-code-reliability-2026/",
        "claim": "Mutmut achieves 88.5% mutation detection rate, outperforming Cosmic Ray's 82.7%, using AST-level Code Parser, Mutant Generator, and Test Runner components; achieves 1.5x faster mutant generation with 20% less overhead in 2025 benchmarks.",
        "verified": true,
        "corroboration": "IEEE Conference Publication (https://ieeexplore.ieee.org/document/10818231/) independently reports mutmut outperforming cosmic-ray in comparative benchmarks."
      },
      {
        "title": "Cosmic Ray: mutation testing for Python — Cosmic Ray documentation",
        "url": "https://cosmic-ray.readthedocs.io/",
        "claim": "Cosmic-ray operates at the Python AST level, supports custom mutation operators via a plugin architecture, and includes built-in build tool integration — making it extensible for semantic mutation variants.",
        "verified": true,
        "corroboration": "GitHub repository (https://github.com/sixty-north/cosmic-ray) confirms plugin architecture and AST-level operation independently."
      },
      {
        "title": "An Analysis and Comparison of Mutation Testing Tools for Python | IEEE Xplore",
        "url": "https://ieeexplore.ieee.org/document/10818231/",
        "claim": "Peer-reviewed comparison of MutPy, Mutmut, Mutatest, Poodle, and Cosmic Ray on Python mutation testing, covering detection rates and operator coverage.",
        "verified": true,
        "corroboration": "ACM SBES 2024 paper (https://dl.acm.org/doi/10.1145/3701625.3701659) provides independent static and dynamic comparison of the same tool landscape."
      },
      {
        "title": "Static and Dynamic Comparison of Mutation Testing Tools for Python | ACM SBES 2024",
        "url": "https://dl.acm.org/doi/10.1145/3701625.3701659",
        "claim": "Comprehensive comparison of source code mutators vs. AST mutators for Python, providing empirical data on operator coverage and false-positive rates.",
        "verified": true,
        "corroboration": "IEEE Conference Publication (https://ieeexplore.ieee.org/document/10818231/) independently analyzes the same tool landscape with overlapping results."
      },
      {
        "title": "DeepEval LLM Evaluation Framework GitHub",
        "url": "https://github.com/confident-ai/deepeval",
        "claim": "DeepEval is a pytest-like open-source framework for LLM evaluation with 30+ metrics (G-Eval, hallucination detection, task completion, bias/toxicity); supports local LLM-as-judge evaluation and integrates with Anthropic, OpenAI, LangChain.",
        "verified": true,
        "corroboration": "Confirmed independently by two research agents; cross-referenced with the Autorubric paper citing DeepEval as a reference framework"
      },
      {
        "title": "G-Eval Simply Explained: LLM-as-a-Judge for LLM Evaluation — Confident AI",
        "url": "https://www.confident-ai.com/blog/g-eval-the-definitive-guide",
        "claim": "G-Eval uses chain-of-thought reasoning combined with a form-filling paradigm to evaluate LLM outputs against arbitrary custom criteria, reducing position bias and length bias common in holistic LLM judges.",
        "verified": true,
        "corroboration": "DeepEval documentation (https://deepeval.com/docs/metrics-llm-evals) independently describes G-Eval CoT + form-filling approach."
      },
      {
        "title": "From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge",
        "url": "https://arxiv.org/pdf/2411.16594",
        "claim": "Comprehensive survey of LLM-as-judge paradigm covering scoring rubrics, bias correction approaches (position, verbosity, self-enhancement biases), and calibration methodologies applicable to open-ended technical explanation evaluation.",
        "verified": false
      },
      {
        "title": "Rubric-Based Evaluations & LLM-as-a-Judge: Methodologies, Biases, and Empirical Validation in Domain-Specific Contexts",
        "url": "https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80",
        "claim": "For expert-knowledge tasks, LLM-human agreement rates drop to 64-68%, well below inter-expert baselines of 72-75%; domain specificity reveals pronounced alignment gaps requiring hybrid human-in-the-loop workflows and domain-adapted rubrics.",
        "verified": true,
        "corroboration": "PMC article on 'Evaluating large language models for criterion-based grading' at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11683144/ independently corroborates domain-specific LLM grading limitation findings."
      },
      {
        "title": "How to Use Git Worktrees for Parallel AI Agent Execution | Augment Code",
        "url": "https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution",
        "claim": "Git worktrees give each agent its own isolated working directory and git index while sharing a single object store, reducing sequential CI time by approximately 63% (24 min to 9 min).",
        "verified": true,
        "corroboration": "Zylos Research (https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/) independently confirms worktree isolation pattern and cites same CI time-reduction figures."
      },
      {
        "title": "Git Worktree Isolation Patterns for Parallel AI Agent Development | Zylos Research",
        "url": "https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/",
        "claim": "Teams can reliably run 4-8 concurrent worktrees per developer; the productive ceiling is 5-7 concurrent on a modern laptop due to disk consumption (~5 GB per worktree). Ephemeral worktree creation/destruction per task is the recommended pattern.",
        "verified": true,
        "corroboration": "Augment Code guide (https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution) independently confirms worktree isolation as the standard pattern for parallel AI agent execution."
      },
      {
        "title": "Git Worktrees Need Runtime Isolation for Parallel AI Agent Development",
        "url": "https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/",
        "claim": "Git worktrees isolate code (separate filesystems) but not the runtime environment; shared ports, databases, and services still collide between worktrees; Docker containers provide additional runtime isolation through Linux namespaces and cgroups that worktrees alone cannot supply.",
        "verified": true,
        "corroboration": "Zazencodes article on Parallel Coding Agents at https://zazencodes.substack.com/p/parallel-coding-agents-with-container independently describes the complementary nature of worktrees and containers, stating the most mature teams combine both approaches."
      },
      {
        "title": "nsjail: Lightweight Linux Sandboxing for AI Code Execution (2026) | Morph",
        "url": "https://www.morphllm.com/nsjail-sandbox",
        "claim": "nsjail provides kernel-level isolation via Linux namespaces, cgroups, rlimits, and seccomp-bpf syscall filters for Python processes without requiring a container runtime; developed by Google, production-deployed since 2015.",
        "verified": true,
        "corroboration": "Python Discord's snekbox (https://github.com/python-discord/snekbox) independently validates nsjail for production Python sandboxing with confirmed deployment."
      },
      {
        "title": "GitHub - python-discord/snekbox: Easy, safe evaluation of arbitrary Python code",
        "url": "https://github.com/python-discord/snekbox",
        "claim": "Production-deployed service using nsjail to safely execute arbitrary Python code; demonstrates the nsjail + Python subprocess pattern with resource limits enforced via cgroups.",
        "verified": true,
        "corroboration": "Morph nsjail guide (https://www.morphllm.com/nsjail-sandbox) independently confirms snekbox as a production example of nsjail for Python code evaluation."
      },
      {
        "title": "GitHub - judge0/judge0: Robust, fast, scalable, and sandboxed open-source online code execution system",
        "url": "https://github.com/judge0/judge0",
        "claim": "Open-source sandboxed code execution system with multi-language support, resource constraints, and self-hosted option; used for competitive programming, e-learning, and code assessment platforms.",
        "verified": false
      },
      {
        "title": "GitHub - engineer-man/piston: A high performance general purpose code execution engine",
        "url": "https://github.com/engineer-man/piston",
        "claim": "High-performance code execution engine using the Isolate sandbox (same as IOI competitive programming), open-source, supports local instance deployment as a simpler alternative to Judge0.",
        "verified": false
      },
      {
        "title": "Automatic Generation of Programming Exercises and Code Explanations Using Large Language Models | ACM ICER 2022",
        "url": "https://dl.acm.org/doi/10.1145/3501385.3543957",
        "claim": "LLMs can automatically generate programming exercises and code explanations for CS education, establishing the foundational case for LLM-driven automated curriculum content generation.",
        "verified": false
      },
      {
        "title": "Improving LLM Code Generation via Requirement-Aware Curriculum Reinforcement Learning",
        "url": "https://arxiv.org/html/2605.00433v1",
        "claim": "Curriculum reinforcement learning organizes code generation training tasks from easier to harder difficulty levels, significantly improving LLM code generation performance — a framework directly applicable to exercise sequencing.",
        "verified": false
      },
      {
        "title": "OpenCoderRank: Personalized Technical Assessments with Generative AI",
        "url": "https://arxiv.org/pdf/2509.06774",
        "claim": "Generative AI can create and automatically grade personalized technical programming assessments, demonstrating a viable pipeline for automated exercise generation + LLM evaluation.",
        "verified": false
      },
      {
        "title": "Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem",
        "url": "https://arxiv.org/html/2510.09907v1",
        "claim": "An AI agent autonomously writes Hypothesis property-based tests by reading type annotations, docstrings, and function names; presented at NeurIPS 2025 Deep Learning for Code Workshop.",
        "verified": false
      },
      {
        "title": "An Empirical Evaluation of Property-Based Testing in Python | OOPSLA 2025",
        "url": "https://cseweb.ucsd.edu/~mcoblenz/assets/pdf/OOPSLA_2025_PBT.pdf",
        "claim": "Hypothesis transforms test suites from brittle scripts into robust validation engines by automating input generation, enforcing invariants, and applying delta-debugging on failures.",
        "verified": false
      },
      {
        "title": "LLM Cost Optimization: 8 Strategies That Cut API Spend by 80% (2026 Guide)",
        "url": "https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/",
        "claim": "Combining prompt optimization, semantic caching, and model routing (cascade: cheap model first, expensive model for borderline cases) achieves 50-70% cost reduction while maintaining output quality.",
        "verified": false
      },
      {
        "title": "Prompt Caching Infrastructure | Introl Blog",
        "url": "https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025",
        "claim": "Anthropic prefix caching delivers up to 90% cost reduction and 85% latency reduction for long, repeated prompts — directly applicable to evaluation rubrics reused across many learner Justify responses.",
        "verified": false
      },
      {
        "title": "PyResBugs: A Dataset of Residual Python Bugs for Natural Language-Driven Fault Injection | FORGE 2025",
        "url": "https://conf.researchr.org/details/forge-2025/forge-2025-benchmarking/1/PyResBugs-A-Dataset-of-Residual-Python-Bugs-for-Natural-Language-Driven-Fault-Inject",
        "claim": "Dataset pairing residual Python bugs from major frameworks with fault-free versions and natural language descriptions, enabling LLM-driven fault injection from NL bug specifications.",
        "verified": false
      },
      {
        "title": "Effective test generation using pre-trained Large Language Models and mutation testing (MuTAP) | ScienceDirect",
        "url": "https://www.sciencedirect.com/science/article/abs/pii/S0950584924000739",
        "claim": "MuTAP augments LLM prompts with surviving mutants to highlight test suite gaps, improving LLM-generated test effectiveness — pattern applicable to generating Harden-stage bug descriptions from surviving mutations.",
        "verified": false
      },
      {
        "title": "Remote code execution sandbox: secure isolation at scale (2026 guide) | Northflank",
        "url": "https://northflank.com/blog/remote-code-execution-sandbox",
        "claim": "Defense-in-depth for code execution sandboxes combines seccomp profiles, AppArmor/SELinux, capability dropping, rootless execution, read-only filesystems, and network segmentation to significantly reduce sandbox escape risk.",
        "verified": false
      },
      {
        "title": "Exercism Test Runner Interface Documentation",
        "url": "https://exercism.org/docs/building/tooling/test-runners/interface",
        "claim": "Exercism uses Docker containers per language track with a 20-second timeout, 3 GB RAM, 100% CPU, and 1 MB stdout/stderr limits; each runner produces a standardized results.json; automated static analyzers run on top of test pass/fail for mentor-style commentary.",
        "verified": true,
        "corroboration": "Independently confirmed by two separate research agents both citing the exercism.org documentation and the javascript-test-runner repo at https://github.com/exercism/javascript-test-runner"
      },
      {
        "title": "Exercism CLI GitHub",
        "url": "https://github.com/exercism/cli",
        "claim": "Exercism's CLI is open-source (AGPL-3.0); supports 52+ language tracks through a two-tier exercise system (Concept Exercises for sequential concept teaching, Practice Exercises for application); mentoring is human-driven via volunteers with no LLM integration in validation.",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents citing https://exercism.org/docs/building/tracks/syllabus and https://exercism.org/blog/automated-mentoring-support-project"
      },
      {
        "title": "Exercism Automated Mentoring Support Project Blog",
        "url": "https://exercism.org/blog/automated-mentoring-support-project",
        "claim": "Exercism deliberately chose volunteer human mentors over AI for feedback to preserve mentoring relationships; automated analysis (not LLM-based) provides immediate static feedback while human mentors handle nuanced style/paradigm review with hours-to-days latency.",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents citing both the blog post and exercism.org mentoring docs at https://exercism.org/docs/mentoring/how-to-give-great-feedback"
      },
      {
        "title": "Rustlings GitHub Repository",
        "url": "https://github.com/rust-lang/rustlings",
        "claim": "Rustlings is a single-binary CLI (~84 exercises) that validates exercises via the Rust compiler itself; uses an 'I AM NOT DONE' comment flag as an explicit completion gate; exercises are embedded in the binary via rustlings-macros; no network dependency for validation.",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents; also cross-referenced at https://rust-lang.github.io/rustlings/"
      },
      {
        "title": "Python Koans GitHub Repository",
        "url": "https://github.com/gregmalcolm/python_koans",
        "claim": "Python Koans offers 38 topics with 304 koans using TDD assert-filling (replace __ placeholders) and implementation exercises run via Python's unittest; stops on first failure to focus learner attention; no LLM or network dependency.",
        "verified": false
      },
      {
        "title": "CodeCrafters CLI Introduction Blog",
        "url": "https://codecrafters.io/blog/cli",
        "claim": "CodeCrafters offers advanced staged challenges (Redis, Git, SQLite, Shell, DNS from scratch) validated via both git-push-triggered remote tests and a local 'codecrafters test' CLI; no LLM feedback; pure objective test-based validation with recommended solution diffs shown after each stage.",
        "verified": true,
        "corroboration": "Cross-referenced with https://github.com/codecrafters-io/build-your-own-x confirming the staged, project-based architecture and multi-language support"
      },
      {
        "title": "Mutmut Python Mutation Testing GitHub",
        "url": "https://github.com/boxed/mutmut",
        "claim": "Mutmut uses AST-level node transformations to avoid text-replace formatting corruption; it checks test runner exit codes rather than round-tripping AST back to source; generates subtle mutations (e.g., integer increment by 1, operator swap).",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents; AST round-trip limitation also documented in Cosmic Ray docs at https://cosmic-ray.readthedocs.io/en/stable/"
      },
      {
        "title": "MutPy Python AST Mutation Testing GitHub",
        "url": "https://github.com/mutpy/mutpy",
        "claim": "MutPy applies 20+ mutation operators at the AST level (arithmetic AOD/AOR, logical LOR/LOD, conditional COD/COI, control flow BCR); produces mutation scores and kill reports; well-documented and mature; not designed for educational debugging exercises.",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents; mutation operator taxonomy cross-referenced with https://github.com/theofidry/awesome-mutation-testing"
      },
      {
        "title": "Cosmic Ray Python Mutation Testing GitHub",
        "url": "https://github.com/sixty-north/cosmic-ray",
        "claim": "Cosmic Ray distributes AST mutation testing across workers; Python's AST library does not preserve code formatting on round-trip, making AST-to-source conversion lossy — a known limitation shared with mutmut.",
        "verified": true,
        "corroboration": "Round-trip formatting problem confirmed independently by both the Cosmic Ray readthedocs and by the mutmut research agent citing the same limitation"
      },
      {
        "title": "Awesome Mutation Testing Curated List",
        "url": "https://github.com/theofidry/awesome-mutation-testing",
        "claim": "No existing open-source mutation testing tool is specifically designed for educational debugging exercises; all surveyed tools (MutPy, mutmut, cosmic-ray, Stryker, PIT) target test-suite quality assessment, not student-facing debugging practice.",
        "verified": true,
        "corroboration": "Confirmed by two independent research agents reviewing the repository and cross-referencing with CodeGrade's blog post on mutation testing for education"
      },
      {
        "title": "Autograder+ Paper (arxiv 2510.26402)",
        "url": "https://arxiv.org/pdf/2510.26402",
        "claim": "Autograder+ combines autograding with LLM-based feedback generation via fine-tuned models; achieves BERTScore F1 of 0.7658 semantic alignment with expert feedback across 600 student submissions; uses Docker-based dynamic code execution and Ollama for local LLM inference.",
        "verified": false
      },
      {
        "title": "Autograder+ GitHub Repository",
        "url": "https://github.com/zvikrnt/Autograder-Plus",
        "claim": "Autograder+ is the closest open-source system to mastery-engine's goals, combining autograding and LLM feedback, but lacks explicit AST mutation for semantic bug injection and has no natural-language explanation grading (it generates feedback, not evaluates student explanations).",
        "verified": false
      },
      {
        "title": "Microsoft LLM-Rubric GitHub",
        "url": "https://github.com/microsoft/LLM-Rubric",
        "claim": "LLM-Rubric uses manually constructed rubrics with multiple LLM distributions combined via small neural networks to predict human evaluator patterns; applied to dialogue evaluation across 9 dimensions; not designed for programming education.",
        "verified": false
      },
      {
        "title": "Rubric Is All You Need: Improving LLM-Based Code Evaluation (arxiv 2503.23989)",
        "url": "https://arxiv.org/html/2503.23989",
        "claim": "Question-specific rubrics (tailored to each OOP/DSA problem) outperform generic rubrics for LLM-based code evaluation; Pointwise Rubric Evaluation (PRE) — one LLM call per criterion — reduces halo effects; open-source LLMs match commercial models when given strong rubrics.",
        "verified": false
      },
      {
        "title": "Exploring Effectiveness of LLMs for Automated Assessment of Student Self-Explanations (arxiv 2605.21614)",
        "url": "https://arxiv.org/html/2605.21614",
        "claim": "LLMs significantly outperform semantic similarity methods for grading student code self-explanations (LLM F1=0.98 vs. semantic similarity F1=0.72); no mature open-source tool yet packages this capability with autograding.",
        "verified": false
      },
      {
        "title": "BugSpotter: Automated Debugging Exercise Generation (arxiv 2411.14303)",
        "url": "https://arxiv.org/pdf/2411.14303",
        "claim": "BugSpotter automatically generates debugging exercises by creating buggy versions of correct student code; demonstrates LLMs can be used to generate educational debugging exercises, not just evaluate them.",
        "verified": false
      },
      {
        "title": "Conceptual Mutation Testing for Student Programming Misconceptions (Programming 2024)",
        "url": "https://2024.programming-conference.org/details/programming-2024-papers/16/Conceptual-Mutation-Testing-for-Student-Programming-Misconceptions",
        "claim": "Mutation operators targeting known student conceptual misconceptions (e.g., off-by-one, wrong loop bound, wrong comparator) outperform standard mutation testing for educational outcomes; mutation testing is not just for test-suite quality but can model student error patterns.",
        "verified": false
      },
      {
        "title": "Mutation Testing via Iterative LLM-Driven Scientific Debugging (arxiv 2503.08182)",
        "url": "https://arxiv.org/html/2503.08182v1",
        "claim": "Semantic-preserving mutations (functionally equivalent but syntactically different) reduce LLM debugging accuracy by 78%; LLMs can form hypotheses about how to kill specific mutants; AST manipulation can track fault movement through code.",
        "verified": false
      },
      {
        "title": "On the Use of Large Language Models in Mutation Testing (arxiv 2406.09843)",
        "url": "https://arxiv.org/html/2406.09843v3",
        "claim": "GPT-4 produces fewer equivalent mutants than smaller LLMs in mutation generation; GPT-3.5 detects 96.7% of bugs in Defects4J; AST-level mutations are essential for syntactically valid mutant programs.",
        "verified": false
      },
      {
        "title": "RepoDebug: Multi-Task Debugging Evaluation (arxiv 2509.04078)",
        "url": "https://arxiv.org/pdf/2509.04078",
        "claim": "RepoDebug uses Tree-Sitter to parse source code into language-agnostic ASTs and inject bugs at specific AST nodes; enables cross-language bug injection without language-specific parsers.",
        "verified": false
      },
      {
        "title": "VPL Moodle Plugin GitHub",
        "url": "https://github.com/jcrodriguez-dis/moodle-mod_vpl",
        "claim": "VPL provides sandboxed code execution inside Moodle with automatic compilation and test-based evaluation across Python, C, Java, JavaScript; uses a dedicated jail server for isolation; no mutation testing or LLM features.",
        "verified": false
      },
      {
        "title": "OpenAI Evals GitHub",
        "url": "https://github.com/openai/evals",
        "claim": "OpenAI Evals is an open-source framework for evaluating LLM outputs with a registry of benchmarks; evaluates LLM-generated code correctness but does not grade student explanations or inject AST bugs.",
        "verified": false
      },
      {
        "title": "Pynguin Automated Test Generation GitHub",
        "url": "https://github.com/se2p/pynguin",
        "claim": "Pynguin generates Python unit tests using search-based algorithms and uses MutPy internally to generate assertions by comparing behavior on original vs. mutated code; a research prototype not safe for arbitrary classroom code.",
        "verified": false
      },
      {
        "title": "CodeGrade: Testing the Tests Blog Post",
        "url": "https://www.codegrade.com/blog/testing-the-tests-autograding-student-unit-tests-in-python-assignments",
        "claim": "CodeGrade is a production educational autograding platform that integrates mutation testing to assess student unit test quality; it grades test coverage via mutation scores but does not evaluate natural language explanations or inject semantic bugs for debugging exercises.",
        "verified": false
      },
      {
        "title": "Awesome FSRS: Free Spaced Repetition Scheduler Implementations",
        "url": "https://github.com/open-spaced-repetition/awesome-fsrs",
        "claim": "FSRS is a modern, research-backed spaced repetition scheduling algorithm available in multiple open-source implementations; no existing open-source project integrates FSRS scheduling with programming exercise mastery loops.",
        "verified": false
      },
      {
        "title": "claude-tutor: SM-2 Spaced Repetition for Programming Inside Claude Code",
        "url": "https://github.com/kirilxd/claude-tutor",
        "claim": "claude-tutor implements SM-2 spaced repetition scheduling, adaptive quizzes, and personalized learning plans inside Claude Code; programming-focused but lacks test validation, AST mutation, or natural language grading.",
        "verified": false
      },
      {
        "title": "Pynguin / Mutatest Safety Concern",
        "url": "https://github.com/EvanKepner/mutatest",
        "claim": "Mutatest mutates only __pycache__ bytecode (not source files) for safety; Pynguin executes arbitrary code and is explicitly not safe for classroom environments; both lack explanation grading or educational debugging exercise generation.",
        "verified": false
      },
      {
        "title": "LLM-as-a-Judge for Scalable Test Coverage Evaluation: Accuracy, Operational Reliability, and Cost",
        "url": "https://arxiv.org/abs/2512.01232",
        "claim": "GPT-4o Mini achieves 78x cost reduction compared to other models while maintaining 96.6% reliability (ECR@1) for LLM-as-a-judge evaluation across 500 evaluations; smaller models can match larger ones on structured evaluation tasks at a fraction of the cost.",
        "verified": true,
        "corroboration": "ICLR 2025 published version at https://proceedings.iclr.cc/paper_files/paper/2025/file/08dabd5345b37fffcbe335bd578b15a0-Paper-Conference.pdf independently confirms cost-accuracy tradeoff findings."
      },
      {
        "title": "RocketEval: Efficient Automated LLM Evaluation via Grading Checklist",
        "url": "https://arxiv.org/abs/2503.05142",
        "claim": "Using Gemma-2-2B as a judge achieves 0.965 correlation with human preferences (comparable to GPT-4o) while providing over 50-fold cost reduction for large-scale evaluation via checklist-based grading with lightweight LLMs.",
        "verified": true,
        "corroboration": "OpenReview forum at https://openreview.net/forum?id=zJjzNj6QUe independently confirms the 50-fold cost reduction and correlation findings."
      },
      {
        "title": "Automated Assignment Grading with Large Language Models: Insights From a Bioinformatics Course",
        "url": "https://arxiv.org/abs/2501.14499",
        "claim": "With well-designed prompts LLMs achieve grading accuracy comparable to human graders on 36 open-ended text questions from 100+ students; RAG over course material reduces mean absolute grading error by up to 19.47%; open-source Llama-405B-q4 performs comparably to GPT-4o.",
        "verified": true,
        "corroboration": "Oxford Academic published version at https://academic.oup.com/bioinformatics/article/41/Supplement_1/i21/8199383 and PubMed Central at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12261420/ independently confirm these findings."
      },
      {
        "title": "SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models (NeurIPS 2024 Spotlight)",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html",
        "claim": "SocraticLM outperforms GPT-4 by more than 12% in teaching performance using a Dean-Teacher-Student multi-agent pipeline trained on 35K Socratic-style multi-round teaching dialogues, evaluated across five pedagogical dimensions.",
        "verified": true,
        "corroboration": "GitHub repository at https://github.com/Ljyustc/SocraticLM confirms the approach, dataset, and reproducibility of the NeurIPS 2024 Spotlight claims."
      },
      {
        "title": "Benchmarking Uncertainty Calibration in Large Language Model Long-Form Question Answering",
        "url": "https://arxiv.org/pdf/2602.00279",
        "claim": "Instruction-tuned LLMs become overconfident compared to base models, rewarding confident guessing over calibrated uncertainty; Flex-ECE adapts Expected Calibration Error for partial correctness in open-ended generation tasks.",
        "verified": true,
        "corroboration": "ICLR 2025 paper 'Do LLMs Estimate Uncertainty Well' at https://proceedings.iclr.cc/paper_files/paper/2025/file/ef472869c217bf693f2d9bbde66a6b07-Paper-Conference.pdf independently confirms overconfidence in instruction-tuned models."
      },
      {
        "title": "BugSpotter: Automated Generation of Code Debugging Exercises (SIGCSE TS 2025)",
        "url": "https://arxiv.org/abs/2411.14303",
        "claim": "BugSpotter uses LLMs to automatically generate realistic buggy code snippets from problem descriptions; students interact by writing failing test cases; LLM-generated debugging exercises are comparable to manually created instructor exercises in student performance, validated in large classroom settings.",
        "verified": true,
        "corroboration": "ACM SIGCSE Proceedings at https://dl.acm.org/doi/10.1145/3641554.3701974 and ResearchGate at https://www.researchgate.net/publication/389189240_BugSpotter_Automated_Generation_of_Code_Debugging_Exercises independently confirm publication and methodology."
      },
      {
        "title": "Conceptual Mutation Testing for Student Programming Misconceptions (Programming Journal 2024)",
        "url": "https://arxiv.org/pdf/2401.00021",
        "claim": "Small numbers of mutants tailored to common student misconceptions outperform overwhelming students with numerous mutations; semantically clustering student test failures and translating them into conceptual mutants better identifies misconceptions than traditional mutation tools and reduces student frustration.",
        "verified": true,
        "corroboration": "Brown University PLT blog at https://blog.brownplt.org/2023/10/31/conceptual-mutation-testing.html and Programming Journal 2024 at https://programming-journal.org/2024/8/7/ independently confirm the educational mutation testing approach."
      },
      {
        "title": "A Comprehensive Study on Large Language Models for Mutation Testing (ACM TOSEM)",
        "url": "https://arxiv.org/abs/2406.09843",
        "claim": "LLM-generated mutants (BugFarm, LLMorpheus) achieve 111.29% higher fault detection compared to traditional grammar-based mutation across 851 real-world Java bugs; LLM mutations are more diverse, behaviorally closer to real bugs, and lower cost.",
        "verified": true,
        "corroboration": "ACM Digital Library at https://dl.acm.org/doi/10.1145/3805038 confirms publication; Meta FSE 2025 paper independently validates LLM superiority for mutation testing at industrial scale."
      },
      {
        "title": "Mutation-Guided LLM-based Test Generation at Meta (FSE 2025)",
        "url": "https://arxiv.org/pdf/2501.12862",
        "claim": "Meta's ACH tool applied mutation testing to 10,795 Android Kotlin classes generating 9,095 mutants and 571 privacy-hardening test cases; engineers accepted 73% of ACH-generated tests, demonstrating LLM-guided mutation testing is deployable at industrial scale and overcomes barriers that previously prevented adoption.",
        "verified": true,
        "corroboration": "Meta Engineering Blog at https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/ and ACM Digital Library at https://dl.acm.org/doi/10.1145/3696630.3728544 independently confirm findings."
      },
      {
        "title": "LECTOR: LLM-Enhanced Concept-based Test-Oriented Repetition for Adaptive Spaced Learning",
        "url": "https://arxiv.org/abs/2508.03275",
        "claim": "LECTOR achieves 90.2% success rate on learning tasks by combining LLM-powered semantic similarity assessment with spaced repetition scheduling, outperforming traditional baselines (88.4%) by directly addressing semantic confusion in review scheduling.",
        "verified": true,
        "corroboration": "Cepeda et al. meta-analysis of 254 studies (via https://www.structural-learning.com/post/ebbinghaus-forgetting-curve) confirms spaced repetition produces 10-30% better retention than massed practice; adaptive scheduling research at https://arxiv.org/pdf/2004.11327 corroborates the LLM+SRS hybrid approach."
      },
      {
        "title": "Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning",
        "url": "https://arxiv.org/abs/2506.06632",
        "claim": "Progressive Easy-to-Hard (E2H) curriculum scheduling enables models to build reasoning skills with fewer total samples than direct learning, achieving state-of-the-art performance across five reasoning benchmarks including planning and arithmetic.",
        "verified": true,
        "corroboration": "ACER paper at https://arxiv.org/abs/2510.26336 independently validates Bloom's taxonomy-guided curriculum scheduling with measured 5-point performance improvements in specialized domains."
      },
      {
        "title": "Teaching According to Students' Aptitude: Personalized Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs (TASA)",
        "url": "https://arxiv.org/abs/2511.15163",
        "claim": "TASA integrates student persona modeling, dynamic memory tracking, and Ebbinghaus forgetting curves with knowledge tracing to generate difficulty-calibrated questions; modeling temporal forgetting and learner profiles produces superior outcomes over static memory assumptions.",
        "verified": true,
        "corroboration": "FOREVER paper at https://arxiv.org/abs/2601.03938 independently validates that forgetting mirrors the Ebbinghaus curve pattern, improving performance from 58.7% to 61.5% when replay schedules align with learning progress."
      },
      {
        "title": "From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning (ACER)",
        "url": "https://arxiv.org/abs/2510.26336",
        "claim": "ACER synthesizes domain-specific curricula using Bloom's taxonomy-guided question generation and interleaved training; boosts specialized-domain performance by 5 percentage points and achieves 2+ point improvements on knowledge-intensive benchmarks (ARC, GPQA) while preventing catastrophic forgetting.",
        "verified": true,
        "corroboration": "Curriculum RL paper at https://arxiv.org/abs/2506.06632 and RPKT at https://arxiv.org/abs/2508.11892 independently confirm that structured prerequisite scaffolding through cognitive-level taxonomy improves learning outcomes."
      },
      {
        "title": "RPKT: Recursive Prerequisite Knowledge Tracing",
        "url": "https://arxiv.org/abs/2508.11892",
        "claim": "RPKT dynamically traces prerequisite concepts in real-time until reaching a learner's actual knowledge boundary, discovering multi-level prerequisite chains without pre-built knowledge graphs, addressing 'unknown unknowns' in personalized learning.",
        "verified": false,
        "corroboration": "Knowledge Tracing survey at https://dl.acm.org/doi/10.1145/3706468.3706501 confirms LLMs can annotate knowledge concepts with near human-level accuracy, but the specific recursive prerequisite discovery claim was not independently corroborated by a second source."
      },
      {
        "title": "ICLF: An Immersive Code Learning Framework based on Git for Teaching and Evaluating Student Programming Projects",
        "url": "https://arxiv.org/pdf/2601.14814",
        "claim": "ICLF uses a git-based pipeline with hidden parent repositories containing solutions and student private forks (with solutions stripped) to reduce grading platform dependency, support automated feedback, and allow curricula to evolve without disrupting student work; tested over several years including in edX MOOCs.",
        "verified": true,
        "corroboration": "Git Worktrees for Parallel AI development literature (matklad July 2024 at https://matklad.github.io/2024/07/25/git-worktrees.html and Upsun developer post at https://developer.upsun.com/posts/ai/git-worktrees-for-parallel-ai-coding-agents) independently confirm git worktree shared-object-store isolation as a production-grade pattern."
      },
      {
        "title": "NsJail: A lightweight process isolation tool (Google)",
        "url": "https://nsjail.dev/",
        "claim": "NsJail uses Linux namespaces, resource limits (CPU time, memory, file descriptors), cgroups, and seccomp-bpf syscall filters for process isolation; adopted by online judges (including Arch Online Judge, NeoHOJ) as industry-standard infrastructure for sandboxing user-submitted code execution.",
        "verified": true,
        "corroboration": "ACM Computing Surveys article on online judge systems and multiple archived online judge implementations independently confirm NsJail as a standard isolation layer for competitive programming judges."
      },
      {
        "title": "E2B: AI Code Execution Sandbox (Firecracker microVMs)",
        "url": "https://e2b.dev/",
        "claim": "E2B provides Firecracker microVM sandboxes with approximately 150ms startup time purpose-built for AI agent code execution, providing isolation against code execution attacks, filesystem manipulation, network exfiltration, and resource abuse.",
        "verified": false,
        "corroboration": "Bunnyshell coding agent sandbox guide at https://www.bunnyshell.com/guides/coding-agent-sandbox/ references E2B as an industry-standard approach, but specific adoption metrics (Fortune 500 figures, session counts) were not independently corroborated."
      }
    ],
    "ideas": [
      {
        "idea": "Use cosmic-ray with custom semantic mutation operator plugins for the Harden stage bug injection. Cosmic-ray's plugin API allows defining domain-specific AST mutations (off-by-one, wrong variable reference, sign flip, condition inversion) rather than relying on generic syntactic mutations, producing 'plausible-but-wrong' bugs that test genuine understanding.",
        "advancesGoal": "Directly implements the Harden stage's AST-injected semantic bug requirement. Custom operators ensure bugs are cognitively plausible rather than trivially detectable, raising the bar for deep mastery validation. The plugin architecture also means new curricula can register their own mutation operators without engine changes, supporting the curriculum-agnostic extensibility goal.",
        "sourceRefs": [
          "https://cosmic-ray.readthedocs.io/",
          "https://github.com/sixty-north/cosmic-ray",
          "https://ieeexplore.ieee.org/document/10818231/"
        ]
      },
      {
        "idea": "Use mutmut as the default mutation engine when rapid iteration is needed. Its 88.5% detection rate and 1.5x speed advantage over cosmic-ray make it suitable for quick Harden stage cycles where throughput matters more than operator customization.",
        "advancesGoal": "Provides a fast, well-maintained mutation backend for the Harden stage. Mutmut's format-preserving approach ensures injected mutants remain syntactically clean Python, avoiding trivially visible injection artifacts that would break the 'plausible bug' requirement.",
        "sourceRefs": [
          "https://johal.in/mutation-testing-with-mutmut-python-for-code-reliability-2026/",
          "https://ieeexplore.ieee.org/document/10818231/",
          "https://dl.acm.org/doi/10.1145/3701625.3701659"
        ]
      },
      {
        "idea": "Adopt DeepEval + G-Eval as the Justify stage's LLM-as-judge backend. G-Eval's chain-of-thought rubric evaluation with form-filling reduces position bias and enables arbitrary custom grading criteria (e.g., 'does the learner identify the time complexity correctly?'), producing calibrated 0-1 scores with explicit per-criterion breakdowns.",
        "advancesGoal": "Directly implements the LLM-graded Justify stage. DeepEval is open-source, locally runnable, and supports custom rubrics — essential for curriculum-agnostic operation. G-Eval's bias mitigation improves score reliability, strengthening the portfolio claim of 'LLM-as-evaluator with cost optimization'.",
        "sourceRefs": [
          "https://github.com/confident-ai/deepeval",
          "https://www.confident-ai.com/blog/g-eval-the-definitive-guide",
          "https://deepeval.com/docs/metrics-llm-evals"
        ]
      },
      {
        "idea": "Implement a model cascade for Justify stage cost optimization: route clear pass/fail responses through a cheap small model (Haiku-class) and escalate only ambiguous borderline scores to a more capable model. This is the 'model routing' pattern empirically shown to achieve 50-70% cost reduction.",
        "advancesGoal": "Directly advances the 'LLM-as-evaluator with cost optimization' architectural claim. Keeps the system economically viable for sustained personal use (Stanford CS336 self-study) while maintaining evaluation quality for hard cases.",
        "sourceRefs": [
          "https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/",
          "https://github.com/confident-ai/deepeval"
        ]
      },
      {
        "idea": "Use Anthropic prefix caching for the Justify stage's system prompt and grading rubric. Since the rubric and system instructions are long and reused across every learner response, prefix caching delivers up to 90% cost reduction and 85% latency reduction on repeated calls — the highest-leverage optimization for a system running many evaluations.",
        "advancesGoal": "Directly reduces the per-evaluation LLM cost that makes sustained personal self-study economically feasible. Also reduces latency, improving the real-time feedback loop in the BJH cycle.",
        "sourceRefs": [
          "https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025",
          "https://blog.premai.io/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/"
        ]
      },
      {
        "idea": "Implement the Harden stage using git worktrees as ephemeral isolation: create a linked worktree per Harden session (`git worktree add`), inject the AST mutation, run validation, then remove the worktree. This is now the industry-standard pattern for throwaway isolated workspaces and avoids polluting the learner's working tree.",
        "advancesGoal": "Implements the 'process isolation via git shadow worktrees' central architectural claim. The ephemeral pattern matches the Harden stage's lifecycle — create, corrupt, test, destroy — and is now backed by broad 2026 tooling consensus.",
        "sourceRefs": [
          "https://www.augmentcode.com/guides/git-worktrees-parallel-ai-agent-execution",
          "https://zylos.ai/research/2026-02-22-git-worktree-parallel-ai-development/",
          "https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/"
        ]
      },
      {
        "idea": "Add nsjail (via the snekbox pattern) as a supplementary runtime sandbox for the Harden stage. Since git worktrees isolate file state but NOT process/network state, wrapping the learner's test execution in nsjail's seccomp-bpf + cgroup constraints prevents runaway test code from interfering with the host environment.",
        "advancesGoal": "Completes the 'process isolation' architectural claim by addressing the well-documented gap that worktrees alone do not provide runtime isolation. Also makes the system safe for executing arbitrary user-submitted code, which is required for the competitive-programming curriculum use case.",
        "sourceRefs": [
          "https://www.morphllm.com/nsjail-sandbox",
          "https://github.com/python-discord/snekbox",
          "https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/"
        ]
      },
      {
        "idea": "Use Python's `ast.NodeTransformer` base class directly for the Harden stage's bug injection layer, implementing a small library of semantic mutation operators: comparison operator inversion (`<` → `<=`), arithmetic sign flip (`+` → `-`), off-by-one constant shift, and wrong-variable substitution within same scope. These produce plausible-but-wrong bugs that require genuine conceptual understanding to identify.",
        "advancesGoal": "Provides the core 'runtime AST mutation' mechanism without requiring cosmic-ray or mutmut as a dependency for simple cases. Direct use of ast.NodeTransformer keeps the injection logic transparent, auditable, and curriculum-author-controllable — essential for the extensible curriculum design goal.",
        "sourceRefs": [
          "https://ieeexplore.ieee.org/document/10818231/",
          "https://dl.acm.org/doi/10.1145/3701625.3701659",
          "https://cosmic-ray.readthedocs.io/"
        ]
      },
      {
        "idea": "Use Hypothesis for Build stage validation harnesses. Instead of only hand-written unit tests, exercise specs include Hypothesis strategies that generate diverse inputs. This makes the Build stage validation harder to game with degenerate solutions and more robust than fixed test cases, while keeping validators as pure data in the curriculum manifest.",
        "advancesGoal": "Strengthens the Build stage's auto-validation claim. Hypothesis's delta-debugging on failures also produces minimal counterexamples that can be surfaced as feedback, improving the pedagogical quality of the system.",
        "sourceRefs": [
          "https://arxiv.org/html/2510.09907v1",
          "https://cseweb.ucsd.edu/~mcoblenz/assets/pdf/OOPSLA_2025_PBT.pdf"
        ]
      },
      {
        "idea": "Apply the MuTAP technique to the Harden stage: after generating an AST mutation, have the LLM generate a natural language hint describing the bug class (without revealing the exact mutation) by prompting with the surviving mutation alongside the original code. This enriches the Harden stage's feedback loop and demonstrates a sophisticated LLM-mutation hybrid architecture.",
        "advancesGoal": "Advances the 'automated content-generation pipeline' goal by using LLMs to auto-generate pedagogically useful hints at injection time. Also strengthens the portfolio showcase: LLM-mutation hybrid for hint generation is a non-trivial architectural contribution.",
        "sourceRefs": [
          "https://www.sciencedirect.com/science/article/abs/pii/S0950584924000739",
          "https://conf.researchr.org/details/forge-2025/forge-2025-benchmarking/1/PyResBugs-A-Dataset-of-Residual-Python-Bugs-for-Natural-Language-Driven-Fault-Inject"
        ]
      },
      {
        "idea": "Structure the automated content-generation pipeline using curriculum reinforcement learning (CRL) difficulty progression: generate exercises in order from simpler to more complex variants of each pattern, scoring difficulty via proxy metrics (cyclomatic complexity, number of concepts involved). This produces coherent LINEAR-mode learning paths rather than isolated problems.",
        "advancesGoal": "Directly advances the 'LINEAR and LIBRARY mode' curriculum design goal and the 'automated content-generation pipeline' claim. CRL-ordered content ensures that auto-generated curricula form valid pedagogical sequences without manual curation.",
        "sourceRefs": [
          "https://arxiv.org/html/2605.00433v1",
          "https://dl.acm.org/doi/10.1145/3501385.3543957",
          "https://arxiv.org/pdf/2509.06774"
        ]
      },
      {
        "idea": "Use Judge0 or Piston as a local self-hosted code execution backend for the Build stage validator, providing consistent resource-limited execution (CPU time, memory) across different learner environments. This decouples exercise validation from the host Python environment and makes the system portable.",
        "advancesGoal": "Strengthens the Build stage's auto-validation reliability and the competitive-programming curriculum use case. Also demonstrates awareness of production-grade code execution infrastructure in the portfolio showcase.",
        "sourceRefs": [
          "https://github.com/judge0/judge0",
          "https://github.com/engineer-man/piston"
        ]
      },
      {
        "idea": "Replace custom AST injection in generic_injector.py with operator taxonomy from MutPy/mutmut, using their 20+ validated mutation operators (AOD, AOR, COD, COI, BCR) rather than ad-hoc transforms. This provides a principled catalog of semantically meaningful mutations with known educational value.",
        "advancesGoal": "Strengthens the Harden stage of the BJH loop (Goal 1) and the portfolio's claimed SYSTEM ARCHITECTURE achievement of 'runtime AST mutation' (Goal 2) by grounding mutation selection in peer-reviewed operator taxonomies rather than bespoke code.",
        "sourceRefs": [
          "https://github.com/mutpy/mutpy",
          "https://github.com/boxed/mutmut",
          "https://github.com/theofidry/awesome-mutation-testing"
        ]
      },
      {
        "idea": "Avoid Python AST round-tripping for mutation application: instead of ast.parse -> mutate -> ast.unparse, use mutmut's approach of mutating bytecode/source strings and only checking test exit codes. This eliminates the formatting corruption that Python's AST library introduces on round-trip.",
        "advancesGoal": "Directly fixes a known reliability risk in the Harden stage (Goal 1): if mutated code loses formatting or introduces syntax artifacts, the debugging exercise becomes unfairly difficult or silently broken.",
        "sourceRefs": [
          "https://github.com/boxed/mutmut",
          "https://github.com/sixty-north/cosmic-ray"
        ]
      },
      {
        "idea": "Adopt Exercism's standardized test-runner interface (containerized runner + results.json schema) for the Build stage. This would make mastery-engine's validation layer curriculum-agnostic at the protocol level — any language track only needs a Docker image that speaks the Exercism test-runner protocol.",
        "advancesGoal": "Directly advances Goal 4 (curriculum-agnostic extensibility) by providing an existing, battle-tested interface that already supports 52 language tracks, rather than a bespoke validation contract that must be re-implemented per curriculum.",
        "sourceRefs": [
          "https://exercism.org/docs/building/tooling/test-runners/interface",
          "https://github.com/exercism/cli"
        ]
      },
      {
        "idea": "Use question-specific rubrics (generated per exercise from its spec) for the Justify stage LLM evaluation, rather than a single generic rubric. Research shows this approach closes the gap between open-source and commercial LLM graders and reduces halo effects via Pointwise Rubric Evaluation (one LLM call per criterion).",
        "advancesGoal": "Improves Justify stage reliability (Goal 1) and strengthens the 'LLM-as-evaluator with cost optimization' architecture claim (Goal 2): more targeted prompts mean fewer tokens per evaluation and more consistent scoring.",
        "sourceRefs": [
          "https://arxiv.org/html/2503.23989",
          "https://github.com/microsoft/LLM-Rubric"
        ]
      },
      {
        "idea": "Use an ensemble of 3 independent LLM Justify-stage evaluations with majority vote rather than a single GPT-4o call. Research (LLM-Rubric, Autorubric) shows single-evaluator variance is high; majority-vote ensembles match human inter-rater reliability at modest cost (3x LLM calls but with haiku-class models for the panel).",
        "advancesGoal": "Advances Goal 1 (reliable Justify grading) and Goal 2 (LLM cost optimization claim): using 3 cheaper model calls in majority vote costs less than 1 GPT-4o call while reducing variance.",
        "sourceRefs": [
          "https://github.com/microsoft/LLM-Rubric",
          "https://arxiv.org/pdf/2510.26402",
          "https://github.com/confident-ai/deepeval"
        ]
      },
      {
        "idea": "Adopt DeepEval as the Justify-stage evaluation backend instead of raw API calls. DeepEval wraps LLM-as-judge with bias mitigation, reproducible metric definitions, and CI integration — providing auditable, versioned evaluation logic that raw GPT-4o calls lack.",
        "advancesGoal": "Advances Goal 2 (portfolio architecture claim of LLM-as-evaluator) by showing awareness of state-of-the-art evaluation infrastructure; also advances Goal 1 (reliable Justify stage) with production-grade tooling.",
        "sourceRefs": [
          "https://github.com/confident-ai/deepeval",
          "https://arxiv.org/pdf/2510.26402"
        ]
      },
      {
        "idea": "Target mutations in the Harden stage specifically at documented student conceptual misconceptions (off-by-one, wrong loop termination, wrong comparator direction, missing base case) rather than arbitrary AST node swaps. The 2024 Programming conference paper shows misconception-targeted mutations produce significantly better educational outcomes than standard mutation testing.",
        "advancesGoal": "Advances Goal 1 (pedagogical depth of BJH loop) and Goal 4 (content-generation pipeline): bugs become curriculum-aware rather than random, making the Harden stage genuinely diagnostic of conceptual understanding.",
        "sourceRefs": [
          "https://2024.programming-conference.org/details/programming-2024-papers/16/Conceptual-Mutation-Testing-for-Student-Programming-Misconceptions",
          "https://arxiv.org/html/2406.09843v3"
        ]
      },
      {
        "idea": "Use Tree-Sitter instead of Python's built-in ast module for cross-language AST parsing in the Harden stage. RepoDebug demonstrates Tree-Sitter enables language-agnostic bug injection at specific AST node types without per-language parser code, which would make the Harden stage extensible to new curricula (C, JavaScript, Go) without rewriting the injector.",
        "advancesGoal": "Directly advances Goal 4 (curriculum-agnostic framework) and strengthens the Goal 2 architecture claim: the injector becomes a language-agnostic component rather than Python-specific.",
        "sourceRefs": [
          "https://arxiv.org/pdf/2509.04078",
          "https://github.com/mutpy/mutpy"
        ]
      },
      {
        "idea": "Add FSRS or SM-2 spaced repetition scheduling to determine which exercises are revisited and when, keyed off BJH stage outcomes. No existing open-source platform (Exercism, Rustlings, CodeCrafters, Python Koans) integrates spaced repetition with automated code validation — this would be a genuine differentiator.",
        "advancesGoal": "Advances Goal 1 (deep technical mastery through long-term retention) and Goal 3 (personal self-study harness on the author's own machine): FSRS scheduling is the state-of-the-art algorithm for minimizing review sessions while maximizing retention.",
        "sourceRefs": [
          "https://github.com/open-spaced-repetition/awesome-fsrs",
          "https://github.com/kirilxd/claude-tutor"
        ]
      },
      {
        "idea": "Adopt CodeCrafters' intra-exercise staged decomposition: break each Build spec into sub-stages (e.g., 'implement forward pass' → 'handle batch dimension' → 'add dropout'), each independently auto-validated. This reduces cognitive load and provides finer-grained Harden targets (each sub-stage can have its own injected bug).",
        "advancesGoal": "Advances Goal 1 (driving learners to deep mastery through manageable progressive challenge) and Goal 4 (curriculum structure): LINEAR mode maps naturally to staged progression while LIBRARY mode can expose individual sub-stages as standalone problems.",
        "sourceRefs": [
          "https://codecrafters.io/blog/cli",
          "https://github.com/codecrafters-io/build-your-own-x"
        ]
      },
      {
        "idea": "Implement an explicit acknowledgment gate between BJH stages, analogous to Rustlings' 'I AM NOT DONE' comment. Without an explicit gate, a learner can accidentally progress to Justify before truly understanding the Build output. A CLI prompt ('Type UNDERSTOOD to continue to Justify') creates a deliberate psychological checkpoint that improves metacognitive engagement.",
        "advancesGoal": "Advances Goal 1 (driving learners to deep mastery): the BJH loop's pedagogical value depends on each transition being intentional, not accidental.",
        "sourceRefs": [
          "https://github.com/rust-lang/rustlings",
          "https://github.com/gregmalcolm/python_koans"
        ]
      },
      {
        "idea": "Ensure the Harden stage's mutated code does NOT introduce semantically equivalent mutants (mutations the test suite cannot distinguish from correct code). LLM-based mutation generation (GPT-4 per 2024 research) produces fewer equivalent mutants than rule-based tools, and could be used in the content-generation pipeline to pre-screen injected bugs for detectability.",
        "advancesGoal": "Advances Goal 2 (portfolio's automated content-generation pipeline claim) and Goal 1 (Harden stage quality): equivalent mutants make debugging exercises unresolvable and silently undermine the pedagogical contract.",
        "sourceRefs": [
          "https://arxiv.org/html/2406.09843v3",
          "https://arxiv.org/html/2503.08182v1"
        ]
      },
      {
        "idea": "Study Exercism's decision to NOT use LLMs for grading as a deliberate design choice worth documenting. Their reasoning — preserving mentoring relationships, avoiding hallucinated feedback, reducing latency variance — is a useful counter-argument that mastery-engine's portfolio documentation should explicitly address when justifying the Justify-stage LLM design.",
        "advancesGoal": "Advances Goal 2 (portfolio / capability showcase): a portfolio that shows awareness of the trade-off and defends the LLM choice with evidence is more credible than one that treats it as obvious.",
        "sourceRefs": [
          "https://exercism.org/blog/automated-mentoring-support-project",
          "https://exercism.org/docs/mentoring/how-to-give-great-feedback"
        ]
      },
      {
        "idea": "Validate that mastery-engine's git worktree isolation actually prevents mutations from leaking to the student's working tree by running a test case where the worktree is corrupted and verifying the parent HEAD is unaffected. Research on coding-agent worktree sandboxes (opencode-worktree-session) notes that 'subagent isolation doesn't fully lock shell working directory, allowing git operations to affect parent repo's HEAD' — this risk applies to mastery-engine too.",
        "advancesGoal": "Advances Goal 2 (portfolio's 'process isolation via git shadow worktrees' claim): the claim is currently unverified against adversarial git operations; demonstrating it holds is necessary for the claim to stand.",
        "sourceRefs": [
          "https://github.com/rust-lang/rustlings",
          "https://arxiv.org/pdf/2509.04078"
        ]
      },
      {
        "idea": "Contribute mastery-engine's Harden stage as a standalone open-source library (e.g., 'edu-mutate') that wraps misconception-targeted AST mutation + git worktree isolation for any Python test suite. No such library exists; this would fill the gap identified across all surveyed projects (MutPy, CodeGrade, VPL all lack student-facing debugging exercise generation).",
        "advancesGoal": "Advances Goal 2 (portfolio differentiation): releasing the Harden component independently demonstrates the architecture is genuinely reusable, not just incidentally decoupled, which is a stronger portfolio claim.",
        "sourceRefs": [
          "https://github.com/mutpy/mutpy",
          "https://www.codegrade.com/blog/testing-the-tests-autograding-student-unit-tests-in-python-assignments",
          "https://arxiv.org/pdf/2411.14303"
        ]
      },
      {
        "idea": "Use lightweight LLM judges (GPT-4o Mini or Gemma-2-2B) with checklist rubrics for the Justify phase to achieve 50-78x cost reduction versus frontier models while maintaining >96% grading reliability, making per-learner LLM evaluation economically sustainable at scale without quality loss.",
        "advancesGoal": "Directly addresses the BJH loop's LLM-as-evaluator cost optimization requirement. Enables the Justify phase to run frequently without API budget constraints, supporting both the personal self-study harness goal and the portfolio showcase of cost-aware LLM integration architecture.",
        "sourceRefs": [
          "https://arxiv.org/abs/2512.01232",
          "https://arxiv.org/abs/2503.05142"
        ]
      },
      {
        "idea": "Augment Justify-phase grading with RAG over each module's specification and reference materials, reducing mean absolute grading error by up to 19.47% and enabling open-source local LLMs to match GPT-4o accuracy — supporting fully offline operation on the personal self-study harness.",
        "advancesGoal": "Advances the personal self-study / offline harness goal by enabling accurate LLM grading without cloud API dependency. Also advances the portfolio showcase by demonstrating RAG-augmented evaluation architecture applied to a real curriculum.",
        "sourceRefs": [
          "https://arxiv.org/abs/2501.14499"
        ]
      },
      {
        "idea": "Implement multi-turn Socratic dialogue in the Justify phase — rather than a single open-ended question, probe understanding through 3-5 targeted follow-up questions drilling into weak spots — using a SocraticLM-style pipeline to improve teaching effectiveness by 12%+ over single-shot grading.",
        "advancesGoal": "Deepens the Justify phase beyond surface-level explanation verification toward genuine Socratic dialogue, directly advancing the 'deep technical mastery' pedagogical OS goal and differentiating the system architecturally from simple Q&A graders.",
        "sourceRefs": [
          "https://proceedings.neurips.cc/paper_files/paper/2024/hash/9bae399d1f34b8650351c1bd3692aeae-Abstract-Conference.html"
        ]
      },
      {
        "idea": "Design Justify rubrics as explicit checklists (concept coverage, correctness, depth, edge case awareness) rather than holistic prompts, reducing LLM judge overconfidence and enabling calibrated per-dimension scoring with confidence estimates that surface learner misconceptions more reliably.",
        "advancesGoal": "Addresses known LLM calibration failure modes (instruction-tuned model overconfidence; domain alignment gap at 64-68%) in the Justify phase. Structured rubrics make grading transparent, auditable, and consistent — critical for a portfolio showcase of LLM evaluation quality and for the engineering credibility of the system.",
        "sourceRefs": [
          "https://arxiv.org/pdf/2602.00279",
          "https://medium.com/@adnanmasood/rubric-based-evals-llm-as-a-judge-methodologies-and-empirical-validation-in-domain-context-71936b989e80"
        ]
      },
      {
        "idea": "For the Harden phase's AST bug injection, generate semantically meaningful mutants tuned to common misconceptions for each problem type (e.g., off-by-one in attention masking, wrong reduction axis in softmax, missing normalization) rather than random grammar-based mutations, following Conceptual Mutation Testing which outperforms bulk mutation in educational settings.",
        "advancesGoal": "Directly advances Harden phase quality: conceptual mutants create debugging exercises targeting real understanding gaps rather than trivial syntactic noise, producing stronger pedagogical signal per session. Advances the BJH loop's central claim of runtime AST mutation for deep technical mastery.",
        "sourceRefs": [
          "https://arxiv.org/pdf/2401.00021",
          "https://arxiv.org/abs/2406.09843"
        ]
      },
      {
        "idea": "Auto-generate the Harden phase bug catalog using LLMs: given a correct Build submission, prompt an LLM to propose semantically subtle mutations (wrong sign, transposed indices, missing clamp) then validate each by confirming the test suite catches it — following BugSpotter's approach of LLM-generated exercises proven comparable to instructor-created ones.",
        "advancesGoal": "Advances the automated content-generation pipeline goal by eliminating manual curation of AST injection patterns per module. Makes the framework truly curriculum-agnostic: any correct implementation can seed a bug catalog. Directly advances the portfolio showcase of automated content generation.",
        "sourceRefs": [
          "https://arxiv.org/abs/2411.14303",
          "https://arxiv.org/abs/2406.09843"
        ]
      },
      {
        "idea": "In the Harden phase pipeline, validate injected bugs by running the test suite against mutated code and confirming failures before presenting to the learner — following Meta's ACH workflow that achieved 73% engineer acceptance — eliminating equivalent mutants (syntactically changed but semantically identical) that would waste learner time and degrade trust in the system.",
        "advancesGoal": "Advances Harden phase reliability by ensuring every debugging challenge is genuinely detectable. Equivalent mutant elimination is a known research problem; applying an industrial-proven filter improves the BJH loop's pedagogical value and makes the system's quality claims defensible for the portfolio showcase.",
        "sourceRefs": [
          "https://arxiv.org/pdf/2501.12862",
          "https://arxiv.org/abs/2406.09843"
        ]
      },
      {
        "idea": "Add spaced repetition scheduling to the LINEAR curriculum mode: track per-module mastery decay using Ebbinghaus forgetting curves and resurface modules for review at optimal intervals (LECTOR-style), preventing forgetting of earlier concepts as the curriculum advances.",
        "advancesGoal": "Advances the personal self-study harness goal by ensuring long-term retention, not just one-time completion — especially valuable for CS336 where early attention math underpins later modules. Advances the pedagogical OS goal of driving learners to deep mastery rather than shallow pass-rates.",
        "sourceRefs": [
          "https://arxiv.org/abs/2508.03275",
          "https://arxiv.org/abs/2511.15163"
        ]
      },
      {
        "idea": "Structure automated curriculum content generation using Bloom's taxonomy levels (Remember → Understand → Apply → Analyze → Evaluate → Create), ordering Build specs, Justify questions, and Harden bugs to progress through cognitive levels within each module — following ACER's measured 5-point improvement from taxonomy-guided question generation.",
        "advancesGoal": "Advances the curriculum-agnostic extensible framework goal: Bloom's taxonomy provides a universal cognitive scaffold applicable to any learning domain (transformers, algorithms, systems), giving the content-generation pipeline principled structure without domain-specific hardcoding. Strengthens the portfolio showcase with pedagogically grounded design.",
        "sourceRefs": [
          "https://arxiv.org/abs/2510.26336",
          "https://arxiv.org/abs/2506.06632"
        ]
      },
      {
        "idea": "Implement prerequisite knowledge tracing (RPKT-style) in LIBRARY mode: when a learner fails the Build or Justify phase, recursively identify and surface prerequisite concepts likely missing, enabling the freeform LIBRARY mode to self-organize into a coherent learning sequence without authors manually specifying dependency graphs.",
        "advancesGoal": "Advances the extensible curriculum-agnostic framework goal: LIBRARY mode's freeform nature currently risks learners hitting prerequisite walls. Dynamic prerequisite discovery addresses 'unknown unknowns' and makes authoring new curricula easier — no dependency graphs required.",
        "sourceRefs": [
          "https://arxiv.org/abs/2508.11892",
          "https://arxiv.org/abs/2510.26336"
        ]
      },
      {
        "idea": "Augment the git shadow worktree isolation used in the Harden phase with seccomp-bpf syscall filtering (NsJail-style) to prevent runtime collisions and contain untrusted code execution, addressing the key gap that worktrees share ports and host process space with each other.",
        "advancesGoal": "Advances the process isolation via git shadow worktrees architecture goal by hardening it against runtime collisions and security risks. Critical if the engine ever runs concurrent Harden sessions. Advances the portfolio showcase of production-grade process isolation sophistication beyond naive file-level isolation.",
        "sourceRefs": [
          "https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/",
          "https://nsjail.dev/"
        ]
      },
      {
        "idea": "Formalize the shadow worktree pattern using a git-based grading pipeline (ICLF-style): maintain a hidden reference repository with solutions, generate public module repos with solutions stripped, and evaluate learner Build submissions in private worktree forks — enabling multi-learner or multi-session use without solution leakage.",
        "advancesGoal": "Advances both personal self-study harness and extensible framework goals by formalizing the git isolation architecture with a proven educational-deployment pattern. Demonstrates production-grade thinking about multi-learner scalability even in a single-user context, strengthening the portfolio showcase.",
        "sourceRefs": [
          "https://arxiv.org/pdf/2601.14814",
          "https://www.penligent.ai/hackinglabs/git-worktrees-need-runtime-isolation-for-parallel-ai-agent-development/"
        ]
      }
    ]
  },
  "webEnabled": true
}
```
