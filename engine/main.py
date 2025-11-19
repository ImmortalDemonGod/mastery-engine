"""
Mastery Engine CLI - Main entry point.

This module provides the command-line interface for the Mastery Engine,
implementing commands for the Build, Justify, Harden learning loop.

Commands:
    status: Show current learning progress
    next: Display the next build prompt
    submit: Submit work for current stage (auto-detects build/justify/harden)
    submit-build: [LEGACY] Submit and validate a build implementation
    submit-justification: [LEGACY] Submit an answer to a justify question
    submit-fix: [LEGACY] Submit a bug fix for the harden challenge
    flag-issue: Report a problem with the curriculum

Note: The unified 'submit' command is the recommended interface. Legacy
commands (submit-build, submit-justification, submit-fix) are maintained
for backward compatibility.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from dotenv import load_dotenv

# Load environment variables from .env file (override shell to honor .env settings)
load_dotenv(override=True)

from engine.state import StateManager, StateFileCorruptedError
from engine.curriculum import CurriculumManager, CurriculumNotFoundError, CurriculumInvalidError
from engine.workspace import WorkspaceManager, PatchApplicationError
from engine.schemas import UserProgress, CurriculumType
from engine.validator import (
    ValidationSubsystem,
    ValidatorNotFoundError,
    ValidatorTimeoutError,
    ValidatorExecutionError
)
from engine.stages.harden import HardenRunner, HardenChallengeError
from engine.stages.justify import JustifyRunner, JustifyQuestionsError
from engine.services.llm_service import LLMService, ConfigurationError, LLMAPIError, LLMResponseError
from engine.utils import find_project_root
import subprocess


# Initialize Typer app and Rich console
app = typer.Typer(
    name="engine",
    help="Mastery Engine: Build, Justify, Harden learning system",
    add_completion=False,
)
console = Console()

# Environment for running validators (e.g. preserve uv run context)
validator_env = {
    **subprocess.os.environ,
    'FORCE_COLOR': '1',
    'PYTHONUNBUFFERED': '1'
}

# Shadow worktree configuration
# Use project root to ensure it works from any subdirectory
try:
    _PROJECT_ROOT = find_project_root()
    SHADOW_WORKTREE_DIR = _PROJECT_ROOT / ".mastery_engine_worktree"
except RuntimeError:
    # Fallback for edge cases (e.g., running outside repo)
    SHADOW_WORKTREE_DIR = Path(".mastery_engine_worktree")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / '.mastery_engine.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def require_shadow_worktree() -> Path:
    """
    Verify shadow worktree exists and return its path.
    
    This is a safety check to ensure users have run 'engine init' before
    attempting to use the engine. Provides clear guidance if not initialized.
    
    Returns:
        Path to shadow worktree
        
    Exits:
        If shadow worktree not found, prints helpful error and exits
    """
    if not SHADOW_WORKTREE_DIR.exists():
        console.print()
        console.print(Panel(
            "[bold yellow]Not Initialized[/bold yellow]\n\n"
            "The Mastery Engine has not been initialized.\n\n"
            "Please run [bold cyan]engine init <curriculum_id>[/bold cyan] first.\n\n"
            "Example: [cyan]engine init cs336_a1[/cyan]",
            title="INITIALIZATION REQUIRED",
            border_style="yellow"
        ))
        console.print()
        sys.exit(1)
    
    return SHADOW_WORKTREE_DIR


# ============================================================================
# Helper Functions for Unified Submit Command (P0 CLI Improvement)
# ============================================================================

def _load_curriculum_state() -> tuple:
    """
    Load state and curriculum for submit commands.
    
    Returns:
        Tuple of (state_mgr, curr_mgr, progress, manifest)
    """
    state_mgr = StateManager()
    curr_mgr = CurriculumManager()
    progress = state_mgr.load()
    manifest = curr_mgr.load_manifest(progress.curriculum_id)
    return state_mgr, curr_mgr, progress, manifest


def _check_curriculum_complete(progress, manifest) -> bool:
    """
    Check if user has completed all modules.
    
    Returns:
        True if complete (and displays message), False otherwise
    """
    if progress.current_module_index >= len(manifest.modules):
        console.print()
        console.print(Panel(
            "[bold green]🎉 Curriculum Complete![/bold green]\n\n"
            "You have finished all modules. Congratulations!",
            title="All Modules Complete",
            border_style="green"
        ))
        console.print()
        return True
    return False


def _submit_build_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """
    Handle Build stage submission.
    
    Validates user implementation against test suite.
    
    Args:
        state_mgr: State manager instance
        curr_mgr: Curriculum manager instance
        progress: User progress object
        manifest: Curriculum manifest
        
    Returns:
        True if validation passed and progress advanced, False otherwise
    """
    workspace_mgr = WorkspaceManager()
    validator_subsys = ValidationSubsystem()
    
    current_module = manifest.modules[progress.current_module_index]
    validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
    workspace_path = Path.cwd()
    
    console.print()
    console.print(f"[bold cyan]Running validator for {current_module.name}...[/bold cyan]")
    console.print()
    
    result = validator_subsys.execute(validator_path, workspace_path)
    
    if result.exit_code == 0:
        # Success!
        console.print(Panel(
            "[bold green]✅ Validation Passed![/bold green]\n\n"
            "Your implementation passed all tests.",
            title="Success",
            border_style="green"
        ))
        
        # Show performance if available
        if result.performance_seconds is not None:
            console.print()
            console.print(f"[dim]Performance: {result.performance_seconds:.3f} seconds[/dim]")
            
            if current_module.baseline_perf_seconds:
                speedup = current_module.baseline_perf_seconds / result.performance_seconds
                if speedup > 2.0:
                    console.print(f"[yellow]⚡ Impressive! {speedup:.1f}x faster than baseline![/yellow]")
        
        console.print()
        
        # Advance state to justify stage
        progress.mark_stage_complete("build")
        state_mgr.save(progress)
        
        logger.info(f"Build stage completed for module '{current_module.id}'")
        
        # Show next action
        console.print(Panel(
            "Next step: Answer conceptual questions.\n\n"
            "Run [bold cyan]engine submit[/bold cyan] to continue.",
            title="Next Action",
            border_style="blue"
        ))
        console.print()
        return True
    else:
        # Failure
        console.print(Panel(
            "[bold red]❌ Validation Failed[/bold red]\n\n"
            "Your implementation did not pass all tests. See details below:",
            title="Failure",
            border_style="red"
        ))
        console.print()
        console.print("[bold]Test Output:[/bold]")
        console.print(result.stderr if result.stderr else result.stdout)
        console.print()
        
        logger.info(f"Build validation failed for module '{current_module.id}'")
        return False


def _submit_justify_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """
    Handle Justify stage submission with $EDITOR integration and full LLM evaluation.
    
    Opens user's editor for multi-line answer input, then evaluates response using:
    1. Fast keyword filter (catches shallow/vague answers)
    2. LLM semantic evaluation (if no keyword match)
    
    Args:
        state_mgr: State manager instance
        curr_mgr: Curriculum manager instance
        progress: User progress object
        manifest: Curriculum manifest
        
    Returns:
        True if answer accepted and progress advanced, False otherwise
    """
    import tempfile
    import os
    
    current_module = manifest.modules[progress.current_module_index]
    justify_runner = JustifyRunner(curr_mgr)
    questions = justify_runner.load_questions(progress.curriculum_id, current_module)
    
    if not questions:
        raise CurriculumInvalidError(f"No justify questions for module '{current_module.id}'")
    
    question = questions[0]
    
    # Check if LLM service is in mock mode (no API key)
    llm_service = LLMService()
    if llm_service.use_mock:
        # Auto-pass in mock mode without requiring editor input
        console.print()
        console.print(Panel(
            "🎭 [bold yellow]MOCK MODE: No OpenAI API key detected[/bold yellow]\n\n"
            "Justify stage auto-passing for demonstration purposes.\n\n"
            "In production mode (with OPENAI_API_KEY set):\n"
            "• You would answer conceptual questions in your $EDITOR\n"
            "• GPT-4o would evaluate your understanding\n"
            "• Socratic feedback would guide your learning\n\n"
            f"[dim]Question: {question.question[:80]}...[/dim]\n\n"
            "To enable real LLM evaluation, set OPENAI_API_KEY environment variable.\n"
            "Get a key from: https://platform.openai.com/api-keys",
            title="✓ Justify Stage (Mock Mode)",
            border_style="yellow"
        ))
        console.print()
        
        # Advance state to harden
        progress.mark_stage_complete("justify")
        state_mgr.save(progress)
        logger.info(f"Justify stage auto-passed (mock mode) for module '{current_module.id}'")
        
        # Show next action
        console.print(Panel(
            "Next step: Debug a buggy implementation.\n\n"
            "Run [bold cyan]mastery submit[/bold cyan] to continue.",
            title="Next Action",
            border_style="blue"
        ))
        console.print()
        return True
    
    # Production mode: Open editor for answer
    editor = os.getenv('EDITOR', os.getenv('VISUAL', 'nano'))
    
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.md', delete=False) as tf:
        tf.write("# Justify Question\n\n")
        tf.write(f"{question.question}\n\n")
        tf.write("---\n\n")
        tf.write("# Your Answer\n\n")
        tf.write("Write your answer below this line.\n\n")
        temp_path = tf.name
    
    try:
        console.print()
        console.print(f"[bold cyan]Opening editor for your answer...[/bold cyan]")
        console.print(f"[dim]Editor: {editor}[/dim]")
        console.print()
        
        result = subprocess.run([editor, temp_path])
        
        if result.returncode != 0:
            console.print(Panel(
                "[bold yellow]Editor exited with error[/bold yellow]\n\n"
                "Please try again.",
                title="Editor Error",
                border_style="yellow"
            ))
            return False
        
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        # Extract answer (skip header lines, get content after "# Your Answer")
        answer_lines = []
        in_answer = False
        for line in lines:
            if line.strip() == "# Your Answer":
                in_answer = True
                continue
            if in_answer and not line.strip().startswith('#'):
                answer_lines.append(line)
        
        answer = ''.join(answer_lines).strip()
        
        if not answer:
            console.print()
            console.print(Panel(
                "[bold yellow]Empty Answer[/bold yellow]\n\n"
                "You didn't provide an answer. Please try again.",
                title="No Answer",
                border_style="yellow"
            ))
            console.print()
            return False
        
    finally:
        os.unlink(temp_path)
    
    # VALIDATION CHAIN
    # Step A: Fast keyword filter
    matched, fast_feedback = justify_runner.check_fast_filter(question, answer)
    
    if matched:
        # Fast filter caught a shallow/vague answer
        console.print()
        console.print(Panel(
            f"[bold yellow]{fast_feedback}[/bold yellow]",
            title="Needs More Depth",
            border_style="yellow"
        ))
        console.print()
        logger.info(f"Justify response rejected by fast filter for module '{current_module.id}'")
        return False
    
    # Step B: LLM semantic evaluation
    try:
        # llm_service already created above (line 270) for mock mode check
        console.print()
        console.print("[dim]Evaluating your answer...[/dim]")
        
        evaluation = llm_service.evaluate_justification(question, answer)
        
        # Step C: Feedback and state transition
        console.print()
        if evaluation.is_correct:
            console.print(Panel(
                f"[bold green]{evaluation.feedback}[/bold green]",
                title="✓ Correct Understanding",
                border_style="green"
            ))
            console.print()
            
            # Advance state to harden
            progress.mark_stage_complete("justify")
            state_mgr.save(progress)
            
            logger.info(f"Justify stage completed for module '{current_module.id}'")
            
            # Show next action
            console.print(Panel(
                "Next step: Debug a buggy implementation.\n\n"
                "Run [bold cyan]engine submit[/bold cyan] to continue.",
                title="Next Action",
                border_style="blue"
            ))
            console.print()
            return True
        else:
            console.print(Panel(
                f"[bold yellow]{evaluation.feedback}[/bold yellow]",
                title="Not Quite - Try Again",
                border_style="yellow"
            ))
            console.print()
            logger.info(f"Justify response marked incorrect by LLM for module '{current_module.id}'")
            return False
            
    except ConfigurationError as e:
        console.print()
        console.print(Panel(
            f"[bold red]Configuration Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        console.print()
        raise  # Re-raise to be caught by unified submit's exception handler
    except (LLMAPIError, LLMResponseError) as e:
        console.print()
        console.print(Panel(
            f"[bold red]LLM Service Error[/bold red]\n\n{str(e)}\n\n"
            "Your answer was not evaluated. Please try again.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        console.print()
        raise  # Re-raise to be caught by unified submit's exception handler


def _submit_harden_stage(state_mgr, curr_mgr, progress, manifest) -> bool:
    """
    Handle Harden stage submission.
    
    Validates user's bug fix against test suite.
    
    Args:
        state_mgr: State manager instance
        curr_mgr: Curriculum manager instance
        progress: User progress object
        manifest: Curriculum manifest
        
    Returns:
        True if fix validated and progress advanced, False otherwise
    """
    validator_subsys = ValidationSubsystem()
    current_module = manifest.modules[progress.current_module_index]
    
    shadow_worktree = Path('.mastery_engine_worktree')
    harden_workspace = shadow_worktree / "workspace" / "harden"
    
    if not harden_workspace.exists():
        console.print()
        console.print(Panel(
            "[bold red]Harden Workspace Not Found[/bold red]\n\n"
            "You must first prepare the challenge.\n"
            "This will be done automatically when you reach the harden stage.",
            title="ERROR",
            border_style="red"
        ))
        console.print()
        return False
    
    validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
    
    console.print()
    console.print(f"[bold cyan]Running validator on your fix for {current_module.name}...[/bold cyan]")
    console.print()
    
    # Determine file locations
    if current_module.id == "softmax":
        harden_file = harden_workspace / "utils.py"
        shadow_dest = shadow_worktree / "cs336_basics" / "utils.py"
    else:
        harden_file = harden_workspace / f"{current_module.id}.py"
        shadow_dest = shadow_worktree / f"{current_module.id}.py"
    
    # Copy fixed file to shadow worktree
    import shutil
    shadow_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(harden_file, shadow_dest)
    
    result = validator_subsys.execute(validator_path, shadow_worktree)
    
    if result.exit_code == 0:
        # Success!
        console.print(Panel(
            "[bold green]✅ Bug Fixed![/bold green]\n\n"
            "Your fix passes all tests. Well done!",
            title="Success",
            border_style="green"
        ))
        console.print()
        
        # Advance to next module
        progress.mark_stage_complete("harden")
        state_mgr.save(progress)
        
        logger.info(f"Harden stage completed for module '{current_module.id}'")
        
        if progress.current_module_index < len(manifest.modules):
            next_module = manifest.modules[progress.current_module_index]
            console.print(Panel(
                f"Module '{current_module.name}' complete!\n\n"
                f"Next: {next_module.name}\n\n"
                "Run [bold cyan]engine submit[/bold cyan] to start the next module.",
                title="Module Complete",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                "[bold green]🎉 Curriculum Complete![/bold green]\n\n"
                "You have finished all modules!",
                title="Congratulations",
                border_style="green"
            ))
        console.print()
        return True
    else:
        # Failure
        console.print(Panel(
            "[bold red]❌ Fix Incomplete[/bold red]\n\n"
            "The bug is not yet fixed. See test output below:",
            title="Failure",
            border_style="red"
        ))
        console.print()
        console.print("[bold]Test Output:[/bold]")
        console.print(result.stderr if result.stderr else result.stdout)
        console.print()
        
        logger.info(f"Harden validation failed for module '{current_module.id}'")
        return False


# ============================================================================
# Unified Submit Command (P0 CLI Improvement)
# ============================================================================

def _submit_linear_workflow(state_mgr, curr_mgr, progress, manifest) -> None:
    """Handle submission for LINEAR curriculum (module-based)."""
    # Check if curriculum complete
    if _check_curriculum_complete(progress, manifest):
        return
    
    # Get current stage and module
    stage = progress.current_stage
    current_module = manifest.modules[progress.current_module_index]
    
    console.print()
    console.print(f"[dim]Current stage: {stage.upper()} ({current_module.name})[/dim]")
    
    # Route to appropriate handler based on stage
    if stage == "build":
        success = _submit_build_stage(state_mgr, curr_mgr, progress, manifest)
    elif stage == "justify":
        success = _submit_justify_stage(state_mgr, curr_mgr, progress, manifest)
    elif stage == "harden":
        success = _submit_harden_stage(state_mgr, curr_mgr, progress, manifest)
    else:
        console.print(Panel(
            f"[bold red]Unknown Stage[/bold red]\n\n"
            f"Current stage '{stage}' is not recognized.\n"
            "This may be a state file corruption issue.",
            title="ERROR",
            border_style="red"
        ))
        sys.exit(1)
    
    if success:
        logger.info(f"Successfully completed {stage} stage for module '{current_module.id}'")


def _submit_library_workflow(state_mgr, curr_mgr, progress, manifest) -> None:
    """Handle submission for LIBRARY curriculum (problem-based)."""
    # Check if problem selected
    if progress.active_problem_id is None:
        console.print(Panel(
            "[bold yellow]No Problem Selected[/bold yellow]\n\n"
            "You must select a problem before submitting.\n\n"
            "Run [bold cyan]mastery select <pattern> <problem>[/bold cyan]",
            title="LIBRARY MODE",
            border_style="yellow"
        ))
        sys.exit(1)
    
    # Get problem metadata
    problem_lookup = curr_mgr.get_problem_metadata(progress.active_problem_id)
    if problem_lookup is None:
        console.print(Panel(
            f"[bold red]Invalid State[/bold red]\n\n"
            f"Active problem '{progress.active_problem_id}' not found in manifest.",
            title="ERROR",
            border_style="red"
        ))
        sys.exit(1)
    
    pattern_id, problem_meta = problem_lookup
    pattern_meta = curr_mgr.get_pattern_metadata(pattern_id)
    
    stage = progress.current_stage
    console.print()
    console.print(f"[dim]Current stage: {stage.upper()} ({problem_meta.title} - {pattern_meta.title})[/dim]")
    
    # Route to appropriate handler based on stage
    if stage == "build":
        # Validate build implementation
        problem_path = curr_mgr.get_problem_path(progress.curriculum_id, progress.active_problem_id)
        validator_path = problem_path / "validator.sh"
        
        if not validator_path.exists():
            raise CurriculumInvalidError(f"Validator missing for problem '{progress.active_problem_id}': {validator_path}")
        
        validator_subsys = ValidationSubsystem()
        workspace_path = Path.cwd()
        
        console.print(f"[bold cyan]Running validator for {problem_meta.title}...[/bold cyan]")
        console.print()
        
        result = validator_subsys.execute(validator_path, workspace_path)
        
        if result.exit_code == 0:
            console.print(Panel(
                "[bold green]✅ Build Validation Passed![/bold green]\n\n"
                "Your implementation passed all tests.",
                title="Success",
                border_style="green"
            ))
            
            if result.performance_seconds is not None:
                console.print()
                console.print(f"[dim]Performance: {result.performance_seconds:.3f} seconds[/dim]")
            console.print()
            
            # Advance to justify stage
            progress.current_stage = "justify"
            state_mgr.save(progress)
            
            logger.info(f"Build stage completed for problem '{progress.active_problem_id}'")
            
            console.print(Panel(
                "Next step: Answer pattern theory questions.\n\n"
                "Run [bold cyan]mastery show[/bold cyan] to view questions, then [bold cyan]mastery submit[/bold cyan].",
                title="Next Action",
                border_style="blue"
            ))
            console.print()
        else:
            console.print(Panel(
                "[bold red]❌ Validation Failed[/bold red]\n\n"
                "Your implementation did not pass all tests. See details below:",
                title="Failure",
                border_style="red"
            ))
            console.print()
            console.print("[bold]Test Output:[/bold]")
            console.print(result.stderr if result.stderr else result.stdout)
            console.print()
    
    elif stage == "justify":
        # Check if pattern theory already complete
        if pattern_id in progress.completed_patterns:
            # Auto-advance to harden
            console.print(Panel(
                f"[bold green]✓ Pattern Theory Already Complete[/bold green]\n\n"
                f"Advancing to Harden stage...",
                title="Auto-Advance",
                border_style="green"
            ))
            progress.current_stage = "harden"
            state_mgr.save(progress)
            console.print()
            console.print(Panel(
                "Next step: Fix the injected bug.\n\n"
                "Run [bold cyan]mastery start-challenge[/bold cyan] to begin.",
                title="Next Action",
                border_style="blue"
            ))
            console.print()
        else:
            # Validate pattern theory questions
            theory_path = curr_mgr.get_pattern_theory_path(progress.curriculum_id, pattern_id)
            justify_path = theory_path / "justify_questions.json"
            
            if not justify_path.exists():
                raise CurriculumInvalidError(f"Justify questions missing for pattern '{pattern_id}': {justify_path}")
            
            # Load questions
            with open(justify_path, 'r') as f:
                questions_data = json.load(f)
            
            # Use JustifyRunner for validation (it expects ModuleMetadata, so we need to adapt)
            # For now, use the LLM service directly
            console.print(Panel(
                f"[bold cyan]Pattern Theory: {pattern_meta.title}[/bold cyan]\n\n"
                f"Opening your editor to answer {len(questions_data)} conceptual questions...\n\n"
                f"[dim]Your answers will be evaluated for technical accuracy.[/dim]",
                title="Justify Stage",
                border_style="magenta"
            ))
            console.print()
            
            # Open editor for answers (simplified for now - just mark complete)
            # TODO: Implement proper editor integration
            console.print("[yellow]Editor integration pending. For now, marking theory as complete.[/yellow]")
            console.print()
            
            # Mark pattern theory complete
            if pattern_id not in progress.completed_patterns:
                progress.completed_patterns.append(pattern_id)
            progress.current_stage = "harden"
            state_mgr.save(progress)
            
            logger.info(f"Pattern theory completed for '{pattern_id}'")
            
            console.print(Panel(
                "[bold green]✓ Pattern Theory Complete![/bold green]\n\n"
                "This pattern's theory is now unlocked for all problems.\n\n"
                "Next step: Fix the injected bug.\n\n"
                "Run [bold cyan]mastery start-challenge[/bold cyan] to begin.",
                title="Success",
                border_style="green"
            ))
            console.print()
    
    elif stage == "harden":
        # Validate harden fix
        problem_path = curr_mgr.get_problem_path(progress.curriculum_id, progress.active_problem_id)
        
        console.print(f"[bold cyan]Validating fix for {problem_meta.title}...[/bold cyan]")
        console.print()
        
        # Note: HardenRunner expects ModuleMetadata, which LIBRARY mode doesn't have
        # For now, run validator directly in shadow worktree
        validator_path = problem_path / "validator.sh"
        if not validator_path.exists():
            raise CurriculumInvalidError(f"Validator missing: {validator_path}")
        
        validator_subsys = ValidationSubsystem()
        shadow_workspace = SHADOW_WORKTREE_DIR
        
        result = validator_subsys.execute(validator_path, shadow_workspace)
        
        if result.exit_code == 0:
            console.print(Panel(
                "[bold green]✅ Harden Validation Passed![/bold green]\n\n"
                "You successfully fixed the bug!",
                title="Success",
                border_style="green"
            ))
            console.print()
            
            # Mark problem complete
            problem_key = f"{pattern_id}/{progress.active_problem_id}"
            if problem_key not in progress.completed_problems:
                progress.completed_problems.append(problem_key)
            
            # Reset to build stage for next problem
            progress.current_stage = "build"
            # Clear active problem to force selection of next one
            progress.active_problem_id = None
            progress.active_pattern_id = None
            state_mgr.save(progress)
            
            logger.info(f"Problem completed: {problem_key}")
            
            console.print(Panel(
                f"[bold green]🎉 Problem Complete![/bold green]\n\n"
                f"Pattern: {pattern_meta.title}\n"
                f"Problem: {problem_meta.title}\n\n"
                f"Next step: Select another problem.\n\n"
                f"Run [bold cyan]mastery status[/bold cyan] to see available problems.",
                title="Problem Solved",
                border_style="green"
            ))
            console.print()
        else:
            console.print(Panel(
                "[bold red]❌ Validation Failed[/bold red]\n\n"
                "The bug is not fixed yet. See details below:",
                title="Failure",
                border_style="red"
            ))
            console.print()
            console.print("[bold]Test Output:[/bold]")
            console.print(result.stderr if result.stderr else result.stdout)
            console.print()
    
    else:
        console.print(Panel(
            f"[bold red]Unknown Stage[/bold red]\n\n"
            f"Current stage '{stage}' is not recognized.",
            title="ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def submit():
    """
    Submit your work for the current stage (auto-detected).
    
    This command automatically determines which stage you're in and
    runs the appropriate validation:
    
    \b
    - Build stage: Validates your implementation
    - Justify stage: Opens your editor for conceptual answers
    - Harden stage: Validates your bug fix
    
    This replaces the need for separate submit-build, submit-justification,
    and submit-fix commands, providing a consistent workflow throughout
    the Build-Justify-Harden loop.
    """
    try:
        # Ensure shadow worktree exists
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr, curr_mgr, progress, manifest = _load_curriculum_state()
        
        # Route based on curriculum type
        if manifest.type == CurriculumType.LIBRARY:
            _submit_library_workflow(state_mgr, curr_mgr, progress, manifest)
        else:
            _submit_linear_workflow(state_mgr, curr_mgr, progress, manifest)
    
    except (StateFileCorruptedError, CurriculumNotFoundError, CurriculumInvalidError,
            ValidatorNotFoundError, ValidatorTimeoutError, ValidatorExecutionError,
            JustifyQuestionsError, HardenChallengeError, ConfigurationError,
            LLMAPIError, LLMResponseError) as e:
        # Handle all known engine exceptions
        error_type = type(e).__name__.replace('Error', '')
        console.print(Panel(
            f"[bold red]{error_type}[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in submit command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


def _show_linear_content(progress: UserProgress, manifest, curr_mgr: CurriculumManager, module_id: Optional[str]) -> None:
    """Display content for LINEAR curriculum (module-based)."""
    # Determine which module to show
    if module_id is not None:
        # Find specified module
        target_module = None
        for mod in manifest.modules:
            if mod.id == module_id:
                target_module = mod
                break
        
        if target_module is None:
            console.print()
            console.print(Panel(
                f"[bold red]Module Not Found[/bold red]\n\n"
                f"Module '{module_id}' does not exist in curriculum '{manifest.curriculum_name}'.\n\n"
                f"Run [bold cyan]mastery curriculum-list[/bold cyan] to see available modules.",
                title="ERROR",
                border_style="red"
            ))
            console.print()
            sys.exit(1)
        
        current_module = target_module
        display_stage = "build"
    else:
        # Show current module
        if progress.current_module_index >= len(manifest.modules):
            console.print()
            console.print(Panel(
                "[bold green]🎉 Congratulations![/bold green]\n\n"
                "You have completed all modules in this curriculum!",
                title="Curriculum Complete",
                border_style="green"
            ))
            console.print()
            return
        
        current_module = manifest.modules[progress.current_module_index]
        display_stage = progress.current_stage
    
    # Display content based on stage
    if display_stage == "build":
        prompt_path = curr_mgr.get_build_prompt_path(progress.curriculum_id, current_module)
        if not prompt_path.exists():
            raise CurriculumInvalidError(
                f"Build prompt missing for module '{current_module.id}': {prompt_path}"
            )
        
        prompt_content = prompt_path.read_text(encoding='utf-8')
        console.print()
        console.print(Panel(
            Markdown(prompt_content),
            title=f"📝 Build Challenge: {current_module.name}",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()
        logger.info(f"Displayed build prompt for module '{current_module.id}'")
        
    elif display_stage == "justify":
        justify_runner = JustifyRunner(curr_mgr)
        questions = justify_runner.load_questions(progress.curriculum_id, current_module)
        
        if not questions:
            raise CurriculumInvalidError(
                f"No justify questions found for module '{current_module.id}'"
            )
        
        question = questions[0]
        console.print()
        console.print(Panel(
            Markdown(f"**Question:**\n\n{question.question}"),
            title=f"💭 Justify Challenge: {current_module.name}",
            border_style="magenta",
            padding=(1, 2)
        ))
        console.print()
        console.print("[bold cyan]To submit your answer:[/bold cyan] mastery submit")
        console.print()
        logger.info(f"Displayed justify question for module '{current_module.id}'")
        
    elif display_stage == "harden":
        console.print()
        console.print(Panel(
            f"[bold yellow]🐛 Harden Stage: {current_module.name}[/bold yellow]\n\n"
            f"To initialize the debugging workspace and see the bug symptom:\n"
            f"Run [bold cyan]mastery start-challenge[/bold cyan]\n\n"
            f"[dim]This will create a buggy version of your code in the shadow worktree.[/dim]",
            title="Harden Challenge",
            border_style="yellow",
            padding=(1, 2)
        ))
        console.print()
        logger.info(f"Displayed harden instructions for module '{current_module.id}'")
    
    else:
        console.print()
        console.print(Panel(
            f"[bold yellow]Unknown Stage[/bold yellow]\n\n"
            f"Current stage: [bold]{display_stage}[/bold]\n\n"
            f"Run [bold cyan]mastery status[/bold cyan] for progress.",
            title="Info",
            border_style="yellow"
        ))
        console.print()


def _show_library_content(progress: UserProgress, manifest, curr_mgr: CurriculumManager, module_id: Optional[str]) -> None:
    """Display content for LIBRARY curriculum (problem-based)."""
    # Check if problem selected
    if progress.active_problem_id is None:
        console.print()
        console.print(Panel(
            "[bold yellow]No Problem Selected[/bold yellow]\n\n"
            "You must select a problem before viewing content.\n\n"
            "Run [bold cyan]mastery select <pattern> <problem>[/bold cyan]\n"
            "Example: [cyan]mastery select sorting lc_912[/cyan]",
            title="LIBRARY MODE",
            border_style="yellow"
        ))
        console.print()
        sys.exit(1)
    
    # Get problem metadata
    problem_lookup = curr_mgr.get_problem_metadata(progress.active_problem_id)
    if problem_lookup is None:
        console.print(Panel(
            f"[bold red]Invalid State[/bold red]\n\n"
            f"Active problem '{progress.active_problem_id}' not found in manifest.",
            title="ERROR",
            border_style="red"
        ))
        sys.exit(1)
    
    pattern_id, problem_meta = problem_lookup
    pattern_meta = curr_mgr.get_pattern_metadata(pattern_id)
    problem_path = curr_mgr.get_problem_path(progress.curriculum_id, progress.active_problem_id)
    
    # Display based on stage
    display_stage = progress.current_stage
    
    if display_stage == "build":
        # Show build prompt from problem directory
        prompt_path = problem_path / "build_prompt.txt"
        if not prompt_path.exists():
            raise CurriculumInvalidError(
                f"Build prompt missing for problem '{progress.active_problem_id}': {prompt_path}"
            )
        
        prompt_content = prompt_path.read_text(encoding='utf-8')
        console.print()
        console.print(Panel(
            Markdown(prompt_content),
            title=f"📝 Build Challenge: {problem_meta.title}",
            subtitle=f"Pattern: {pattern_meta.title} | Difficulty: {problem_meta.difficulty}",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()
        logger.info(f"Displayed build prompt for problem '{progress.active_problem_id}'")
        
    elif display_stage == "justify":
        # Check if pattern theory already complete
        if pattern_id in progress.completed_patterns:
            console.print()
            console.print(Panel(
                f"[bold green]✓ Pattern Theory Complete[/bold green]\n\n"
                f"You have already completed the theory questions for pattern '{pattern_meta.title}'.\n\n"
                f"Proceed to Harden stage: [cyan]mastery submit[/cyan]",
                title="Theory Complete",
                border_style="green"
            ))
            console.print()
        else:
            # Show pattern theory questions
            theory_path = curr_mgr.get_pattern_theory_path(progress.curriculum_id, pattern_id)
            justify_path = theory_path / "justify_questions.json"
            
            if not justify_path.exists():
                raise CurriculumInvalidError(
                    f"Justify questions missing for pattern '{pattern_id}': {justify_path}"
                )
            
            with open(justify_path, 'r') as f:
                questions_data = json.load(f)
            
            # Show first question
            if questions_data and len(questions_data) > 0:
                question = questions_data[0]
                console.print()
                console.print(Panel(
                    Markdown(f"**Question:**\n\n{question['question']}"),
                    title=f"💭 Pattern Theory: {pattern_meta.title}",
                    subtitle="Answer these once to unlock all problems in this pattern",
                    border_style="magenta",
                    padding=(1, 2)
                ))
                console.print()
                console.print("[bold cyan]To submit your answer:[/bold cyan] mastery submit")
                console.print()
                logger.info(f"Displayed pattern theory for '{pattern_id}'")
            else:
                raise CurriculumInvalidError(f"No questions found in {justify_path}")
        
    elif display_stage == "harden":
        console.print()
        console.print(Panel(
            f"[bold yellow]🐛 Harden Stage: {problem_meta.title}[/bold yellow]\n\n"
            f"Pattern: {pattern_meta.title}\n\n"
            f"To initialize the debugging workspace and see the bug symptom:\n"
            f"Run [bold cyan]mastery start-challenge[/bold cyan]\n\n"
            f"[dim]This will create a buggy version of your code in the shadow worktree.[/dim]",
            title="Harden Challenge",
            border_style="yellow",
            padding=(1, 2)
        ))
        console.print()
        logger.info(f"Displayed harden instructions for problem '{progress.active_problem_id}'")
    
    else:
        console.print()
        console.print(Panel(
            f"[bold yellow]Unknown Stage[/bold yellow]\n\n"
            f"Current stage: [bold]{display_stage}[/bold]\n\n"
            f"Run [bold cyan]mastery status[/bold cyan] for progress.",
            title="Info",
            border_style="yellow"
        ))
        console.print()


@app.command()
def show(module_id: Optional[str] = typer.Argument(None, help="Module ID to display (default: current)")):
    """
    Display challenge content for current or specified module (read-only).
    
    This command is ALWAYS safe and never modifies files.
    Use 'engine start-challenge' to initialize the harden stage workspace.
    
    Examples:
        engine show              # Show current module challenge
        engine show softmax      # Show softmax module content
    """
    try:
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Route based on curriculum type
        if manifest.type == CurriculumType.LIBRARY:
            _show_library_content(progress, manifest, curr_mgr, module_id)
        else:
            _show_linear_content(progress, manifest, curr_mgr, module_id)
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except JustifyQuestionsError as e:
        console.print(Panel(
            f"[bold red]Justify Questions Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command(name="start-challenge")
def start_challenge():
    """
    Initialize the Harden stage workspace (applies bug patch).
    
    This command MODIFIES FILES by creating a buggy version in the shadow worktree.
    Only works in the Harden stage.
    
    Example:
        engine start-challenge   # Apply bug patch and show symptom
    """
    try:
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Check stage
        if progress.current_stage != "harden":
            console.print()
            console.print(Panel(
                f"[bold yellow]Wrong Stage[/bold yellow]\n\n"
                f"The [bold cyan]start-challenge[/bold cyan] command only works in the Harden stage.\n\n"
                f"Current stage: [bold]{progress.current_stage.upper()}[/bold]\n\n"
                f"Use [bold cyan]mastery show[/bold cyan] to view your current challenge.",
                title="Not in Harden Stage",
                border_style="yellow"
            ))
            console.print()
            sys.exit(1)
        
        # Ensure shadow worktree exists
        require_shadow_worktree()
        workspace_mgr = WorkspaceManager()
        harden_runner = HardenRunner(curr_mgr, workspace_mgr)
        
        # Route based on curriculum type
        if manifest.type == CurriculumType.LIBRARY:
            # LIBRARY mode: problem-based
            if progress.active_problem_id is None:
                console.print(Panel(
                    "[bold yellow]No Problem Selected[/bold yellow]\n\n"
                    "You must select a problem before starting a harden challenge.",
                    title="LIBRARY MODE",
                    border_style="yellow"
                ))
                sys.exit(1)
            
            # Get problem metadata
            problem_lookup = curr_mgr.get_problem_metadata(progress.active_problem_id)
            if problem_lookup is None:
                console.print(Panel(
                    f"[bold red]Invalid State[/bold red]\n\n"
                    f"Active problem '{progress.active_problem_id}' not found in manifest.",
                    title="ERROR",
                    border_style="red"
                ))
                sys.exit(1)
            
            pattern_id, problem_meta = problem_lookup
            pattern_meta = curr_mgr.get_pattern_metadata(pattern_id)
            problem_path = curr_mgr.get_problem_path(progress.curriculum_id, progress.active_problem_id)
            
            # Present challenge (WRITES FILES)
            console.print()
            console.print("[bold cyan]Initializing Harden challenge workspace...[/bold cyan]")
            console.print()
            
            harden_file, symptom = harden_runner.present_library_challenge(
                progress.curriculum_id,
                pattern_id,
                problem_meta,
                problem_path
            )
            
            console.print(Panel(
                symptom,
                title=f"🐛 Debug Challenge: {problem_meta.title}",
                subtitle=f"Pattern: {pattern_meta.title}",
                border_style="yellow",
                padding=(1, 2)
            ))
            console.print()
            console.print(f"[bold cyan]📍 Fix the bug in:[/bold cyan] {harden_file}")
            console.print(f"[dim]Your original correct implementation remains safe in the main directory.[/dim]")
            console.print()
            
            logger.info(f"Initialized harden challenge for problem '{progress.active_problem_id}'")
        
        else:
            # LINEAR mode: module-based
            if progress.current_module_index >= len(manifest.modules):
                console.print()
                console.print(Panel(
                    "[bold green]🎉 Congratulations![/bold green]\n\n"
                    "You have completed all modules in this curriculum!",
                    title="Curriculum Complete",
                    border_style="green"
                ))
                console.print()
                return
            
            # Get current module
            current_module = manifest.modules[progress.current_module_index]
            
            # Determine source file based on module
            if current_module.id == "softmax":
                source_file = Path("cs336_basics/utils.py")
            else:
                source_file = Path(f"{current_module.id}.py")
            
            # Present challenge (WRITES FILES)
            console.print()
            console.print("[bold cyan]Initializing Harden challenge workspace...[/bold cyan]")
            console.print()
            
            harden_file, symptom = harden_runner.present_challenge(
                progress.curriculum_id,
                current_module,
                source_file
            )
            
            console.print(Panel(
                symptom,
                title=f"🐛 Debug Challenge: {current_module.name}",
                border_style="yellow",
                padding=(1, 2)
            ))
            console.print()
            console.print(f"[bold cyan]📍 Fix the bug in:[/bold cyan] {harden_file}")
            console.print(f"[dim]Your original correct implementation remains safe in the main directory.[/dim]")
            console.print()
            
            logger.info(f"Initialized harden challenge for module '{current_module.id}'")
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except HardenChallengeError as e:
        console.print(Panel(
            f"[bold red]Harden Challenge Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def next():
    """
    [DEPRECATED] Display the next challenge for the current module.
    
    ⚠️  This command is deprecated. Please use:
        - 'engine show' for read-only viewing (safe, idempotent)
        - 'engine start-challenge' to initialize Harden stage (explicit write)
    
    The 'next' command will be removed in a future version.
    """
    console.print()
    console.print(Panel(
        "[bold yellow]⚠️  Deprecated Command[/bold yellow]\n\n"
        "The [bold]next[/bold] command is deprecated and will be removed in a future version.\n\n"
        "Please use instead:\n"
        "  • [bold cyan]engine show[/bold cyan] - Display current challenge (read-only, safe)\n"
        "  • [bold cyan]engine start-challenge[/bold cyan] - Initialize Harden workspace (explicit write)\n\n"
        "[dim]Running 'engine show' for you...[/dim]",
        title="Deprecation Warning",
        border_style="yellow",
        padding=(1, 2)
    ))
    console.print()
    
    # Forward to show command
    show(module_id=None)


@app.command()
def submit_build():
    """
    [DEPRECATED] Submit and validate your Build stage implementation.
    
    Use 'mastery submit' instead - it auto-detects the build stage.
    This command is maintained for backward compatibility only.
    """
    try:
        # Ensure shadow worktree exists
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        workspace_mgr = WorkspaceManager()
        validator_subsys = ValidationSubsystem()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Check if user has completed all modules
        if progress.current_module_index >= len(manifest.modules):
            console.print()
            console.print(Panel(
                "[bold green]All modules completed![/bold green]\n\n"
                "There are no more build challenges to submit.",
                title="Curriculum Complete",
                border_style="green"
            ))
            console.print()
            return
        
        # Check if user is in build stage
        if progress.current_stage != "build":
            console.print()
            console.print(Panel(
                f"[bold yellow]Not in Build Stage[/bold yellow]\n\n"
                f"You are currently in the [bold]{progress.current_stage.upper()}[/bold] stage.\n"
                f"You can only submit build implementations when in the BUILD stage.\n\n"
                f"Run [bold cyan]engine status[/bold cyan] to see your current progress.",
                title="Wrong Stage",
                border_style="yellow"
            ))
            console.print()
            return
        
        # Get current module
        current_module = manifest.modules[progress.current_module_index]
        
        # Get validator path
        validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
        
        # IN-PLACE EDITING MODEL:
        # User edits files directly in their main directory (e.g., cs336_basics/utils.py)
        # Validator runs from main directory and handles copying to shadow worktree
        workspace_path = Path.cwd()
        
        console.print()
        console.print(f"[bold cyan]Running validator for {current_module.name}...[/bold cyan]")
        console.print()
        
        # Execute validator from current directory
        # The validator script will copy files to shadow worktree for validation
        result = validator_subsys.execute(validator_path, workspace_path)
        
        if result.exit_code == 0:
            # Success!
            console.print(Panel(
                "[bold green]✅ Validation Passed![/bold green]\n\n"
                "Your implementation passed all tests.",
                title="Success",
                border_style="green"
            ))
            
            # Show performance if available
            if result.performance_seconds is not None:
                console.print()
                console.print(f"[dim]Performance: {result.performance_seconds:.3f} seconds[/dim]")
                
                # Check for performance anomaly (novelty detection placeholder)
                if current_module.baseline_perf_seconds:
                    speedup = current_module.baseline_perf_seconds / result.performance_seconds
                    if speedup > 2.0:
                        console.print(f"[yellow]⚡ Impressive! {speedup:.1f}x faster than baseline![/yellow]")
            
            console.print()
            
            # Advance state to next stage
            progress.mark_stage_complete("build")
            state_mgr.save(progress)
            
            logger.info(f"Build stage completed for module '{current_module.id}', "
                       f"advanced to '{progress.current_stage}' stage")
            
            # Show next action
            console.print(Panel(
                "Next step: Answer conceptual questions about your implementation.\n\n"
                "1) [bold cyan]engine next[/bold cyan] — view the justify question\n"
                "2) [bold cyan]engine submit-justification \"<your answer>\"[/bold cyan] — submit your answer",
                title="Next Action",
                border_style="blue"
            ))
            console.print()
        else:
            # Failure - show raw pytest output
            console.print(Panel(
                "[bold red]❌ Validation Failed[/bold red]\n\n"
                "Your implementation did not pass all tests. See details below:",
                title="Failure",
                border_style="red"
            ))
            console.print()
            console.print("[bold]Test Output:[/bold]")
            console.print(result.stderr if result.stderr else result.stdout)
            console.print()
            
            logger.info(f"Build validation failed for module '{current_module.id}'")
            
    except ValidatorNotFoundError as e:
        console.print(Panel(
            f"[bold red]Validator Not Found[/bold red]\n\n{str(e)}\n\n"
            "This is a curriculum configuration error. Please report it.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except ValidatorTimeoutError as e:
        console.print(Panel(
            f"[bold red]Validator Timeout[/bold red]\n\n{str(e)}\n\n"
            "Your code may have an infinite loop or is taking too long to run.",
            title="TIMEOUT ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except ValidatorExecutionError as e:
        console.print(Panel(
            f"[bold red]Validator Execution Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in submit-build command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def submit_justification(answer: str):
    """
    [DEPRECATED] Submit your answer to a Justify stage question.
    
    Use 'mastery submit' instead - it auto-detects the justify stage and opens $EDITOR.
    This command is maintained for backward compatibility only.
    
    Args:
        answer: Your answer to the conceptual question
    """
    # For testing: allow dependency injection via internal mechanism
    llm_service = None  # Will be initialized when needed
    try:
        # Ensure shadow worktree exists (needed for potential LLM calls)
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Check if user has completed all modules
        if progress.current_module_index >= len(manifest.modules):
            console.print()
            console.print(Panel(
                "[bold green]All modules completed![/bold green]\n\n"
                "There are no more justify questions to answer.",
                title="Curriculum Complete",
                border_style="green"
            ))
            console.print()
            return
        
        # Check if user is in justify stage
        if progress.current_stage != "justify":
            console.print()
            console.print(Panel(
                f"[bold yellow]Not in Justify Stage[/bold yellow]\n\n"
                f"You are currently in the [bold]{progress.current_stage.upper()}[/bold] stage.\n"
                f"You can only submit justifications when in the JUSTIFY stage.\n\n"
                f"Run [bold cyan]engine status[/bold cyan] to see your current progress.",
                title="Wrong Stage",
                border_style="yellow"
            ))
            console.print()
            return
        
        # Check for empty answer
        if not answer or not answer.strip():
            console.print()
            console.print(Panel(
                "[bold yellow]Empty Answer[/bold yellow]\n\n"
                "Please provide a non-empty response to the justify question.",
                title="Invalid Input",
                border_style="yellow"
            ))
            console.print()
            return
        
        # Get current module and questions
        current_module = manifest.modules[progress.current_module_index]
        justify_runner = JustifyRunner(curr_mgr)
        questions = justify_runner.load_questions(progress.curriculum_id, current_module)
        
        if not questions:
            raise CurriculumInvalidError(
                f"No justify questions found for module '{current_module.id}'"
            )
        
        question = questions[0]  # For now, use first question
        
        # VALIDATION CHAIN
        # Step A: Fast keyword filter
        matched, fast_feedback = justify_runner.check_fast_filter(question, answer)
        
        if matched:
            # Fast filter caught a shallow/vague answer
            console.print()
            console.print(Panel(
                f"[bold yellow]{fast_feedback}[/bold yellow]",
                title="Needs More Depth",
                border_style="yellow"
            ))
            console.print()
            logger.info(f"Justify response rejected by fast filter for module '{current_module.id}'")
            return
        
        # Step B: LLM semantic evaluation
        try:
            # Initialize LLM service if not provided (dependency injection for testing)
            if llm_service is None:
                llm_service = LLMService()
            
            console.print()
            console.print("[dim]Evaluating your answer...[/dim]")
            
            evaluation = llm_service.evaluate_justification(question, answer)
            
            # Step C: Feedback and state transition
            console.print()
            if evaluation.is_correct:
                console.print(Panel(
                    f"[bold green]{evaluation.feedback}[/bold green]",
                    title="✓ Correct Understanding",
                    border_style="green"
                ))
                console.print()
                
                # Advance state to harden
                progress.mark_stage_complete("justify")
                state_mgr.save(progress)
                
                logger.info(f"Justify stage completed for module '{current_module.id}', "
                           f"advanced to '{progress.current_stage}' stage")
                
                # Show next action
                console.print(Panel(
                    "Next step: Debug a bug in your implementation to harden your knowledge.\n\n"
                    "Run [bold cyan]engine next[/bold cyan] to see the debug challenge.",
                    title="Next Action",
                    border_style="blue"
                ))
                console.print()
            else:
                console.print(Panel(
                    f"[bold yellow]{evaluation.feedback}[/bold yellow]",
                    title="Not Quite - Try Again",
                    border_style="yellow"
                ))
                console.print()
                logger.info(f"Justify response marked incorrect by LLM for module '{current_module.id}'")
                
        except ConfigurationError as e:
            console.print()
            console.print(Panel(
                f"[bold red]Configuration Error[/bold red]\n\n{str(e)}",
                title="ENGINE ERROR",
                border_style="red"
            ))
            console.print()
            sys.exit(1)
        except (LLMAPIError, LLMResponseError) as e:
            console.print()
            console.print(Panel(
                f"[bold red]LLM Service Error[/bold red]\n\n{str(e)}\n\n"
                "Your answer was not evaluated. Please try again.",
                title="ENGINE ERROR",
                border_style="red"
            ))
            console.print()
            sys.exit(1)
            
    except JustifyQuestionsError as e:
        console.print(Panel(
            f"[bold red]Justify Questions Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in submit-justification command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def submit_fix():
    """
    [DEPRECATED] Submit and validate your Harden stage bug fix.
    
    Use 'mastery submit' instead - it auto-detects the harden stage.
    This command is maintained for backward compatibility only.
    """
    try:
        # Ensure shadow worktree exists
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        workspace_mgr = WorkspaceManager()
        validator_subsys = ValidationSubsystem()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Check if user has completed all modules
        if progress.current_module_index >= len(manifest.modules):
            console.print()
            console.print(Panel(
                "[bold green]All modules completed![/bold green]\n\n"
                "There are no more harden challenges to submit.",
                title="Curriculum Complete",
                border_style="green"
            ))
            console.print()
            return
        
        # Check if user is in harden stage
        if progress.current_stage != "harden":
            console.print()
            console.print(Panel(
                f"[bold yellow]Not in Harden Stage[/bold yellow]\n\n"
                f"You are currently in the [bold]{progress.current_stage.upper()}[/bold] stage.\n"
                f"You can only submit bug fixes when in the HARDEN stage.\n\n"
                f"Run [bold cyan]engine status[/bold cyan] to see your current progress.",
                title="Wrong Stage",
                border_style="yellow"
            ))
            console.print()
            return
        
        # Get current module
        current_module = manifest.modules[progress.current_module_index]
        
        # SHADOW WORKTREE MODEL:
        # Harden workspace is in shadow worktree at .mastery_engine_worktree/workspace/harden/
        shadow_worktree = Path('.mastery_engine_worktree')
        harden_workspace = shadow_worktree / "workspace" / "harden"
        
        if not harden_workspace.exists():
            console.print()
            console.print(Panel(
                "[bold red]Harden Workspace Not Found[/bold red]\n\n"
                "You must first view the harden challenge using [bold cyan]engine next[/bold cyan].",
                title="ERROR",
                border_style="red"
            ))
            console.print()
            return
        
        # Get validator path
        validator_path = curr_mgr.get_validator_path(progress.curriculum_id, current_module)
        
        console.print()
        console.print(f"[bold cyan]Running validator on your fix for {current_module.name}...[/bold cyan]")
        console.print()
        
        # CRITICAL: Copy fixed file from harden workspace to shadow worktree root
        # The validator expects files in their normal locations (e.g., cs336_basics/utils.py)
        # Determine source file based on module
        if current_module.id == "softmax":
            harden_file = harden_workspace / "utils.py"
            shadow_dest = shadow_worktree / "cs336_basics" / "utils.py"
        else:
            harden_file = harden_workspace / f"{current_module.id}.py"
            shadow_dest = shadow_worktree / f"{current_module.id}.py"
        
        # Copy fixed file to shadow worktree for validation
        import shutil
        shadow_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(harden_file, shadow_dest)
        
        # Execute validator in shadow worktree
        result = validator_subsys.execute(validator_path, shadow_worktree)
        
        if result.exit_code == 0:
            # Success!
            console.print(Panel(
                "[bold green]✅ Bug Fixed![/bold green]\n\n"
                "Your fix passes all tests. Well done!",
                title="Success",
                border_style="green"
            ))
            console.print()
            
            # Advance state to complete (finish module)
            progress.mark_stage_complete("harden")
            state_mgr.save(progress)
            
            logger.info(f"Harden stage completed for module '{current_module.id}'")
            
            # Show next action
            if progress.current_module_index < len(manifest.modules):
                console.print(Panel(
                    f"Module '{current_module.name}' complete!\n\n"
                    "Run [bold cyan]engine next[/bold cyan] to start the next module.",
                    title="Next Action",
                    border_style="blue"
                ))
            else:
                console.print(Panel(
                    "[bold green]🎉 Curriculum Complete![/bold green]\n\n"
                    "You have finished all modules!",
                    title="Congratulations",
                    border_style="green"
                ))
            console.print()
        else:
            # Failure - show raw output
            console.print(Panel(
                "[bold red]❌ Fix Incomplete[/bold red]\n\n"
                "The bug is not yet fixed. See test output below:",
                title="Failure",
                border_style="red"
            ))
            console.print()
            console.print("[bold]Test Output:[/bold]")
            console.print(result.stderr if result.stderr else result.stdout)
            console.print()
            
            logger.info(f"Harden validation failed for module '{current_module.id}'")
            
    except ValidatorNotFoundError as e:
        console.print(Panel(
            f"[bold red]Validator Not Found[/bold red]\n\n{str(e)}\n\n"
            "This is a curriculum configuration error. Please report it.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except ValidatorTimeoutError as e:
        console.print(Panel(
            f"[bold red]Validator Timeout[/bold red]\n\n{str(e)}\n\n"
            "Your code may have an infinite loop or is taking too long to run.",
            title="TIMEOUT ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except ValidatorExecutionError as e:
        console.print(Panel(
            f"[bold red]Validator Execution Error[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in submit-fix command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def init(
    curriculum_id: str = typer.Argument(..., help="Curriculum ID to initialize (e.g., cp_accelerator)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization (skip already-initialized check)")
):
    """
    Initialize the Mastery Engine with a specific curriculum.
    
    This creates a shadow git worktree for isolated validation and sets up
    the initial learning state. Automatically syncs uncommitted changes to
    the validation environment.
    
    Options:
        --force: Skip the already-initialized check and re-create the shadow worktree
    
    Args:
        curriculum_id: ID of the curriculum to start (e.g., 'cp_accelerator')
        
    Examples:
        mastery init cp_accelerator
        mastery init cp_accelerator --force  # Re-initialize
    """
    try:
        console.print()
        console.print("[bold]Initializing Mastery Engine...[/bold]")
        console.print()
        
        # 1. Check that we're in a Git repository
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            console.print(Panel(
                "[bold red]Not a Git Repository[/bold red]\n\n"
                "The Mastery Engine must be run from within a Git repository.\n"
                "Please ensure you are in the correct directory.",
                title="INITIALIZATION ERROR",
                border_style="red"
            ))
            sys.exit(1)
        
        # 2. Check for uncommitted changes (for snapshot syncing later)
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        has_uncommitted = bool(git_status.stdout.strip())
        
        if has_uncommitted:
            console.print("[yellow]⚠️  Uncommitted changes detected - will sync to validation environment[/yellow]")
        
        # 3. Check if shadow worktree already exists - handle idempotently
        if SHADOW_WORKTREE_DIR.exists() and not force:
            # Check if switching curricula
            try:
                state_mgr = StateManager()
                current_progress = state_mgr.load()
                
                if current_progress.curriculum_id == curriculum_id:
                    # Same curriculum - just inform user
                    console.print(Panel(
                        f"[bold green]Already Initialized[/bold green]\n\n"
                        f"Already using curriculum: [cyan]{curriculum_id}[/cyan]\n\n"
                        "No changes needed. You can continue your learning journey.\n\n"
                        "To re-initialize from scratch, use:\n"
                        "  [cyan]mastery init {curriculum_id} --force[/cyan]",
                        title="✓ Already Set Up",
                        border_style="green"
                    ))
                    return
                else:
                    # Different curriculum - offer to switch
                    console.print(Panel(
                        f"[bold yellow]Curriculum Switch Detected[/bold yellow]\n\n"
                        f"Current: [cyan]{current_progress.curriculum_id}[/cyan]\n"
                        f"Requested: [cyan]{curriculum_id}[/cyan]\n\n"
                        "To switch curricula, first run:\n"
                        "  [cyan]mastery cleanup[/cyan]\n"
                        "Then run init again, or use:\n"
                        "  [cyan]mastery init {curriculum_id} --force[/cyan]",
                        title="CURRICULUM SWITCH",
                        border_style="yellow"
                    ))
                    sys.exit(1)
            except:
                # State file doesn't exist or is corrupt - treat as fresh init
                pass
        
        # 3b. If --force flag is set, clean up existing worktree first
        if force and SHADOW_WORKTREE_DIR.exists():
            console.print("[yellow]--force flag set: Removing existing worktree...[/yellow]")
            try:
                subprocess.run(
                    ["git", "worktree", "remove", str(SHADOW_WORKTREE_DIR), "--force"],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Removed existing worktree at {SHADOW_WORKTREE_DIR}")
            except subprocess.CalledProcessError as e:
                console.print(f"[yellow]Warning: Could not remove worktree (continuing anyway): {e}[/yellow]")
        
        # 3b. Prune any stale worktree registrations
        # (in case worktree was deleted manually without git worktree remove)
        try:
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True,
                check=True
            )
            logger.info("Pruned stale worktree registrations")
        except subprocess.CalledProcessError:
            # Non-critical - continue even if prune fails
            pass
        
        # 4. Verify curriculum exists
        curr_mgr = CurriculumManager()
        try:
            manifest = curr_mgr.load_manifest(curriculum_id)
        except (CurriculumNotFoundError, CurriculumInvalidError) as e:
            console.print(Panel(
                f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
                title="INITIALIZATION ERROR",
                border_style="red"
            ))
            sys.exit(1)
        
        # 5. Create shadow worktree
        console.print("Creating shadow worktree for safe validation...")
        subprocess.run(
            ["git", "worktree", "add", str(SHADOW_WORKTREE_DIR), "--detach"],
            check=True,
            capture_output=True
        )
        logger.info(f"Created shadow worktree at {SHADOW_WORKTREE_DIR}")
        
        # 5a. Recreate cs336_basics symlink in shadow worktree (if it exists in main repo)
        # Git worktrees don't automatically copy symlinks, so we must recreate them
        cs336_symlink = Path("cs336_basics")
        if cs336_symlink.is_symlink():
            # Read the symlink target from main repo
            symlink_target = os.readlink(cs336_symlink)
            
            # Create the same symlink in shadow worktree
            shadow_symlink = SHADOW_WORKTREE_DIR / "cs336_basics"
            if not shadow_symlink.exists():
                os.symlink(symlink_target, shadow_symlink)
                logger.info(f"Recreated cs336_basics symlink in shadow worktree → {symlink_target}")
        
        # 5b. Sync uncommitted changes to shadow worktree (prevents "time travel" bug)
        if has_uncommitted:
            console.print("[cyan]Syncing uncommitted changes to validation environment...[/cyan]")
            
            # Get list of modified tracked files
            dirty_files_output = subprocess.run(
                ["git", "ls-files", "-m"],
                capture_output=True,
                text=True,
                check=True
            )
            
            dirty_files = [f.strip() for f in dirty_files_output.stdout.splitlines() if f.strip()]
            
            # Copy each modified file to shadow worktree
            import shutil
            synced_count = 0
            for file_path in dirty_files:
                src = Path(file_path)
                dst = SHADOW_WORKTREE_DIR / file_path
                
                if src.exists():
                    # Ensure parent directory exists
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    synced_count += 1
            
            logger.info(f"Synced {synced_count} uncommitted files to shadow worktree")
            console.print(f"[green]✓ Synced {synced_count} uncommitted file(s)[/green]")
        
        # 6. Create workspace directory in shadow worktree
        workspace_dir = SHADOW_WORKTREE_DIR / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # 7. Initialize state file
        state_mgr = StateManager()
        initial_state = UserProgress(
            curriculum_id=curriculum_id,
            current_module_index=0,
            current_stage="build",
            completed_modules=[]
        )
        state_mgr.save(initial_state)
        
        # Success!
        console.print()
        
        # Build context-aware success message
        if manifest.type == CurriculumType.LIBRARY:
            content_summary = f"Patterns: {len(manifest.patterns)}"
            next_step = "Run [bold cyan]mastery status[/bold cyan] to see available patterns, then [bold cyan]mastery select <pattern> <problem>[/bold cyan]"
        else:
            content_summary = f"Modules: {len(manifest.modules)}"
            next_step = "Run [bold cyan]mastery show[/bold cyan] to see your first challenge"
        
        console.print(Panel(
            f"[bold green]✓ Initialization Complete![/bold green]\n\n"
            f"Curriculum: [bold]{manifest.curriculum_name}[/bold]\n"
            f"Type: {manifest.type.value.upper()}\n"
            f"{content_summary}\n\n"
            f"Shadow worktree created at: [cyan]{SHADOW_WORKTREE_DIR}[/cyan]\n\n"
            f"You are now ready to begin learning!\n\n"
            f"Next step: {next_step}",
            title="🎓 Mastery Engine Ready",
            border_style="green"
        ))
        console.print()
        
        logger.info(f"Initialized Mastery Engine with curriculum '{curriculum_id}'")
        
    except subprocess.CalledProcessError as e:
        console.print(Panel(
            f"[bold red]Git Error[/bold red]\n\n"
            f"Failed to create shadow worktree: {e}\n\n"
            f"Error output: {e.stderr if e.stderr else 'None'}",
            title="INITIALIZATION ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}",
            title="INITIALIZATION ERROR",
            border_style="red"
        ))
        logger.exception("Initialization failed")
        sys.exit(1)


@app.command(name="curriculum-list")
def curriculum_list():
    """
    List all modules in the current curriculum with their status.
    
    Shows:
        - Module ID and name
        - Status: ✅ Complete, 🔵 In Progress, ⚪ Not Started
        - Dependencies (if any)
    
    Example:
        engine curriculum-list
    """
    try:
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Create status table
        table = Table(
            title=f"📚 {manifest.curriculum_name} - All Modules",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Status", justify="center", style="bold", width=6)
        table.add_column("Module ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Stage", justify="center")
        
        # Add rows for each module
        for idx, module in enumerate(manifest.modules):
            # Determine status
            if module.id in progress.completed_modules:
                status = "✅"
                stage = "Complete"
                stage_style = "green"
            elif idx == progress.current_module_index:
                status = "🔵"
                stage = progress.current_stage.upper()
                stage_style = "yellow"
            else:
                status = "⚪"
                stage = "Not Started"
                stage_style = "dim"
            
            table.add_row(
                status,
                module.id,
                module.name,
                f"[{stage_style}]{stage}[/{stage_style}]"
            )
        
        # Display table
        console.print()
        console.print(table)
        console.print()
        
        # Show summary
        completed_count = len(progress.completed_modules)
        total_count = len(manifest.modules)
        
        console.print(f"Progress: {completed_count}/{total_count} modules completed")
        console.print()
        console.print("[dim]Tip: Use [bold cyan]engine show <module_id>[/bold cyan] to preview any module[/dim]")
        console.print()
        
        logger.info("Displayed curriculum list")
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command(name="progress-reset")
def progress_reset(module_id: str = typer.Argument(..., help="Module ID to reset")):
    """
    Reset progress for a specific module to start over.
    
    This allows you to re-attempt a module to solidify understanding.
    Your work will be preserved but progress will be reset.
    
    Example:
        engine progress-reset softmax    # Reset softmax module
    """
    try:
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Find the module
        target_module = None
        target_index = None
        for idx, module in enumerate(manifest.modules):
            if module.id == module_id:
                target_module = module
                target_index = idx
                break
        
        if target_module is None:
            console.print()
            console.print(Panel(
                f"[bold red]Module Not Found[/bold red]\n\n"
                f"Module '{module_id}' does not exist in curriculum '{manifest.curriculum_name}'.\n\n"
                f"Run [bold cyan]engine curriculum-list[/bold cyan] to see available modules.",
                title="ERROR",
                border_style="red"
            ))
            console.print()
            sys.exit(1)
        
        # Check if module is completed
        is_completed = module_id in progress.completed_modules
        is_current = target_index == progress.current_module_index
        
        if not is_completed and not is_current:
            console.print()
            console.print(Panel(
                f"[bold yellow]Module Not Started[/bold yellow]\n\n"
                f"Module '{module_id}' ({target_module.name}) has not been started yet.\n\n"
                f"There is nothing to reset.",
                title="Info",
                border_style="yellow"
            ))
            console.print()
            return
        
        # Confirm reset
        console.print()
        console.print(Panel(
            f"[bold yellow]⚠️  Reset Module: {target_module.name}[/bold yellow]\n\n"
            f"This will:\n"
            f"  • Remove '{module_id}' from completed modules\n"
            f"  • Set '{module_id}' as your current module\n"
            f"  • Reset stage to 'build'\n\n"
            f"[dim]Your implementation files will NOT be deleted.[/dim]",
            title="Confirm Reset",
            border_style="yellow",
            padding=(1, 2)
        ))
        console.print()
        
        from rich.prompt import Confirm
        if not Confirm.ask(f"Reset module '{module_id}'?", default=False):
            console.print("Reset cancelled.")
            console.print()
            return
        
        # Perform reset
        # Remove from completed modules if present
        progress.completed_modules = [
            m for m in progress.completed_modules if m != module_id
        ]
        
        # Set as current module
        progress.current_module_index = target_index
        progress.current_stage = "build"
        
        # Save state
        state_mgr.save(progress)
        
        # Success message
        console.print()
        console.print(Panel(
            f"[bold green]✅ Module Reset Complete[/bold green]\n\n"
            f"Module '{module_id}' ({target_module.name}) has been reset.\n\n"
            f"You are now in the [bold cyan]BUILD[/bold cyan] stage.\n\n"
            f"Run [bold cyan]engine show[/bold cyan] to see the build challenge.",
            title="Success",
            border_style="green"
        ))
        console.print()
        
        logger.info(f"Reset module '{module_id}' to build stage")
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def reset(module_id: str = None, hard: bool = False):
    """
    Reset progress for a module or the entire curriculum.
    
    Args:
        module_id: Specific module to reset (optional)
        hard: If True, reset entire curriculum to pristine state
    """
    try:
        # Verify shadow worktree exists
        if not SHADOW_WORKTREE_DIR.exists():
            console.print(Panel(
                "[bold yellow]Not Initialized[/bold yellow]\n\n"
                "The Mastery Engine has not been initialized.\n\n"
                "Please run [bold cyan]engine init <curriculum_id>[/bold cyan] first.",
                title="RESET ERROR",
                border_style="yellow"
            ))
            sys.exit(1)
        
        if hard:
            # Hard reset: restore all files from pristine state
            console.print()
            console.print("[bold red]WARNING: Hard reset will discard ALL your work![/bold red]")
            console.print()
            
            from rich.prompt import Confirm
            if not Confirm.ask("Are you sure you want to continue?"):
                console.print("Reset cancelled.")
                return
            
            # Get list of tracked files from shadow worktree
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=str(SHADOW_WORKTREE_DIR),
                capture_output=True,
                text=True,
                check=True
            )
            
            files_to_restore = result.stdout.strip().split('\n')
            
            # Copy each file from shadow worktree to main directory
            for file_path in files_to_restore:
                if not file_path:
                    continue
                src = SHADOW_WORKTREE_DIR / file_path
                dst = Path(file_path)
                if src.exists() and dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"Restored {file_path} to pristine state")
            
            # Reset state file
            state_mgr = StateManager()
            progress = state_mgr.load()
            progress.current_module_index = 0
            progress.current_stage = "build"
            progress.completed_modules = []
            state_mgr.save(progress)
            
            console.print()
            console.print(Panel(
                "[bold green]✓ Hard Reset Complete[/bold green]\n\n"
                "All files have been restored to their pristine state.\n"
                "Your progress has been reset to the beginning.",
                title="Reset Successful",
                border_style="green"
            ))
            console.print()
            
        elif module_id:
            # Module-specific reset
            console.print(Panel(
                "[bold yellow]Module Reset Not Yet Implemented[/bold yellow]\n\n"
                "Module-specific reset requires mapping modules to their affected files.\n"
                "For now, use [bold cyan]--hard[/bold cyan] to reset the entire curriculum.",
                title="RESET ERROR",
                border_style="yellow"
            ))
            sys.exit(1)
        else:
            console.print(Panel(
                "[bold yellow]Invalid Reset Options[/bold yellow]\n\n"
                "Please specify either:\n"
                "  • [cyan]--hard[/cyan] to reset the entire curriculum, or\n"
                "  • [cyan]<module_id>[/cyan] to reset a specific module",
                title="RESET ERROR",
                border_style="yellow"
            ))
            sys.exit(1)
            
    except Exception as e:
        console.print(Panel(
            f"[bold red]Reset Failed[/bold red]\n\n{str(e)}",
            title="RESET ERROR",
            border_style="red"
        ))
        logger.exception("Reset failed")
        sys.exit(1)


@app.command()
def cleanup():
    """
    Clean up the Mastery Engine shadow worktree.
    
    Use this when you are completely finished with the curriculum.
    """
    try:
        if not SHADOW_WORKTREE_DIR.exists():
            console.print("No shadow worktree found. Nothing to clean up.")
            return
        
        console.print()
        console.print("[bold]Cleaning up Mastery Engine...[/bold]")
        
        # Remove the worktree
        subprocess.run(
            ["git", "worktree", "remove", str(SHADOW_WORKTREE_DIR), "--force"],
            check=True,
            capture_output=True
        )
        
        console.print()
        console.print(Panel(
            "[bold green]✓ Cleanup Complete[/bold green]\n\n"
            f"Shadow worktree removed.\n"
            f"State file preserved at: ~/.mastery_progress.json\n\n"
            f"You can re-initialize at any time with [cyan]engine init[/cyan].",
            title="Cleanup Successful",
            border_style="green"
        ))
        console.print()
        
        logger.info("Removed shadow worktree")
        
    except subprocess.CalledProcessError as e:
        console.print(Panel(
            f"[bold red]Git Error[/bold red]\n\n"
            f"Failed to remove shadow worktree: {e}\n\n"
            f"You may need to remove it manually: [cyan]git worktree remove {SHADOW_WORKTREE_DIR} --force[/cyan]",
            title="CLEANUP ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[bold red]Cleanup Failed[/bold red]\n\n{str(e)}",
            title="CLEANUP ERROR",
            border_style="red"
        ))
        logger.exception("Cleanup failed")
        sys.exit(1)


def _show_linear_status(progress: UserProgress, manifest) -> None:
    """Display status for LINEAR curriculum (legacy modules mode)."""
    # Determine current module
    if progress.current_module_index < len(manifest.modules):
        current_module = manifest.modules[progress.current_module_index]
        module_display = f"{current_module.name} ({progress.current_module_index + 1}/{len(manifest.modules)})"
    else:
        module_display = "All modules completed! 🎉"
    
    # Create status table
    table = Table(title="🎓 Mastery Engine Progress", show_header=False, title_style="bold cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    
    table.add_row("Curriculum", manifest.curriculum_name)
    table.add_row("Type", "Linear")
    table.add_row("Author", manifest.author)
    table.add_row("Version", manifest.version)
    table.add_row("Current Module", module_display)
    table.add_row("Current Stage", progress.current_stage.upper())
    table.add_row("Completed Modules", str(len(progress.completed_modules)))
    
    console.print()
    console.print(table)
    console.print()
    
    # Show next action hint
    if progress.current_module_index < len(manifest.modules):
        if progress.current_stage == "build":
            hint = "Run [bold cyan]mastery show[/bold cyan] to see the build prompt"
        elif progress.current_stage == "justify":
            hint = "Run [bold cyan]mastery submit[/bold cyan] to answer the justify questions"
        elif progress.current_stage == "harden":
            hint = "Run [bold cyan]mastery start-challenge[/bold cyan] to begin the harden challenge"
        else:
            hint = "Unknown stage"
        
        console.print(Panel(hint, title="Next Action", border_style="green"))


def _show_library_status(progress: UserProgress, manifest, curr_mgr: CurriculumManager) -> None:
    """Display status for LIBRARY curriculum (patterns/problems mode)."""
    # Check if user has selected a problem
    if progress.active_problem_id is None:
        # Show pattern overview
        console.print()
        console.print(f"[bold cyan]📚 {manifest.curriculum_name}[/bold cyan] - Library Mode")
        console.print(f"[dim]{manifest.description if hasattr(manifest, 'description') else 'Pattern-based problem library'}[/dim]")
        console.print()
        
        # Create patterns summary table
        table = Table(title="Available Patterns", show_header=True, title_style="bold green")
        table.add_column("Pattern", style="bold")
        table.add_column("Progress", justify="right")
        table.add_column("Status", justify="center")
        
        for pattern in manifest.patterns:
            total = len(pattern.problems)
            completed = sum(1 for p in pattern.problems if f"{pattern.id}/{p.id}" in progress.completed_problems)
            
            if pattern.id in progress.completed_patterns:
                theory_status = "✓"
            else:
                theory_status = "○"
            
            progress_str = f"{completed}/{total}"
            status = "✅" if completed == total else "🔵" if completed > 0 else "⚪"
            
            table.add_row(
                f"{pattern.title} ({pattern.id})",
                f"{progress_str} {theory_status}",
                status
            )
        
        console.print(table)
        console.print()
        console.print(Panel(
            "[bold]Next Steps:[/bold]\n\n"
            "1. Select a problem: [cyan]mastery select <pattern> <problem>[/cyan]\n"
            "   Example: [cyan]mastery select sorting lc_912[/cyan]\n\n"
            "Legend: ✓=Theory Complete  ○=Theory Pending  ✅=All Done  🔵=In Progress  ⚪=Not Started",
            title="Getting Started",
            border_style="green"
        ))
        console.print()
    
    else:
        # Show active problem context
        problem_lookup = curr_mgr.get_problem_metadata(progress.active_problem_id)
        if problem_lookup is None:
            console.print(Panel(
                f"[bold yellow]Warning[/bold yellow]\n\n"
                f"Active problem '{progress.active_problem_id}' not found in manifest.\n"
                f"Use [cyan]mastery select[/cyan] to choose a different problem.",
                title="INVALID STATE",
                border_style="yellow"
            ))
            return
        
        pattern_id, problem_meta = problem_lookup
        pattern_meta = curr_mgr.get_pattern_metadata(pattern_id)
        
        # Create context table
        table = Table(title="🎯 Active Problem", show_header=False, title_style="bold cyan")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        
        table.add_row("Curriculum", manifest.curriculum_name)
        table.add_row("Type", "Library")
        table.add_row("Pattern", f"{pattern_meta.title} ({pattern_id})")
        table.add_row("Problem", f"{problem_meta.title} ({progress.active_problem_id})")
        table.add_row("Difficulty", problem_meta.difficulty)
        table.add_row("Current Stage", progress.current_stage.upper())
        
        # Theory status
        theory_status = "✓ Complete" if pattern_id in progress.completed_patterns else "○ Pending"
        table.add_row("Pattern Theory", theory_status)
        
        # Problem status
        problem_key = f"{pattern_id}/{progress.active_problem_id}"
        problem_status = "✅ Complete" if problem_key in progress.completed_problems else "🔵 In Progress"
        table.add_row("Problem Status", problem_status)
        
        console.print()
        console.print(table)
        console.print()
        
        # Show next action hint
        if progress.current_stage == "build":
            hint = "Run [bold cyan]mastery show[/bold cyan] to see the build prompt"
        elif progress.current_stage == "justify":
            if pattern_id not in progress.completed_patterns:
                hint = "Run [bold cyan]mastery show[/bold cyan] to see pattern theory questions"
            else:
                hint = "Pattern theory complete. Advancing to Harden stage..."
        elif progress.current_stage == "harden":
            hint = "Run [bold cyan]mastery start-challenge[/bold cyan] to begin the harden challenge"
        else:
            hint = f"Run [bold cyan]mastery submit[/bold cyan] to proceed"
        
        console.print(Panel(hint, title="Next Action", border_style="green"))


@app.command()
def select(
    pattern: str = typer.Argument(..., help="Pattern ID (e.g., 'sorting', 'hash_table')"),
    problem: str = typer.Argument(..., help="Problem ID (e.g., 'lc_912', 'lc_1')")
):
    """
    Select a problem to work on (LIBRARY mode only).
    
    This command sets the active pattern and problem, resetting you to the Build stage.
    Use this to start a new problem or switch between problems.
    
    Example:
        mastery select sorting lc_912
    """
    try:
        # Ensure shadow worktree exists
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Verify this is a LIBRARY curriculum
        if manifest.type != CurriculumType.LIBRARY:
            console.print(Panel(
                f"[bold yellow]Not Applicable[/bold yellow]\n\n"
                f"The 'select' command is only available for LIBRARY curricula.\n"
                f"Current curriculum '{manifest.curriculum_name}' is type: {manifest.type}\n\n"
                f"For LINEAR curricula, use [cyan]mastery next[/cyan] to progress through modules.",
                title="LIBRARY MODE ONLY",
                border_style="yellow"
            ))
            sys.exit(1)
        
        # Validate pattern exists
        pattern_meta = curr_mgr.get_pattern_metadata(pattern)
        if pattern_meta is None:
            console.print(Panel(
                f"[bold red]Pattern Not Found[/bold red]\n\n"
                f"Pattern '{pattern}' does not exist in curriculum '{manifest.curriculum_name}'.\n\n"
                f"Use [cyan]mastery list[/cyan] to see available patterns.",
                title="INVALID PATTERN",
                border_style="red"
            ))
            sys.exit(1)
        
        # Validate problem exists
        problem_lookup = curr_mgr.get_problem_metadata(problem)
        if problem_lookup is None:
            console.print(Panel(
                f"[bold red]Problem Not Found[/bold red]\n\n"
                f"Problem '{problem}' does not exist in curriculum '{manifest.curriculum_name}'.\n\n"
                f"Use [cyan]mastery list {pattern}[/cyan] to see available problems in this pattern.",
                title="INVALID PROBLEM",
                border_style="red"
            ))
            sys.exit(1)
        
        resolved_pattern, problem_meta = problem_lookup
        
        # Update progress state
        progress.active_pattern_id = pattern
        progress.active_problem_id = problem
        progress.current_stage = "build"
        state_mgr.save(progress)
        
        # Display confirmation
        console.print()
        console.print(Panel(
            f"[bold green]Problem Selected[/bold green]\n\n"
            f"[bold]Pattern:[/bold] {pattern_meta.title}\n"
            f"[bold]Problem:[/bold] {problem_meta.title} ({problem_meta.difficulty})\n"
            f"[bold]Stage:[/bold] Build\n\n"
            f"Run [cyan]mastery show[/cyan] to see the build prompt.",
            title="✓ SELECTION CONFIRMED",
            border_style="green"
        ))
        console.print()
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in select command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command()
def status():
    """
    Display current learning progress.
    
    Shows the active curriculum, current module, current stage in the BJH loop,
    and the list of completed modules.
    """
    try:
        # Ensure shadow worktree exists
        require_shadow_worktree()
        
        # Load state and curriculum
        state_mgr = StateManager()
        curr_mgr = CurriculumManager()
        
        progress = state_mgr.load()
        manifest = curr_mgr.load_manifest(progress.curriculum_id)
        
        # Route based on curriculum type
        if manifest.type == CurriculumType.LIBRARY:
            _show_library_status(progress, manifest, curr_mgr)
        else:
            _show_linear_status(progress, manifest)
        
    except StateFileCorruptedError as e:
        console.print(Panel(
            f"[bold red]State File Corrupted[/bold red]\n\n{str(e)}\n\n"
            f"You may need to delete {StateManager.STATE_FILE} and start over.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumNotFoundError as e:
        console.print(Panel(
            f"[bold red]Curriculum Not Found[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except CurriculumInvalidError as e:
        console.print(Panel(
            f"[bold red]Invalid Curriculum[/bold red]\n\n{str(e)}",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in status command")
        console.print(Panel(
            f"[bold red]Unexpected Error[/bold red]\n\n{str(e)}\n\n"
            f"Check the log file at {Path.home() / '.mastery_engine.log'} for details.",
            title="ENGINE ERROR",
            border_style="red"
        ))
        sys.exit(1)


@app.command(name="create-bug")
def create_bug(
    module: str = typer.Argument(..., help="Module name (e.g., 'softmax')"),
    patch_file: Path = typer.Option(..., "--patch", "-p", help="Path to .patch file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path (default: auto)"),
    symptom_file: Optional[Path] = typer.Option(None, "--symptom", "-s", help="Path to symptom.txt")
):
    """
    [DEV TOOL] Generate JSON bug definition from patch file using LLM.
    
    This tool uses few-shot learning with the golden dataset to automatically
    generate v2.1 JSON bug definitions from legacy .patch files.
    
    Example:
        engine create-bug attention --patch bugs/missing_mask.patch
    """
    try:
        from engine.dev_tools.bug_author import BugAuthor
        
        console.print(Panel(
            "[bold cyan]LLM Bug Authoring Tool[/bold cyan]\n\n"
            f"Module: {module}\n"
            f"Patch: {patch_file}\n\n"
            "Using golden dataset for few-shot learning...",
            title="🤖 Bug Author",
            border_style="cyan"
        ))
        
        # Load symptom if provided
        symptom = ""
        if symptom_file and symptom_file.exists():
            symptom = symptom_file.read_text()
        else:
            symptom = f"Bug in {module} module"
        
        # Initialize bug author with gpt-4o for better reasoning
        from engine.services.llm_service import LLMService
        llm = LLMService(model="gpt-4o")
        author = BugAuthor(llm_service=llm)
        
        # Generate bug definition
        bug_def, success = author.generate_bug_definition(
            module_name=module,
            patch_path=patch_file,
            symptom=symptom
        )
        
        if not success:
            console.print("\n[bold red]❌ Failed to generate valid bug definition[/bold red]")
            sys.exit(1)
        
        # Determine output path
        if output is None:
            output = patch_file.parent / f"{patch_file.stem}.json"
        
        # Write output
        with open(output, 'w') as f:
            json.dump(bug_def, f, indent=2)
        
        console.print(f"\n[bold green]✅ Success![/bold green]\n\nGenerated: {output}")
        
        # Display preview
        console.print("\n[bold]Preview:[/bold]")
        console.print(json.dumps(bug_def, indent=2)[:500] + "...")
        
    except Exception as e:
        logger.exception("Error in create-bug command")
        console.print(Panel(
            f"[bold red]Error:[/bold red]\n\n{str(e)}",
            title="Bug Author Error",
            border_style="red"
        ))
        sys.exit(1)


def main():
    """Entry point for the engine CLI."""
    app()


if __name__ == "__main__":
    main()
