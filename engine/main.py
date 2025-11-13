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
from engine.schemas import UserProgress
from engine.validator import (
    ValidationSubsystem,
    ValidatorNotFoundError,
    ValidatorTimeoutError,
    ValidatorExecutionError
)
from engine.stages.harden import HardenRunner, HardenChallengeError
from engine.stages.justify import JustifyRunner, JustifyQuestionsError
from engine.services.llm_service import LLMService, ConfigurationError, LLMAPIError, LLMResponseError
import subprocess


# Initialize Typer app and Rich console
app = typer.Typer(
    name="engine",
    help="Mastery Engine - A sophisticated pedagogical framework for deep technical learning",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / '.mastery_engine.log'),
        logging.StreamHandler()
    ]
)

# Shadow worktree configuration
SHADOW_WORKTREE_DIR = Path(".mastery_engine_worktree")

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
    
    # Open editor for answer
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
                    f"Run [bold cyan]engine curriculum list[/bold cyan] to see available modules.",
                    title="ERROR",
                    border_style="red"
                ))
                console.print()
                sys.exit(1)
            
            current_module = target_module
            # For specified modules, default to showing build content
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
        
        # Display content based on stage (READ-ONLY)
        if display_stage == "build":
            # Display build prompt
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
            # Display justify question (READ-ONLY)
            justify_runner = JustifyRunner(curr_mgr)
            questions = justify_runner.load_questions(progress.curriculum_id, current_module)
            
            if not questions:
                raise CurriculumInvalidError(
                    f"No justify questions found for module '{current_module.id}'"
                )
            
            # Show first question
            question = questions[0]
            
            console.print()
            console.print(Panel(
                Markdown(f"**Question:**\n\n{question.question}"),
                title=f"💭 Justify Challenge: {current_module.name}",
                border_style="magenta",
                padding=(1, 2)
            ))
            console.print()
            console.print("[bold cyan]To submit your answer:[/bold cyan] engine submit")
            console.print()
            
            logger.info(f"Displayed justify question for module '{current_module.id}'")
            
        elif display_stage == "harden":
            # Display harden symptom (READ-ONLY - does NOT apply patch)
            console.print()
            console.print(Panel(
                f"[bold yellow]🐛 Harden Stage: {current_module.name}[/bold yellow]\n\n"
                f"To initialize the debugging workspace and see the bug symptom:\n"
                f"Run [bold cyan]engine start-challenge[/bold cyan]\n\n"
                f"[dim]This will create a buggy version of your code in the shadow worktree.[/dim]",
                title="Harden Challenge",
                border_style="yellow",
                padding=(1, 2)
            ))
            console.print()
            
            logger.info(f"Displayed harden instructions for module '{current_module.id}'")
        
        else:
            # Unknown stage
            console.print()
            console.print(Panel(
                f"[bold yellow]Unknown Stage[/bold yellow]\n\n"
                f"Current stage: [bold]{display_stage}[/bold]\n\n"
                f"Run [bold cyan]engine status[/bold cyan] for progress.",
                title="Info",
                border_style="yellow"
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
                f"Use [bold cyan]engine show[/bold cyan] to view your current challenge.",
                title="Not in Harden Stage",
                border_style="yellow"
            ))
            console.print()
            sys.exit(1)
        
        # Check completion
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
        
        # Ensure shadow worktree exists
        require_shadow_worktree()
        workspace_mgr = WorkspaceManager()
        harden_runner = HardenRunner(curr_mgr, workspace_mgr)
        
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
    Submit and validate your Build stage implementation.
    
    Runs the validator script against your code, shows results,
    and advances your progress if validation passes.
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
    Submit your answer to a Justify stage question.
    
    Implements the Validation Chain:
    1. Fast keyword filter (catches shallow/vague answers)
    2. LLM semantic evaluation (if no keyword match)
    
    Args:
        answer: Your conceptual explanation
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
    Submit and validate your Harden stage bug fix.
    
    Runs the validator against your debugged code, shows results,
    and advances your progress if the fix is correct.
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
def init(curriculum_id: str):
    """
    Initialize the Mastery Engine for a curriculum.
    
    Creates a shadow Git worktree for safe, isolated validation.
    This is the required first step before using the engine.
    
    Args:
        curriculum_id: ID of the curriculum to start (e.g., 'cs336_a1')
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
        
        # 2. Check for clean working directory
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            console.print(Panel(
                "[bold yellow]Uncommitted Changes Detected[/bold yellow]\n\n"
                "The Mastery Engine requires a clean Git working directory to initialize.\n\n"
                "Please commit or stash your current changes first:\n"
                "  [cyan]git add -A && git commit -m 'Pre-mastery checkpoint'[/cyan]\n"
                "  or\n"
                "  [cyan]git stash push -m 'Pre-mastery work'[/cyan]",
                title="INITIALIZATION ERROR",
                border_style="yellow"
            ))
            sys.exit(1)
        
        # 3. Check if shadow worktree already exists
        if SHADOW_WORKTREE_DIR.exists():
            console.print(Panel(
                "[bold yellow]Already Initialized[/bold yellow]\n\n"
                f"The shadow worktree already exists at {SHADOW_WORKTREE_DIR}.\n\n"
                "If you want to re-initialize, first run:\n"
                "  [cyan]engine cleanup[/cyan]",
                title="INITIALIZATION ERROR",
                border_style="yellow"
            ))
            sys.exit(1)
        
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
        console.print(Panel(
            f"[bold green]✓ Initialization Complete![/bold green]\n\n"
            f"Curriculum: [bold]{manifest.curriculum_name}[/bold]\n"
            f"Modules: {len(manifest.modules)}\n\n"
            f"Shadow worktree created at: [cyan]{SHADOW_WORKTREE_DIR}[/cyan]\n\n"
            f"You are now ready to begin learning!\n\n"
            f"Next step: Run [bold cyan]engine next[/bold cyan] to see your first challenge.",
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
                hint = "Run [bold cyan]engine next[/bold cyan] to see the build prompt"
            elif progress.current_stage == "justify":
                hint = "Run [bold cyan]engine submit-justify[/bold cyan] to answer the justify questions"
            elif progress.current_stage == "harden":
                hint = "Run [bold cyan]engine submit-harden[/bold cyan] to fix the injected bug"
            else:
                hint = "Unknown stage"
            
            console.print(Panel(hint, title="Next Action", border_style="green"))
        
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


def main():
    """Entry point for the engine CLI."""
    app()


if __name__ == "__main__":
    main()
