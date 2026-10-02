#!/usr/bin/env python3
"""Backward compatibility adapter for existing ProgressDisplay API.

This adapter bridges the old ProgressDisplay interface to the new UnifiedProgressManager
to maintain compatibility while migrating to the unified system.
"""

from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from whisperjav.utils.unified_progress import UnifiedProgressManager
from whisperjav.utils.logger import logger
from whisperjav.utils.console_gate import get_gate


def _emit_files_progress(file_number: int, total_files: int,
                         pass_number: Optional[int] = None,
                         passes: Optional[int] = None):
    """GUI-only record: total-files bar ("files" channel).

    No-op in terminal runs — the ConsoleGate drops it there. In ensemble
    (two-pass) runs the record carries the pass number, and pct is the
    OVERALL position across all passes, so the GUI files bar advances
    monotonically from the first file of pass 1 to the last of pass 2.
    """
    total_files = max(total_files, 1)
    passes = max(passes or 1, 1)
    pass_number = min(max(pass_number or 1, 1), passes)
    overall = ((pass_number - 1) * total_files + file_number) / (passes * total_files) * 100.0
    if passes > 1:
        label = f"Pass {pass_number}/{passes} · File {file_number}/{total_files}"
    else:
        label = f"File {file_number}/{total_files}"
    get_gate().emit_pipe(
        "files",
        label,
        file=file_number,
        files=total_files,
        **{"pass": pass_number, "passes": passes},
        pct=round(overall, 1),
    )


def _emit_stage_progress(step_name: str, step_number: int, total_steps: int):
    """GUI-only record: pipeline-stage bar ("stage" channel)."""
    total_steps = max(total_steps, 1)
    get_gate().emit_pipe(
        "stage",
        f"Step {step_number}/{total_steps}: {step_name}",
        step=step_number,
        steps=total_steps,
        pct=round(step_number / total_steps * 100.0, 1),
    )


class ProgressDisplayAdapter:
    """Adapter to bridge old ProgressDisplay calls to UnifiedProgressManager."""
    
    def __init__(self, unified_manager: UnifiedProgressManager, total_files: Optional[int] = None):
        self.unified_manager = unified_manager
        # The caller passes the real count. Fall back to whatever the manager was
        # given, then to 1, so a single-file run still reads [1/1].
        if total_files is None:
            total_files = getattr(unified_manager, 'total_files', 1) or 1
        self.total_files = total_files
        self.current_file_context = None
        self.current_step_context = None
        self.current_task_context = None
        self.enabled = True  # Always enabled through unified manager
        
    def close(self):
        """Clean up progress display."""
        self.unified_manager.cleanup()
        
    def update_overall(self, increment: int = 1):
        """Update overall file progress."""
        # This is handled automatically by the unified manager
        # when files are completed
        pass
    
    def set_current_file(self, filename: str, file_number: int):
        """Set current file being processed."""
        self.current_file_context = self.unified_manager.start_file_processing(
            filename, file_number, self.total_files
        )
        _emit_files_progress(file_number, self.total_files)

    def set_current_step(self, step_name: str, step_number: Optional[int] = None, total_steps: int = 5):
        """Set current processing step."""
        if step_number is None:
            step_number = 1

        self.current_step_context = self.unified_manager.start_step(
            step_name, step_number, total_steps, self.current_file_context
        )
        _emit_stage_progress(step_name, step_number, total_steps)
    
    def start_subtask(self, task_name: str, total_items: int):
        """Start a subtask (e.g., scene transcription)."""
        if total_items > 1:  # Only create task context for meaningful subtasks
            self.current_task_context = self.unified_manager.start_task(
                task_name, total_items, self.current_step_context
            )
    
    def update_subtask(self, increment: int = 1):
        """Update subtask progress."""
        if self.current_task_context:
            self.unified_manager.update_task_progress(self.current_task_context, increment)
    
    def finish_subtask(self):
        """Finish the current subtask."""
        if self.current_step_context:
            self.unified_manager.complete_step(self.current_step_context)
        self.current_task_context = None
    
    def show_message(self, message: str, level: str = "info", duration: float = 2.0):
        """Show a message through the logger (unified with overall logging)."""
        # Route messages through the logger to maintain consistency
        # and avoid cluttering the progress display
        log_func = getattr(logger, level, logger.info)
        
        # Only show important messages in quiet/standard modes
        if self.unified_manager.verbosity.value <= 2:  # QUIET or STANDARD
            if level in ["warning", "error"]:
                log_func(message)
        else:  # DETAILED or DEBUG
            log_func(message)
    
    def show_file_complete(self, filename: str, subtitle_count: int, output_path: str):
        """Show file completion message."""
        if self.current_file_context:
            details = {
                'subtitle_count': subtitle_count,
                'output_path': output_path
            }
            self.unified_manager.complete_file(self.current_file_context, success=True, details=details)
            
        # Always log completion for user awareness
        short_name = Path(filename).name
        logger.info(f"✓ {short_name} -> {subtitle_count} subtitle{'s' if subtitle_count != 1 else ''}")
    
    @contextmanager
    def pause_for_input(self):
        """Pause progress display for user input."""
        # The unified manager handles this more gracefully
        # by not interfering with console input
        yield


class DummyProgressAdapter:
    """Dummy adapter that routes everything to logger when progress is disabled."""

    def __init__(self, total_files: Optional[int] = None,
                 pass_number: Optional[int] = None,
                 passes: Optional[int] = None):
        self.total_files = total_files
        # Ensemble pass context (None for single-pass runs): makes the
        # files-channel GUI record pass-aware ("Pass 1/2 · File i/N").
        self.pass_number = pass_number
        self.passes = passes
    
    def close(self):
        pass
    
    def update_overall(self, increment: int = 1):
        pass
    
    def set_current_file(self, filename: str, file_number: int):
        logger.info(f"\nProcessing file {file_number}: {Path(filename).name}")
        _emit_files_progress(file_number, self.total_files or 1,
                             pass_number=self.pass_number, passes=self.passes)

    def set_current_step(self, step_name: str, step_number: Optional[int] = None, total_steps: int = 5):
        if step_number:
            logger.info(f"Step {step_number}/{total_steps}: {step_name}")
            _emit_stage_progress(step_name, step_number, total_steps)
        else:
            logger.info(step_name)
    
    def start_subtask(self, task_name: str, total_items: int):
        if total_items > 1:
            logger.info(f"{task_name} ({total_items} items)")
    
    def update_subtask(self, increment: int = 1):
        pass
    
    def finish_subtask(self):
        pass
    
    def show_message(self, message: str, level: str = "info", duration: float = 2.0):
        getattr(logger, level, logger.info)(message)
    
    def show_file_complete(self, filename: str, subtitle_count: int, output_path: str):
        logger.info(f"✓ Completed: {Path(filename).name} -> {subtitle_count} subtitles")
        logger.info(f"Output saved to: {output_path}")
    
    @contextmanager
    def pause_for_input(self):
        yield


def create_progress_adapter(unified_manager: Optional[UnifiedProgressManager] = None,
                            total_files: Optional[int] = None) -> ProgressDisplayAdapter:
    """Factory function to create appropriate progress adapter."""
    if unified_manager:
        return ProgressDisplayAdapter(unified_manager, total_files)
    else:
        return DummyProgressAdapter()