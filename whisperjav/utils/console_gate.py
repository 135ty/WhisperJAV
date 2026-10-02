#!/usr/bin/env python3
"""ConsoleGate: the single owner of console output for WhisperJAV.

Everything that wants to draw a progress bar or print a line during a run
goes through this module. Two display modes are supported and selected
automatically:

- **terminal mode** (``sys.stdout.isatty()``): a single in-place ``\\r``
  progress bar. ``write_line()`` is the ONLY sanctioned way to break the
  bar — it clears the current line, prints the message, and the next
  ``update_bar()`` redraws the bar beneath it.
- **pipe mode** (GUI subprocess, ensemble workers, any redirected
  stdout): no ``\\r`` frames at all (they would only pile up inside the
  pipe's line buffer). Instead, rate-limited single-line structured
  progress records are emitted::

      PROGRESS\\t{"scene": 12, "scenes": 45, "pct": 26.7, "detail": "..."}

  The GUI frontend (``app.js``) recognizes the ``PROGRESS\\t`` prefix and
  feeds the JSON to its own progress bar instead of the console panel.

- **verbose bypass**: when configured with ``verbose=True`` the gate
  steps aside — ``update_bar`` falls back to the legacy raw ``\\r`` print
  and the logger streams normally, so ``--verbosity verbose`` debugging
  behavior is unchanged.

Errors always win: ``write_line`` (and the gate-aware logger handler for
WARNING+) may freely interrupt the bar; INFO records are deferred while a
bar is active (they still reach ``--log-file`` via the separate file
handler).
"""

import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager

__all__ = [
    "ConsoleGate",
    "GateProgressBar",
    "get_gate",
    "configure_console_gate",
    "silence_external_progress",
]

# Structured-progress line prefix understood by the GUI frontend.
PROGRESS_PREFIX = "PROGRESS\t"

# Rate limit for pipe-mode PROGRESS records (seconds).
PIPE_EMIT_INTERVAL = 0.5

# A bar that has not been refreshed for this long is considered dead;
# logger INFO deferral stops so a forgotten end_bar() can never mute the
# console forever.
BAR_STALE_TIMEOUT = 120.0

# Fragments that identify tqdm-style spinner/bar output.
_TQDM_MARKERS = ("%|", "it/s]", "it/s,", "B/s]", "?it/s")


def _looks_like_tqdm(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _TQDM_MARKERS)


def _has_error_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in ("error", "failed", "exception", "traceback"))


def _format_bar(current: int, total: int, width: int = 24) -> str:
    """ASCII bar in the house style: '=' filled + '-' remainder."""
    filled = int(width * current / total) if total else 0
    return "=" * filled + "-" * (width - filled)


class _QuietStream:
    """Replacement stdout/stderr used inside ``gate.quiet()``.

    Drops everything the running model stack emits (tqdm frames, loading
    logs, warnings-as-noise) EXCEPT lines that carry an error keyword —
    those break through via ``gate.write_line`` so real problems are
    never lost.
    """

    def __init__(self, gate, original):
        self._gate = gate
        self._original = original
        self._buffer = ""

    def write(self, s):
        try:
            if not s:
                return 0
            self._buffer += s
            if "\n" not in self._buffer:
                # Incomplete line — hold it back; tqdm frames never send
                # a newline, so noise dies silently here.
                if len(self._buffer) > 4096:
                    self._buffer = self._buffer[-256:]
                return len(s)
            lines = self._buffer.split("\n")
            self._buffer = lines.pop()
            for line in lines:
                if _has_error_keyword(line):
                    self._gate.write_line(line.rstrip())
            return len(s)
        except Exception:
            return len(s) if s else 0

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        # Some libraries (tqdm included) call fileno(); delegate so they
        # do not crash, they just see the real descriptor.
        return self._original.fileno()

    def writable(self):
        return True


class ConsoleGate:
    """Single owner of console rendering. See module docstring."""

    def __init__(self):
        self._lock = threading.RLock()
        self._enabled = True       # False when --no-progress
        self._verbose = False      # bypass: legacy behavior
        self._bar_text = ""
        self._bar_active = False
        self._last_width = 0
        self._last_update = 0.0
        self._last_pipe_emit = 0.0
        # Real output stream while quiet() has stdout/stderr replaced —
        # write_line/update_bar must bypass the filter streams, not feed
        # them (which would recurse/drop their own output).
        self._direct_stream = None
        # Active top-level progress scope (file-level: scene X of N).
        # Inner bars (GateProgressBar) are rendered COMBINED with it, so
        # the console line / GUI progress always shows the big picture:
        #   "Transcribing: 12/45 [26%] | ASR Text Gen 3/8"
        self._scope_label = ""
        self._scope_current = 0
        self._scope_total = 0
        self._scope_start = 0.0
        # Reentrancy guard: update_scope renders through update_bar and
        # must not be prefixed with itself.
        self._scope_rendering = False
        # Per-channel last-emit timestamps for emit_pipe() rate limiting.
        self._pipe_channel_emit: dict = {}

    def _out(self):
        """The stream gate output goes to (never a quiet() filter stream)."""
        return self._direct_stream if self._direct_stream is not None else sys.stdout

    # ------------------------------------------------------------------
    # Top-level scope (file-level scene progress)
    # ------------------------------------------------------------------

    def scope_active(self) -> bool:
        return self._scope_total > 0

    def update_scope(self, current: int, total: int, label: str = ""):
        """Set/advance the file-level scope (scene ``current`` of ``total``).

        Renders its own bar line; inner bars rendered afterwards are
        prefixed with this scope so the display stays one line.
        """
        if not self._enabled:
            return
        with self._lock:
            if label:
                self._scope_label = label
            if not self._scope_start or total != self._scope_total or current < self._scope_current:
                self._scope_start = time.time()
            self._scope_current = current
            self._scope_total = total
        pct = (current / total * 100.0) if total else 0.0
        self._scope_rendering = True
        try:
            self.update_bar(
                self._scope_text(),
                scene=current, scenes=total, pct=round(pct, 1),
                eta=self._scope_eta(),
            )
        finally:
            self._scope_rendering = False

    def clear_scope(self):
        """Drop the file-level scope (end of the scoped phase)."""
        with self._lock:
            self._scope_label = ""
            self._scope_current = 0
            self._scope_total = 0
            self._scope_start = 0.0

    def _scope_pct(self) -> float:
        if self._scope_total:
            return self._scope_current / self._scope_total * 100.0
        return 0.0

    def _scope_eta(self):
        """ETA seconds for the scope, or None when not yet measurable."""
        if not self._scope_total or self._scope_current < 1:
            return None
        elapsed = time.time() - self._scope_start
        if elapsed <= 0:
            return None
        per = elapsed / self._scope_current
        remaining = (self._scope_total - self._scope_current) * per
        return round(remaining)

    def _scope_text(self) -> str:
        pct = self._scope_pct()
        eta = self._scope_eta()
        bar = _format_bar(self._scope_current, self._scope_total)
        label = self._scope_label or "Processing"
        eta_text = ""
        if eta is not None:
            eta_text = f" | ETA: {eta / 60:.1f}m" if eta > 60 else f" | ETA: {eta:.0f}s"
        return f"{label}: [{bar}] {self._scope_current}/{self._scope_total} [{pct:.1f}%]{eta_text}"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, enabled: bool = True, verbose: bool = False):
        """Set gate behavior. Called once at process start."""
        with self._lock:
            self._enabled = enabled
            self._verbose = verbose
            self._bar_active = False
            self._bar_text = ""
            self._last_width = 0

    @property
    def verbose(self) -> bool:
        return self._verbose

    # ------------------------------------------------------------------
    # Mode detection
    # ------------------------------------------------------------------

    def _is_terminal(self) -> bool:
        try:
            return self._out().isatty()
        except Exception:
            return False

    def _terminal_width(self) -> int:
        try:
            return max(shutil.get_terminal_size((120, 20)).columns - 1, 20)
        except Exception:
            return 119

    def _bar_is_live(self) -> bool:
        """True while a terminal bar is on the line and not stale."""
        return (
            self._bar_active
            and not self._verbose
            and self._enabled
            and (time.time() - self._last_update) < BAR_STALE_TIMEOUT
        )

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    def update_bar(self, text: str, **fields):
        """Draw (or emit) the progress bar. The only bar renderer.

        ``fields`` carry structured progress (scene, scenes, pct, eta...)
        used by pipe-mode consumers (the GUI progress bar). An explicit
        ``channel`` field routes the record to a specific GUI bar (e.g.
        ``"asr"``); records without one belong to the main bar.
        """
        if not self._enabled:
            return

        # Combine with the active file-level scope: inner bars (e.g. the
        # per-scene "ASR Text Gen" bar) are prefixed with the scene-level
        # picture, and pipe-mode records carry the file-level numbers.
        # Channelled records (GUI sub-bars) are exempt from the fold, but
        # ONLY in pipe mode — the terminal display keeps the combined line.
        if self.scope_active() and not self._scope_rendering:
            if self._is_terminal():
                text = f"{self._scope_text()} | {text}"
            elif not fields.get("channel"):
                # Pipe consumers (GUI) see ONLY the file-level scope records.
                # Fast inner frames are folded into the scope and emit nothing
                # of their own, so the GUI progress advances once per scene
                # instead of racing with the in-block bar.
                return

        if self._verbose:
            # Bypass: legacy raw in-place print.
            try:
                out = self._out()
                out.write(f"\r{text}")
                out.flush()
            except Exception:
                pass
            return

        if self._is_terminal():
            width = self._terminal_width()
            line = text[:width]
            with self._lock:
                try:
                    out = self._out()
                    # Pad to the previous frame's width so a shrinking
                    # bar does not leave residue on the line.
                    pad = " " * max(self._last_width - len(line), 0)
                    out.write("\r" + line + pad)
                    out.flush()
                except Exception:
                    return
                self._bar_text = line
                self._bar_active = True
                self._last_width = len(line)
                self._last_update = time.time()
        else:
            # Pipe mode: rate-limited structured progress records.
            # Channelled records (GUI sub-bars) go through emit_pipe's
            # PER-CHANNEL limiter — the shared one below would let busy
            # main-bar records starve the sub-bars.
            if fields.get("channel"):
                inner = dict(fields)
                channel = inner.pop("channel")
                self.emit_pipe(channel, text, **inner)
                return
            now = time.time()
            if now - self._last_pipe_emit < PIPE_EMIT_INTERVAL and not fields.get("final"):
                return
            payload = dict(fields)
            payload.setdefault("detail", text)
            payload.setdefault("channel", "main")
            try:
                payload["pct"] = round(float(payload.get("pct", 0.0)), 1)
            except (TypeError, ValueError):
                pass
            line = PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False)
            with self._lock:
                try:
                    out = self._out()
                    out.write(line + "\n")
                    out.flush()
                except Exception:
                    return
                self._last_pipe_emit = now

    def emit_pipe(self, channel: str, text: str, **fields):
        """Emit a GUI-only structured progress record on ``channel``.

        No-op unless running in pipe mode (GUI subprocess) — the terminal
        display is untouched. Used for the GUI's auxiliary bars: total
        files ("files"), pipeline stage ("stage"), and any other channel
        the frontend knows about. Rate-limited per channel.
        """
        if not self._enabled or self._verbose or self._is_terminal():
            return
        now = time.time()
        if now - self._pipe_channel_emit.get(channel, 0.0) < PIPE_EMIT_INTERVAL and not fields.get("final"):
            return
        payload = dict(fields)
        payload["channel"] = channel
        payload.setdefault("detail", text)
        try:
            payload["pct"] = round(float(payload.get("pct", 0.0)), 1)
        except (TypeError, ValueError):
            pass
        line = PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False)
        with self._lock:
            self._pipe_channel_emit[channel] = now
            try:
                out = self._out()
                out.write(line + "\n")
                out.flush()
            except Exception:
                return

    def write_line(self, text: str):
        """Print a full line, allowed to break the bar (errors, [DONE])."""
        if self._verbose or not self._enabled or not self._is_terminal():
            try:
                out = self._out()
                out.write(text + "\n")
                out.flush()
            except Exception:
                pass
            return

        with self._lock:
            try:
                out = self._out()
                if self._bar_active:
                    # Erase the bar, print the message, redraw the bar.
                    clear = " " * self._last_width
                    out.write("\r" + clear + "\r" + text + "\n")
                    if self._bar_text:
                        out.write("\r" + self._bar_text)
                else:
                    out.write(text + "\n")
                out.flush()
            except Exception:
                return
            self._last_update = time.time()

    def end_bar(self, final_text: str = "", **fields):
        """Finish the current bar (newline in terminal, final PROGRESS record)."""
        if not self._enabled:
            return

        if self._verbose:
            try:
                out = self._out()
                out.write("\n")
                out.flush()
            except Exception:
                pass
            return

        if self._is_terminal():
            with self._lock:
                if self._bar_active:
                    try:
                        out = self._out()
                        out.write("\n")
                        out.flush()
                    except Exception:
                        pass
                self._bar_active = False
                self._bar_text = ""
                self._last_width = 0
            if final_text:
                self.write_line(final_text)
        else:
            if fields or final_text:
                fields.setdefault("final", True)
                self._last_pipe_emit = 0.0  # force emit
                self.update_bar(final_text, **fields)

    # ------------------------------------------------------------------
    # Logger integration hook
    # ------------------------------------------------------------------

    def defer_info(self) -> bool:
        """True while INFO console output should be suppressed (bar live)."""
        return self._bar_is_live()

    # ------------------------------------------------------------------
    # External output suppression
    # ------------------------------------------------------------------

    @contextmanager
    def quiet(self):
        """Silence external model output; error lines still break through."""
        # In verbose bypass the user wants everything — no filtering.
        if self._verbose:
            yield
            return

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        # Route gate output (write_line/update_bar) to the real stream so
        # error break-through lines land on the terminal/pipe, not back
        # into the filter streams.
        self._direct_stream = old_stdout
        sys.stdout = _QuietStream(self, old_stdout)
        sys.stderr = _QuietStream(self, old_stderr)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self._direct_stream = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class GateProgressBar:
    """Minimal tqdm-compatible facade that renders through the ConsoleGate.

    Lets call sites swap ``tqdm(...)`` for ``GateProgressBar(...)`` with a
    one-line change: ``set_description``/``update`` render via
    ``update_bar`` (terminal bar or pipe-mode PROGRESS record), and leaving
    the ``with`` block ends the bar cleanly.
    """

    def __init__(self, total=None, desc="", **kwargs):
        self.total = total or 0
        self.n = 0
        self.desc = desc
        # Optional GUI channel (e.g. "asr"): pipe-mode records carry it so
        # the GUI can render this bar separately from the main one.
        self.channel = kwargs.get("channel")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        get_gate().end_bar()
        return False

    def set_description(self, desc=None):
        if desc is not None:
            self.desc = desc
        self._render()

    def update(self, n=1):
        self.n += n
        self._render()

    def _render(self):
        # Same bar style as the file-level scope bar, so the combined line
        # reads e.g.
        #   Transcribing: [======----] 12/45 [26.7%] | ETA: 3.2m | ASR Text Gen: [============] 8/8
        fields = dict(scene=self.n, scenes=self.total, pct=self._pct())
        if self.channel:
            fields["channel"] = self.channel
        get_gate().update_bar(
            f"{self.desc}: [{_format_bar(self.n, self.total)}] {self.n}/{self.total}",
            **fields,
        )

    def _pct(self):
        return round(self.n / self.total * 100.0, 1) if self.total else 0.0


_gate: ConsoleGate = ConsoleGate()


def get_gate() -> ConsoleGate:
    """Return the process-wide console gate."""
    return _gate


def configure_console_gate(enabled: bool = True, verbose: bool = False):
    """Configure the process-wide gate (see ConsoleGate.configure)."""
    _gate.configure(enabled=enabled, verbose=verbose)


def silence_external_progress():
    """Shut external libraries' own progress output at the source.

    - huggingface_hub download bars (tqdm)
    - transformers model-loading logs / bars

    Best-effort: every step is guarded so a missing or newer library
    cannot break startup.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore

        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass
