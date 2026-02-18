"""
Rich console utilities for beautiful terminal output across AQUA.

Every script imports the shared ``console`` object and helper functions
from here so the look-and-feel is consistent.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.text import Text
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Banner / headers
# ---------------------------------------------------------------------------

def banner(title, subtitle=None):
    """Print the AQUA banner at the start of a script."""
    text = Text(title, style="bold white")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    console.print(
        Panel(text, border_style="bright_blue", box=box.DOUBLE_EDGE,
              padding=(1, 4)),
    )


def section(title, step=None):
    """Print a section header rule.

    Args:
        title: section description.
        step: optional step number (e.g. 1, 2, 3).
    """
    if step is not None:
        label = f"Step {step}: {title}"
    else:
        label = title
    console.print()
    console.rule(f"[bold cyan]{label}[/bold cyan]", style="cyan")


# ---------------------------------------------------------------------------
# Config display
# ---------------------------------------------------------------------------

def print_config(cfg, extra=None):
    """Print a configuration summary as a rich table inside a panel.

    Args:
        cfg: Config object (attribute-access dict).
        extra: optional dict of additional key-value pairs to display.
    """
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Key", style="bold cyan", min_width=14)
    table.add_column("Value", style="green")

    table.add_row("Dataset",
                  f"{cfg.dataset}  [dim]({cfg.num_classes} classes, "
                  f"{cfg.img_size}\u00d7{cfg.img_size})[/dim]")
    table.add_row("Model", cfg.model)
    table.add_row("Bit-width", str(cfg.bits))
    table.add_row("Batch size", str(cfg.batch_size))
    if hasattr(cfg, "fp32_epochs"):
        table.add_row("FP32 epochs", str(cfg.fp32_epochs))
        table.add_row("FP32 LR", str(cfg.fp32_lr))
    if hasattr(cfg, "epochs"):
        table.add_row("QAT epochs", str(cfg.epochs))
        table.add_row("QAT LR", str(cfg.lr))
    if extra:
        for k, v in extra.items():
            table.add_row(k, str(v))

    console.print(Panel(table, title="[bold]Configuration[/bold]",
                        border_style="blue", box=box.ROUNDED))


# ---------------------------------------------------------------------------
# Key-value metrics
# ---------------------------------------------------------------------------

def metric(key, value, style="bold green"):
    """Print a single metric line."""
    console.print(f"  [dim]\u2022[/dim] [cyan]{key}:[/cyan] [{style}]{value}[/{style}]")


def success(msg):
    """Print a success message."""
    console.print(f"  [bold green]\u2714[/bold green] {msg}")


def warning(msg):
    """Print a warning message."""
    console.print(f"  [bold yellow]\u26a0[/bold yellow] {msg}")


def info(msg):
    """Print an informational message."""
    console.print(f"  [dim]\u2022[/dim] {msg}")


def file_saved(path):
    """Print a 'file saved' message."""
    console.print(f"  [bold green]\u2714[/bold green] Saved [cyan]{path}[/cyan]")


# ---------------------------------------------------------------------------
# Training progress
# ---------------------------------------------------------------------------

def training_progress(total_epochs):
    """Return a rich Progress bar for epoch-level training.

    Usage::

        with training_progress(200) as progress:
            task = progress.add_task("Training", total=200)
            for epoch in range(1, 201):
                ...
                progress.update(task, advance=1,
                    description=f"Epoch {epoch}  Loss={loss:.4f}  Val={acc:.1f}%")
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]", justify="left"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("[dim]\u2502[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def results_table(rows, title="Results"):
    """Print a results table.

    Args:
        rows: list of dicts with keys that become column headers.
        title: table title.
    """
    if not rows:
        return
    table = Table(title=title, box=box.ROUNDED, border_style="bright_blue",
                  header_style="bold cyan")
    for key in rows[0]:
        justify = "right" if any(isinstance(r[key], (int, float)) for r in rows) else "left"
        table.add_column(key, justify=justify)

    for row in rows:
        table.add_row(*[_fmt(v) for v in row.values()])

    console.print(table)


def _fmt(v):
    """Format a value for display in a table cell."""
    if v is None:
        return "[dim]N/A[/dim]"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)
