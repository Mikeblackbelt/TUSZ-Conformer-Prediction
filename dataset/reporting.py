"""
Dataset build summary visualization and terminal progress bar utilities.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_console = Console() if _RICH_AVAILABLE else None


def _bar(value: float, total: float, width: int = 28, fill: str = "█", empty: str = "░") -> str:
    """Render a Unicode progress bar scaled to *total*."""
    if total <= 0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, value / total))
    filled = int(round(frac * width))
    return fill * filled + empty * (width - filled)


def _print_dataset_summary(
    counts: dict,
    label_map: dict,
    label_col_values: pd.Series,
    split_col: Optional[str],
    split_series: Optional[pd.Series],
    binary_preictal: bool,
) -> None:
    """Print a live graphical summary to the terminal using *rich*.
    Falls back to plain print() when rich is not installed."""

    # --- collect stats -------------------------------------------------------
    initial       = counts.get("initial", 0)
    after_channel = counts.get("after_channel", initial)
    after_status  = counts.get("after_status", after_channel)
    after_exlabels = counts.get("after_exclude_labels", after_status)
    after_ictal   = counts.get("after_ictal_filter", after_exlabels)
    after_prefix  = counts.get("after_label_prefix", after_ictal)
    after_conf    = counts.get("after_confidence", after_prefix)
    after_ckpt    = counts.get("after_checkpoints", after_conf)
    final         = after_ckpt

    # Class counts
    class_counts: dict = {}
    for raw_val, mapped in label_map.items():
        mask = label_col_values == raw_val
        class_counts[mapped] = class_counts.get(mapped, 0) + int(mask.sum())

    majority = max(class_counts.values()) if class_counts else 1

    # Split counts
    split_counts: dict = {}
    if split_series is not None:
        for sp in split_series.unique():
            split_counts[sp] = int((split_series == sp).sum())

    # --- render with rich if available ---------------------------------------
    if _RICH_AVAILABLE and _console is not None:
        _console.rule("[bold cyan]EEGWindowDataset — Build Summary[/bold cyan]")

        # 1. Filter funnel
        funnel = Table(title="Filter Funnel", box=box.SIMPLE_HEAD, show_lines=False)
        funnel.add_column("Stage",   style="dim",    no_wrap=True)
        funnel.add_column("Rows",    justify="right")
        funnel.add_column("Dropped", justify="right", style="red")
        funnel.add_column("Bar",     no_wrap=True)

        stages = [
            ("Initial CSV rows",                 initial),
            ("After channel filter",              after_channel),
            ("After status filter",               after_status),
            ("After exact-label exclusion",       after_exlabels),
            ("After ictal-validity filter",       after_ictal),
            ("After label-prefix filter",         after_prefix),
            ("After confidence filter",           after_conf),
            ("After checkpoint check",            after_ckpt),
        ]
        prev = initial
        for label, count in stages:
            dropped = prev - count
            bar = _bar(count, initial, width=24)
            drop_str = f"-{dropped}" if dropped else "—"
            funnel.add_row(label, str(count), drop_str, f"[green]{bar}[/green]")
            prev = count
        _console.print(funnel)

        # 2. Class distribution
        cls_table = Table(title="Class Distribution", box=box.SIMPLE_HEAD, show_lines=False)
        cls_table.add_column("Class",   style="magenta", no_wrap=True)
        cls_table.add_column("Count",   justify="right")
        cls_table.add_column("Pct",     justify="right")
        cls_table.add_column("Bar",     no_wrap=True)

        label_names = {v: k for k, v in label_map.items()} if not binary_preictal else {0: "non-preictal (0)", 1: "preictal (1)"}
        for cls_idx in sorted(class_counts.keys()):
            cnt  = class_counts[cls_idx]
            pct  = 100.0 * cnt / final if final else 0.0
            bar  = _bar(cnt, majority, width=24)
            name = str(label_names.get(cls_idx, cls_idx))
            cls_table.add_row(f"{cls_idx}  {name}", str(cnt), f"{pct:.1f}%", f"[cyan]{bar}[/cyan]")
        _console.print(cls_table)

        # 3. Split breakdown
        if split_counts:
            sp_table = Table(title="Split Breakdown", box=box.SIMPLE_HEAD, show_lines=False)
            sp_table.add_column("Split",  style="yellow", no_wrap=True)
            sp_table.add_column("Count",  justify="right")
            sp_table.add_column("Pct",    justify="right")
            sp_table.add_column("Bar",    no_wrap=True)
            for sp, cnt in sorted(split_counts.items()):
                pct = 100.0 * cnt / final if final else 0.0
                bar = _bar(cnt, final, width=24)
                sp_table.add_row(str(sp), str(cnt), f"{pct:.1f}%", f"[yellow]{bar}[/yellow]")
            _console.print(sp_table)

        _console.rule(f"[bold green]✓ Dataset ready — {final:,} windows — {len(class_counts)} class(es)[/bold green]")

    else:
        # ---- plain-text fallback -------------------------------------------
        W = 50
        print("\n" + "═" * W)
        print(" EEGWindowDataset — Build Summary ".center(W, "═"))
        print("═" * W)
        print("  Filter Funnel:")
        prev = initial
        for label, count in [
            ("Initial CSV rows",                 initial),
            ("After channel filter",              after_channel),
            ("After status filter",               after_status),
            ("After exact-label exclusion",       after_exlabels),
            ("After ictal-validity filter",       after_ictal),
            ("After label-prefix filter",         after_prefix),
            ("After confidence filter",           after_conf),
            ("After checkpoint check",            after_ckpt),
        ]:
            dropped = prev - count
            bar = _bar(count, initial, width=20)
            drop_str = f" (-{dropped})" if dropped else ""
            print(f"    {label:<28} {count:>6}{drop_str}  {bar}")
            prev = count
        print("  Class distribution:")
        label_names = {v: k for k, v in label_map.items()} if not binary_preictal else {0: "non-preictal", 1: "preictal"}
        for cls_idx in sorted(class_counts.keys()):
            cnt  = class_counts[cls_idx]
            pct  = 100.0 * cnt / final if final else 0.0
            bar  = _bar(cnt, majority, width=20)
            name = str(label_names.get(cls_idx, cls_idx))
            print(f"    [{cls_idx}] {name:<20} {cnt:>6}  {pct:5.1f}%  {bar}")
        if split_counts:
            print("  Split breakdown:")
            for sp, cnt in sorted(split_counts.items()):
                pct = 100.0 * cnt / final if final else 0.0
                bar = _bar(cnt, final, width=20)
                print(f"    {sp:<10} {cnt:>6}  {pct:5.1f}%  {bar}")
        print("═" * W + "\n")
