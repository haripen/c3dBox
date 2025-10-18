"""
Layouts loader/validator for Step 4 check pages.

Updates:
- Aligns sources with provided data structure (point, analog, JRL, IK_markerErr, SO_forces).
- Supports bilateral pairs (left/right), single-series (mono), and multi-series via
  "series" (explicit), "series_regex" (regex over keys in a source group), or "series_glob".
- Rows represent x,y,z where applicable; page-level requirement is to label x-axis as % cycle.

Exposes: load_page(name), available_pages(), LayoutError
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_FILE = Path(__file__).resolve()
LAYOUTS_DIR = THIS_FILE.parent / "layouts"

class LayoutError(ValueError):
    pass

# ---- helpers ---------------------------------------------------------------

def _normalize_name(name: str) -> str:
    n = name.strip().lower().replace(" ", "")
    if n in {"page1", "p1", "1", "anglesmoments"}:
        return "page1"
    if n in {"page2", "p2", "2", "grfsjrfserrorsresiduals"}:
        return "page2"
    return n


def _expect(condition: bool, msg: str) -> None:
    if not condition:
        raise LayoutError(msg)


def _validate_coords(coords: List[int], rows: int, cols: int) -> Tuple[int, int, int]:
    _expect(isinstance(coords, list) and len(coords) in {2, 3},
            "coords must be [row, col] or [row, col, colspan]")
    r, c = coords[0], coords[1]
    span = coords[2] if len(coords) == 3 else 1
    for v in (r, c, span):
        _expect(isinstance(v, int) and v >= 0, "coords values must be non-negative integers")
    _expect(r < rows, f"row {r} outside grid with {rows} rows")
    _expect(c < cols, f"col {c} outside grid with {cols} cols")
    _expect(c + span <= cols, f"colspan {span} at col {c} exceeds grid width {cols}")
    return r, c, span


_ALLOWED_SOURCES = {"point", "analog", "JRL", "IK", "IK_markerErr", "SO_forces", "residual", "grf", "jrf"}


def _validate_series_spec(sp: Dict[str, Any]) -> None:
    """Ensure exactly one way of specifying plotted series is present."""
    bilateral = ("left" in sp and "right" in sp)
    mono = ("mono" in sp)
    series = sp.get("series")
    series_regex = sp.get("series_regex")
    series_glob = sp.get("series_glob")

    ways = int(bilateral) + int(mono) + int(series is not None) + int(series_regex is not None) + int(series_glob is not None)
    _expect(ways == 1, "subplot must use exactly one of: left+right | mono | series | series_regex | series_glob")

    if bilateral:
        _expect(isinstance(sp["left"], str) and isinstance(sp["right"], str),
                "left/right must be strings")
    if mono:
        _expect(isinstance(sp["mono"], str), "mono must be a string")
    if series is not None:
        _expect(isinstance(series, list) and all(isinstance(s, str) for s in series),
                "series must be a list[str]")
    if series_regex is not None:
        _expect(isinstance(series_regex, str) and series_regex != "",
                "series_regex must be a non-empty string")
        try:
            re.compile(series_regex)
        except re.error as e:
            raise LayoutError(f"invalid series_regex: {e}")
    if series_glob is not None:
        _expect(isinstance(series_glob, str) and series_glob != "",
                "series_glob must be a non-empty string")


def _validate_subplot(sp: Dict[str, Any], rows: int, cols: int) -> None:
    for key in ("title", "source", "coords"):
        _expect(key in sp, f"subplot missing required key: {key}")
    _expect(sp["source"] in _ALLOWED_SOURCES, f"unsupported source: {sp['source']}")
    if "component" in sp:
        _expect(sp["component"] in {"x", "y", "z"}, "component must be 'x','y', or 'z'")
    _validate_series_spec(sp)
    _validate_coords(sp["coords"], rows, cols)


def _validate_layout(page: Dict[str, Any]) -> None:
    _expect(isinstance(page, dict), "page must be a JSON object")
    _expect("name" in page and isinstance(page["name"], str), "page.name missing or not a string")
    _expect("grid" in page and isinstance(page["grid"], dict), "page.grid missing or not an object")
    grid = page["grid"]
    for key in ("rows", "cols"):
        _expect(key in grid and isinstance(grid[key], int) and grid[key] > 0,
                f"grid.{key} must be a positive integer")
    rows, cols = grid["rows"], grid["cols"]
    _expect("subplots" in page and isinstance(page["subplots"], list), "page.subplots missing or not a list")

    occupied: set[Tuple[int, int]] = set()
    for sp in page["subplots"]:
        _expect(isinstance(sp, dict), "each subplot must be an object")
        _validate_subplot(sp, rows, cols)
        r, c, _ = _validate_coords(sp["coords"], rows, cols)
        _expect((r, c) not in occupied, f"duplicate subplot at cell ({r},{c})")
        occupied.add((r, c))


def load_page(name: str) -> Dict[str, Any]:
    """Load and validate a page layout by name.

    Args:
        name: e.g., "page1", "page2" (case-insensitive, spaces ignored).
    Returns:
        The parsed JSON dict if valid.
    Raises:
        LayoutError if validation fails, FileNotFoundError if file is missing.
    """
    norm = _normalize_name(name)
    filename = f"{norm}.json" if norm.startswith("page") else f"{name}.json"
    path = LAYOUTS_DIR / filename
    if not path.exists():
        alt = LAYOUTS_DIR / (name.strip().replace(" ", "").lower() + ".json")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Layout file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    _validate_layout(data)
    return data


def available_pages() -> List[str]:
    if not LAYOUTS_DIR.exists():
        return []
    return [p.stem for p in sorted(LAYOUTS_DIR.glob("*.json"))]

__all__ = ["load_page", "available_pages", "LayoutError"]