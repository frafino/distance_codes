from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .geometry import Edge, Vertex, Orientation
from .tile import TileType


X_COLOR = "#F2C94C"
Z_COLOR = "#1687D9"
BOTH_COLOR = "#22BD25"
LOGICAL_COLOR = "#9B51E0"
GRID_COLOR = "0.82"
LAYOUT_COLOR = "0.45"


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _vertex_xy(v: Vertex, shift=(0, 0)):
    return v.x + shift[0], v.y + shift[1]


def _edge_xy(edge: Edge, shift=(0, 0)):
    x, y = _vertex_xy(edge.origin, shift)
    if edge.orientation is Orientation.HORIZONTAL:
        return (x, y), (x + 2, y)
    return (x, y), (x, y + 2)


def _support_bounds(edges, anchors=()):
    xs, ys = [], []
    for edge in edges:
        for x, y in _edge_xy(edge):
            xs.append(x)
            ys.append(y)
    for anchor in anchors:
        xs.append(anchor.x)
        ys.append(anchor.y)
    if not xs:
        return 0, 0, 0, 0
    return min(xs), max(xs), min(ys), max(ys)


def _display_shift(edges, anchors=()):
    xmin, _, ymin, _ = _support_bounds(edges, anchors)
    return -xmin, -ymin


def _anchor_vertex(code, i, j):
    return Vertex(2 * (int(i) - code.padding), 2 * (int(j) - code.padding))


def _mask_anchors(code, mask):
    return {_anchor_vertex(code, i, j) for i, j in np.argwhere(mask)}


def _all_anchors(code):
    return (_mask_anchors(code, code.bulk_mask)
            | _mask_anchors(code, code.x_boundary_mask)
            | _mask_anchors(code, code.z_boundary_mask))


def _union_support(checks):
    return set().union(*(set(support) for support in checks.values())) if checks else set()


def _physical_check_positions(code, x_checks, z_checks):
    x_positions = {anchor.add(code.x_tile.check_qubit) for anchor in x_checks}
    z_positions = {anchor.add(code.z_tile.check_qubit) for anchor in z_checks}
    return x_positions, z_positions


def _tile_background_edges(B):
    return {Edge(Vertex(2 * x, 2 * y), orientation)
            for orientation in Orientation for x in range(B) for y in range(B)}


# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------

def _format_axes(ax, edges, anchors=(), shift=(0, 0), title=None):
    xmin, xmax, ymin, ymax = _support_bounds(edges, anchors)
    ax.set_aspect("equal")
    ax.set_xlim(xmin + shift[0] - 0.5, xmax + shift[0] + 0.5)
    ax.set_ylim(ymin + shift[1] - 0.5, ymax + shift[1] + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)


def _integer_grid(ax, edges, anchors, shift):
    xmin, xmax, ymin, ymax = _support_bounds(edges, anchors)
    ax.set_xticks(range(int(xmin + shift[0]), int(xmax + shift[0]) + 1))
    ax.set_yticks(range(int(ymin + shift[1]), int(ymax + shift[1]) + 1))
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)


def _new_axes(ax, size):
    if ax is None:
        _, ax = plt.subplots(figsize=(size, size))
    return ax


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_edge(ax, edge, color, *, shift=(0, 0), lw=3.0, linestyle="-", zorder=3):
    (x0, y0), (x1, y1) = _edge_xy(edge, shift)
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, linestyle=linestyle,
            solid_capstyle="round", zorder=zorder)


def _draw_layout(ax, layout, *, shift=(0, 0)):
    for edge in layout:
        _draw_edge(ax, edge, LAYOUT_COLOR, shift=shift, lw=0.9, linestyle="--", zorder=1)


def _draw_support_sets(ax, x_edges, z_edges, *, shift=(0, 0), lw=3.0):
    x_set, z_set = set(x_edges), set(z_edges)
    both = x_set & z_set
    for edge in x_set - both:
        _draw_edge(ax, edge, X_COLOR, shift=shift, lw=lw, zorder=4)
    for edge in z_set - both:
        _draw_edge(ax, edge, Z_COLOR, shift=shift, lw=lw, zorder=4)
    for edge in both:
        _draw_edge(ax, edge, BOTH_COLOR, shift=shift, lw=lw, zorder=5)


def _scatter(ax, vertices, color, label, shift, **kwargs):
    if not vertices:
        return
    xy = [_vertex_xy(v, shift) for v in vertices]
    ax.scatter([x for x, _ in xy], [y for _, y in xy], color=color, label=label, **kwargs)


def _draw_checks(ax, x_positions, z_positions, *, shift=(0, 0), size=50):
    x_positions, z_positions = set(x_positions), set(z_positions)
    both = x_positions & z_positions
    style = dict(s=size, marker="s", edgecolors="black", linewidths=0.35, zorder=7)
    _scatter(ax, x_positions - both, X_COLOR, "X check", shift, **style)
    _scatter(ax, z_positions - both, Z_COLOR, "Z check", shift, **style)
    _scatter(ax, both, BOTH_COLOR, "X & Z checks", shift, **style)


# ---------------------------------------------------------------------------
# Tile plots
# ---------------------------------------------------------------------------

def _plot_tile(tile, *, ax=None, title=None):
    ax = _new_axes(ax, 5)
    background = _tile_background_edges(tile.B)
    for edge in background:
        _draw_edge(ax, edge, GRID_COLOR, lw=0.9, zorder=0)

    is_x = tile.type is TileType.X
    for edge in tile.edges:
        _draw_edge(ax, edge, X_COLOR if is_x else Z_COLOR, lw=3.5, zorder=4)

    _format_axes(ax, background | set(tile.edges), title=title or ("X tile" if is_x else "Z tile"))
    return ax


def plot_x_tile(code_or_tile, *, ax=None, title="X tile"):
    """Plot the X tile of a code (or an X Tile directly)."""
    tile = getattr(code_or_tile, "x_tile", code_or_tile)
    if tile.type is not TileType.X:
        raise ValueError("plot_x_tile expects an X tile.")
    return _plot_tile(tile, ax=ax, title=title)


def plot_z_tile(code_or_tile, *, ax=None, title="Z tile"):
    """Plot the Z tile of a code (or a Z Tile directly)."""
    tile = getattr(code_or_tile, "z_tile", code_or_tile)
    if tile.type is not TileType.Z:
        raise ValueError("plot_z_tile expects a Z tile.")
    return _plot_tile(tile, ax=ax, title=title)


# ---------------------------------------------------------------------------
# Edge sequences and logical operators
# ---------------------------------------------------------------------------

def plot_edges(edges, *, B=None, layout=None, ax=None, title="Edge support",
               color=LOGICAL_COLOR, lw=3.5, shift=(0, 0)):
    """Plot a set of edges over a B x B tile grid or over a given layout."""
    if B is not None and layout is not None:
        raise ValueError("Supply either B or layout, not both.")

    edges = tuple(dict.fromkeys(edges))
    ax = _new_axes(ax, 5)

    if B is not None:
        background = _tile_background_edges(B)
        for edge in background:
            _draw_edge(ax, edge, GRID_COLOR, shift=shift, lw=0.9, zorder=0)
    else:
        background = tuple(layout) if layout is not None else ()
        _draw_layout(ax, background, shift=shift)

    for edge in edges:
        _draw_edge(ax, edge, color, shift=shift, lw=lw, zorder=8)

    _format_axes(ax, set(background) | set(edges), shift=shift, title=title)
    return ax


def plot_logical_operator(edges, *, code, ax=None, title="Logical Operator", lw=3.5):
    """Plot the support of a logical operator on the code layout."""
    ax = _new_axes(ax, 7)
    layout = tuple(code.layout)
    anchors = _all_anchors(code)
    shift = _display_shift(layout, anchors)

    _draw_layout(ax, layout, shift=shift)
    for edge in dict.fromkeys(edges):
        _draw_edge(ax, edge, LOGICAL_COLOR, shift=shift, lw=lw, zorder=8)

    _format_axes(ax, layout, anchors, shift=shift, title=title)
    _integer_grid(ax, layout, anchors, shift)
    return ax


# ---------------------------------------------------------------------------
# Anchor plot
# ---------------------------------------------------------------------------

def plot_anchors(code, *, ax=None, title="Tile Anchors"):
    """Plot the bulk, X-boundary and Z-boundary tile anchors on the layout."""
    ax = _new_axes(ax, 7)
    anchors = _all_anchors(code)
    shift = _display_shift(code.layout, anchors)

    _draw_layout(ax, code.layout, shift=shift)
    style = dict(s=46, zorder=5)
    _scatter(ax, _mask_anchors(code, code.bulk_mask), BOTH_COLOR, "X & Z anchor", shift, **style)
    _scatter(ax, _mask_anchors(code, code.x_boundary_mask), X_COLOR, "X anchor", shift, **style)
    _scatter(ax, _mask_anchors(code, code.z_boundary_mask), Z_COLOR, "Z anchor", shift, **style)

    _format_axes(ax, code.layout, anchors, shift=shift, title=title)
    _integer_grid(ax, code.layout, anchors, shift)
    ax.legend(loc="best", frameon=True)
    return ax


# ---------------------------------------------------------------------------
# Tessellation structures
# ---------------------------------------------------------------------------

def checks_before_prune(code):
    """Return the X and Z checks restricted to the layout, before pruning."""
    x_checks = code.tessellate(code.x_tile, code.x_anchor_mask, code.layout)
    z_checks = code.tessellate(code.z_tile, code.z_anchor_mask, code.layout)
    return x_checks, z_checks


def full_support_before_layout_restriction(code):
    """Return the full X and Z tile supports at every anchor, before layout restriction."""
    def translate(tile, mask):
        checks = {}
        for i, j in np.argwhere(mask):
            anchor = _anchor_vertex(code, i, j)
            checks[anchor] = tuple(edge.translated(anchor) for edge in tile.edges)
        return checks

    return translate(code.x_tile, code.x_anchor_mask), translate(code.z_tile, code.z_anchor_mask)


# ---------------------------------------------------------------------------
# Support visualizations
# ---------------------------------------------------------------------------

def _plot_support(code, x_checks, z_checks, extra_edges, *, ax, title, show_checks, grid_edges=None):
    ax = _new_axes(ax, 10)
    x_support, z_support = _union_support(x_checks), _union_support(z_checks)
    all_edges = set(code.layout) | set(extra_edges) | x_support | z_support
    x_positions, z_positions = _physical_check_positions(code, x_checks, z_checks)
    anchors = x_positions | z_positions
    shift = _display_shift(all_edges, anchors)

    _draw_layout(ax, code.layout, shift=shift)
    _draw_support_sets(ax, x_support, z_support, shift=shift)
    if show_checks:
        _draw_checks(ax, x_positions, z_positions, shift=shift)

    _format_axes(ax, all_edges, anchors, shift=shift, title=title)
    _integer_grid(ax, all_edges if grid_edges is None else grid_edges, anchors, shift)
    if show_checks:
        ax.legend(loc="best", frameon=True)
    return ax


def plot_support_before_prune(code, *, ax=None, title="Support Before Pruning", show_checks=True):
    """Plot the full X/Z check supports and check qubits before pruning."""
    x_checks, z_checks = full_support_before_layout_restriction(code)
    return _plot_support(code, x_checks, z_checks, (), ax=ax, title=title, show_checks=show_checks)


def plot_support_after_prune(code, *, ax=None, title="Support After Pruning", show_checks=True):
    """Plot the X/Z check supports and check qubits after pruning."""
    data, x_checks, z_checks = code.parity_checks
    return _plot_support(code, x_checks, z_checks, data, ax=ax, title=title,
                         show_checks=show_checks, grid_edges=set(code.layout))


# ---------------------------------------------------------------------------
# Convenience diagnostic
# ---------------------------------------------------------------------------

def print_visualization_summary(code):
    """Print the number of layout edges, checks and data qubits before and after pruning."""
    before_x, before_z = checks_before_prune(code)
    data, after_x, after_z = code.parity_checks

    print(f"layout edges:           {len(code.layout)}")
    print(f"X checks before prune:  {len(before_x)}")
    print(f"Z checks before prune:  {len(before_z)}")
    print(f"data after prune:       {len(data)}")
    print(f"X checks after prune:   {len(after_x)}")
    print(f"Z checks after prune:   {len(after_z)}")
