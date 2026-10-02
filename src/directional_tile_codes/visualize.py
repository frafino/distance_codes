from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .geometry import Edge, Vertex, Orientation
from .tile import TileType


X_COLOR = "#F2C94C"
Z_COLOR = "#1687D9"
BOTH_COLOR = "#22BD25"
GRID_COLOR = "0.82"
LAYOUT_COLOR = "0.45"


# ---------------------------------------------------------------------------
# Basic coordinate helpers
# ---------------------------------------------------------------------------

def _vertex_xy(v: Vertex, shift=(0, 0)):
    sx, sy = shift
    return v.x / 2 + sx, v.y / 2 + sy


def _edge_xy(edge: Edge, shift=(0, 0)):
    x, y = _vertex_xy(edge.origin, shift)

    if edge.orientation is Orientation.HORIZONTAL:
        return (x, y), (x + 1, y)

    return (x, y), (x, y + 1)


def _support_bounds(edges, anchors=()):
    xs = []
    ys = []

    for e in edges:
        (x0, y0), (x1, y1) = _edge_xy(e)
        xs.extend([x0, x1])
        ys.extend([y0, y1])

    for a in anchors:
        x, y = _vertex_xy(a)
        xs.append(x)
        ys.append(y)

    if not xs:
        return 0, 0, 0, 0

    return min(xs), max(xs), min(ys), max(ys)


def _display_shift(edges, anchors=()):
    xmin, _, ymin, _ = _support_bounds(edges, anchors)
    return -xmin, -ymin


def _format_axes(ax, edges, anchors=(), shift=(0, 0), title=None):
    xmin, xmax, ymin, ymax = _support_bounds(edges, anchors)

    xmin += shift[0]
    xmax += shift[0]
    ymin += shift[1]
    ymax += shift[1]

    ax.set_aspect("equal")
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is not None:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Generic drawing primitives
# ---------------------------------------------------------------------------

def _draw_edge(ax, edge, color, *, shift=(0, 0), lw=3.0, linestyle="-", zorder=3):
    (x0, y0), (x1, y1) = _edge_xy(edge, shift)
    ax.plot(
        [x0, x1],
        [y0, y1],
        color=color,
        lw=lw,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=zorder,
    )


def _draw_layout(ax, layout, *, shift=(0, 0)):
    for edge in layout:
        _draw_edge(
            ax,
            edge,
            LAYOUT_COLOR,
            shift=shift,
            lw=0.9,
            linestyle="--",
            zorder=1,
        )


def _draw_support_sets(ax, x_edges, z_edges, *, shift=(0, 0), lw=3.0):
    x_set = set(x_edges)
    z_set = set(z_edges)

    both = x_set & z_set

    for edge in x_set - both:
        _draw_edge(ax, edge, X_COLOR, shift=shift, lw=lw, zorder=4)

    for edge in z_set - both:
        _draw_edge(ax, edge, Z_COLOR, shift=shift, lw=lw, zorder=4)

    for edge in both:
        _draw_edge(ax, edge, BOTH_COLOR, shift=shift, lw=lw, zorder=5)


def _draw_checks(ax, x_checks, z_checks, *, shift=(0, 0), size=50):
    x_anchors = set(x_checks)
    z_anchors = set(z_checks)
    both = x_anchors & z_anchors

    def scatter(vertices, color, label):
        if not vertices:
            return
        xy = [_vertex_xy(v, shift) for v in vertices]
        ax.scatter(
            [x for x, _ in xy],
            [y for _, y in xy],
            s=size,
            marker="s",
            color=color,
            edgecolors="black",
            linewidths=0.35,
            label=label,
            zorder=7,
        )

    scatter(x_anchors - both, X_COLOR, "X check")
    scatter(z_anchors - both, Z_COLOR, "Z check")
    scatter(both, BOTH_COLOR, "X & Z checks")


# ---------------------------------------------------------------------------
# Tile plots
# ---------------------------------------------------------------------------

def _tile_background_edges(B):
    edges = set()

    for x in range(B):
        for y in range(B):
            edges.add(
                Edge(Vertex(2 * x, 2 * y), Orientation.HORIZONTAL)
            )

    for x in range(B):
        for y in range(B):
            edges.add(
                Edge(Vertex(2 * x, 2 * y), Orientation.VERTICAL)
            )

    return edges


def _plot_tile(tile, *, ax=None, title=None):
    """
    Plot one existing Tile support using tile.get_edges().
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    background = _tile_background_edges(tile.B)

    for edge in background:
        _draw_edge(ax, edge, GRID_COLOR, lw=0.9, zorder=0)

    color = X_COLOR if tile.type is TileType.X else Z_COLOR

    for edge in tile.get_edges():
        _draw_edge(ax, edge, color, lw=3.5, zorder=4)

    ax.scatter([0], [0], s=55, color=color, zorder=6)

    all_edges = set(background) | set(tile.get_edges())
    _format_axes(
        ax,
        all_edges,
        title=title or ("X tile" if tile.type is TileType.X else "Z tile"),
    )
    return ax


def plot_x_tile(code_or_tile, *, ax=None, title="X tile"):
    tile = code_or_tile.x_tile if hasattr(code_or_tile, "x_tile") else code_or_tile

    if tile.type is not TileType.X:
        raise ValueError("plot_x_tile expects an X tile.")

    return _plot_tile(tile, ax=ax, title=title)


def plot_z_tile(code_or_tile, *, ax=None, title="Z tile"):
    tile = code_or_tile.z_tile if hasattr(code_or_tile, "z_tile") else code_or_tile

    if tile.type is not TileType.Z:
        raise ValueError("plot_z_tile expects a Z tile.")

    return _plot_tile(tile, ax=ax, title=title)


# ---------------------------------------------------------------------------
# Anchor plot
# ---------------------------------------------------------------------------

def _mask_anchors(code, mask):
    result = set()

    for i, j in np.argwhere(mask):
        result.add(
            Vertex(
                2 * (int(i) - code.padding),
                2 * (int(j) - code.padding),
            )
        )

    return result


def plot_anchors(code, *, ax=None, title="Tile anchors"):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    bulk = _mask_anchors(code, code.bulk_mask)
    x_boundary = _mask_anchors(code, code.x_boundary_mask)
    z_boundary = _mask_anchors(code, code.z_boundary_mask)

    all_anchors = bulk | x_boundary | z_boundary
    shift = _display_shift(code.layout, all_anchors)

    _draw_layout(ax, code.layout, shift=shift)

    def scatter(vertices, color, label):
        if not vertices:
            return
        xy = [_vertex_xy(v, shift) for v in vertices]
        ax.scatter(
            [x for x, _ in xy],
            [y for _, y in xy],
            s=46,
            color=color,
            label=label,
            zorder=5,
        )

    scatter(bulk, BOTH_COLOR, "X & Z anchor")
    scatter(x_boundary, X_COLOR, "X anchor")
    scatter(z_boundary, Z_COLOR, "Z anchor")

    _format_axes(
        ax,
        code.layout,
        all_anchors,
        shift=shift,
        title=title,
    )

    all_edges = set(code.layout)

    xmin, xmax, ymin, ymax = _support_bounds(all_edges, all_anchors)

    xmin = int(xmin + shift[0])
    xmax = int(xmax + shift[0])
    ymin = int(ymin + shift[1])
    ymax = int(ymax + shift[1])

    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
        zorder=0,
    )
    ax.legend(loc="best", frameon=True,)
    return ax


# ---------------------------------------------------------------------------
# Existing tessellation structures
# ---------------------------------------------------------------------------

def checks_before_prune(code):
    x_checks = code.tessellate(
        code.x_tile,
        code.x_anchor_mask,
        code.layout,
    )
    z_checks = code.tessellate(
        code.z_tile,
        code.z_anchor_mask,
        code.layout,
    )
    return x_checks, z_checks


def full_support_before_layout_restriction(code):
    x_checks = {}
    z_checks = {}

    for i, j in np.argwhere(code.x_anchor_mask):
        anchor = Vertex(
            2 * (int(i) - code.padding),
            2 * (int(j) - code.padding),
        )
        x_checks[anchor] = tuple(
            edge.translated(anchor)
            for edge in code.x_tile.get_edges()
        )

    for i, j in np.argwhere(code.z_anchor_mask):
        anchor = Vertex(
            2 * (int(i) - code.padding),
            2 * (int(j) - code.padding),
        )
        z_checks[anchor] = tuple(
            edge.translated(anchor)
            for edge in code.z_tile.get_edges()
        )

    return x_checks, z_checks


def _union_support(checks):
    if not checks:
        return set()
    return set().union(*(set(support) for support in checks.values()))


# ---------------------------------------------------------------------------
# Support visualizations
# ---------------------------------------------------------------------------

def plot_support_before_prune(
    code,
    *,
    ax=None,
    title="Support before pruning",
    show_checks=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    x_checks, z_checks = full_support_before_layout_restriction(code)

    x_support = _union_support(x_checks)
    z_support = _union_support(z_checks)

    all_edges = set(code.layout) | x_support | z_support
    all_anchors = set(x_checks) | set(z_checks)

    shift = _display_shift(all_edges, all_anchors)

    _draw_layout(ax, code.layout, shift=shift)
    _draw_support_sets(ax, x_support, z_support, shift=shift)

    if show_checks:
        _draw_checks(ax, x_checks, z_checks, shift=shift)

    _format_axes(
        ax,
        all_edges,
        all_anchors,
        shift=shift,
        title=title,
    )

    xmin, xmax, ymin, ymax = _support_bounds(all_edges, all_anchors)

    xmin = int(xmin + shift[0])
    xmax = int(xmax + shift[0])
    ymin = int(ymin + shift[1])
    ymax = int(ymax + shift[1])

    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
        zorder=0,
    )

    if show_checks:
        ax.legend(loc="best", frameon=True,)

    return ax

def plot_support_after_prune(
    code,
    *,
    ax=None,
    title="Support after pruning",
    show_checks=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    data, x_checks, z_checks = code.build_checks()

    x_support = _union_support(x_checks)
    z_support = _union_support(z_checks)

    all_edges = set(code.layout) | set(data) | x_support | z_support
    all_anchors = set(x_checks) | set(z_checks)

    shift = _display_shift(all_edges, all_anchors)

    _draw_layout(ax, code.layout, shift=shift)
    _draw_support_sets(ax, x_support, z_support, shift=shift)

    if show_checks:
        _draw_checks(ax, x_checks, z_checks, shift=shift)

    _format_axes(
        ax,
        all_edges,
        all_anchors,
        shift=shift,
        title=title,
    )

    all_edges = set(code.layout)

    xmin, xmax, ymin, ymax = _support_bounds(all_edges, all_anchors)

    xmin = int(xmin + shift[0])
    xmax = int(xmax + shift[0])
    ymin = int(ymin + shift[1])
    ymax = int(ymax + shift[1])

    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
        zorder=0,
    )

    if show_checks:
        ax.legend(loc="best", frameon=True,)

    return ax


# ---------------------------------------------------------------------------
# Convenience diagnostic
# ---------------------------------------------------------------------------

def print_visualization_summary(code):
    before_x, before_z = checks_before_prune(code)
    data, after_x, after_z = code.build_checks()

    print(f"layout edges:           {len(code.layout)}")
    print(f"X checks before prune:  {len(before_x)}")
    print(f"Z checks before prune:  {len(before_z)}")
    print(f"data after prune:       {len(data)}")
    print(f"X checks after prune:   {len(after_x)}")
    print(f"Z checks after prune:   {len(after_z)}")
