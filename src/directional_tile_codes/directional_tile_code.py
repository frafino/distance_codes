import numpy as np
import galois
from qldpc.codes import CSSCode

from .directional_word import DirectionalWord
from .tile import Tile, TileType
from .geometry import Edge, Vertex, Orientation


class DirTileCode:
    def __init__(self, M: int, N: int, word: DirectionalWord):
        self.M = M
        self.N = N
        self.word = word
        self.x_tile = Tile(TileType.X, word)
        self.z_tile = Tile(TileType.Z, word)
        if not Tile.check(x_tile=self.x_tile, z_tile=self.z_tile):
            raise ValueError(":( Word fails the directional tile mutuality or ordered parity condition.")

        self.B = word.B
        self.padding = self.B - 1
        self.anchor_shape = (self.M + 2 * self.padding, self.N + 2 * self.padding)

    def empty_anchor_mask(self):
        return np.zeros(self.anchor_shape, dtype=bool)

    @property
    def bulk_mask(self):
        p = self.padding
        mask = self.empty_anchor_mask()
        mask[p:p + self.M, p:p + self.N] = True

        return mask

    @property
    def x_boundary_mask(self):
        p = self.padding
        mask = self.empty_anchor_mask()
        mask[p:p + self.M, :p] = True
        mask[p:p + self.M, p + self.N:] = True

        return mask

    @property
    def z_boundary_mask(self):
        p = self.padding
        mask = self.empty_anchor_mask()
        mask[:p, p:p + self.N] = True
        mask[p + self.M:, p:p + self.N] = True

        return mask

    @property
    def x_anchor_mask(self):
        return self.bulk_mask | self.x_boundary_mask

    @property
    def z_anchor_mask(self):
        return self.bulk_mask | self.z_boundary_mask

    @property
    def layout(self):
        width = self.M + self.B - 1
        height = self.N + self.B - 1
        return {Edge(Vertex(2 * x, 2 * y), orientation) for x in range(width) for y in range(height) for orientation in Orientation}

    def tessellate(self, tile, mask, layout):
        checks = {}
        for i, j in np.argwhere(mask):
            anchor = Vertex(2 * (int(i) - self.padding), 2 * (int(j) - self.padding))
            all_edges = tuple(edge.translated(anchor) for edge in tile.get_edges())
            surviving_edges = set(edge.translated(anchor) for edge in tile.get_edges()) & layout
            checks[anchor] = tuple([edge for edge in all_edges if edge in surviving_edges])
        return checks

    def prune(self, checks, data):
        return {
            anchor: tuple(edge for edge in support if edge in data)
            for anchor, support in checks.items()
            if any(edge in data for edge in support)
        }

    def build_checks(self):
        x_checks = self.tessellate(self.x_tile, self.x_anchor_mask, self.layout)
        z_checks = self.tessellate(self.z_tile, self.z_anchor_mask, self.layout)
        
        data = set().union(*x_checks.values()) & set().union(*z_checks.values())

        ordered_data = tuple(sorted(data, key=lambda e: (e.origin.x, e.origin.y, e.orientation.value)))
        return ordered_data, self.prune(x_checks, data), self.prune(z_checks, data)

    def parity_check_matrices(self):
        data, x_checks, z_checks = self.build_checks()
        indices = {edge: i for i, edge in enumerate(data)}

        def matrix(checks):
            result = galois.GF(2).Zeros((len(checks), len(data)))

            for row, support in enumerate(checks.values()):
                for edge in support:
                    result[row, indices[edge]] = 1

            return result

        h_x, h_z = matrix(x_checks), matrix(z_checks)

        if np.any(h_x @ h_z.T):
            raise ValueError("Constructed patch has anticommuting X/Z checks.")

        return h_x, h_z

    @property
    def logical_ops(self):
        data, _, _ = self.build_checks()
        css_code = self.to_css_code()
        logicals = np.asarray(css_code.get_logical_ops())
        n = len(data)

        if logicals.ndim == 1:
            logicals = logicals[np.newaxis, :]

        if logicals.shape[1] == 2 * n:
            supports = (logicals[:, :n] != 0) | (logicals[:, n:] != 0)
        elif logicals.shape[1] == n:
            supports = logicals != 0
        else:
            raise ValueError(
                f"Expected {n} or {2 * n} entries per logical operator, got {logicals.shape[1]}."
            )

        return tuple(
            tuple(data[i] for i in np.flatnonzero(support))
            for support in supports
        )

    @property
    def num_check_qubits(self):
        data, _, _ = self.build_checks()
        return len(data)

    @property
    def num_logical_qubits(self):
        css_code = self.to_css_code()
        rank = css_code.rank
        return self.num_check_qubits - rank

    def to_css_code(self):
        h_x, h_z = self.parity_check_matrices()
        return CSSCode(code_x=h_x, code_z=h_z)

    @property
    def distance_exact(self):
        css_code = self.to_css_code()
        return css_code.get_distance_exact()

    @property
    def distance_x(self):
        css_code = self.to_css_code()
        return css_code.get_distance_exact("X")

    @property
    def distance_z(self):
        css_code = self.to_css_code()
        return css_code.get_distance_exact("Z")
    