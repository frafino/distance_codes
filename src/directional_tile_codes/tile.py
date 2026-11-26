from enum import Enum
from collections import Counter
from functools import cached_property

from .directional_word import DirectionalWord
from .geometry import Edge, Vertex

# TODO: Check if it makes sense to have get_x_edges like in DirectionalWord.
# TODO: Do I need this class?

class TileType(Enum):
    X = 0
    Z = 1


class Tile():
    def __init__(self, type: TileType, word: DirectionalWord):
        self.type = type
        self.B = word.B
        self.word = word

    def get_x_edges(self) -> tuple[tuple[Edge, ...], Vertex]:
        """Return the X tile edges and its check vertex."""
        return self.word.get_x_walk()

    def mutual_z_edges(self) -> tuple[Edge, ...]:
        """Return the Z support required by mutuality (X support rotated by 180 degrees)."""
        c = 2 * (self.B - 1)
        x_edges, _ = self.get_x_edges()
        return tuple(Edge(Vertex(c - e.origin.x, c - e.origin.y), e.orientation.opposite) for e in x_edges)
    
    def get_edges_check(self) -> tuple[tuple[Edge, ...], Vertex]:
        """Return the edges and check vertex for this tile type."""
        if self.type is TileType.X:
            return self.get_x_edges()
        else:
            return self.get_z_edges()

    def get_z_edges(self) -> tuple[tuple[Edge, ...], Vertex]:
        """Return the Z tile edges and check vertex, aligned with the mutual Z support."""
        dual, start = self.word.get_z_walk()
        target = self.mutual_z_edges()

        offset = Vertex(
            min(e.origin.x for e in target) - min(e.origin.x for e in dual),
            min(e.origin.y for e in target) - min(e.origin.y for e in dual),
        )

        return (tuple(edge.translated(offset) for edge in dual), start.add(offset))

    @cached_property
    def edges(self) -> tuple[Edge, ...]:
        """Edges in the support of the tile."""
        edges, _ = self.get_edges_check()
        return edges
    
    @cached_property
    def check_qubit(self) -> Vertex:
        """Position of the check qubit relative to the tile anchor."""
        _, check = self.get_edges_check()
        return check
    
    def fits_box(self) -> bool:
        """Check that all tile edges lie inside the B x B box."""
        c = 2 * (self.B - 1)
        return all(0 <= e.origin.x <= c and 0 <= e.origin.y <= c
                   and e.origin.x % 2 == 0 and e.origin.y % 2 == 0
                   for e in self.edges)
    
    @staticmethod
    def check_mutuality(x_tile: "Tile", z_tile: "Tile") -> bool:
        """Check that the X and Z tiles satisfy the mutuality condition."""
        return (x_tile.B == z_tile.B and x_tile.fits_box() and z_tile.fits_box()
                and set(x_tile.mutual_z_edges()) == set(z_tile.edges))

    @staticmethod
    def check_parity(x_tile: "Tile") -> bool:
        """Check the ordered parity condition on the X string."""
        if x_tile.type is not TileType.X:
            raise ValueError("The parity test uses the ordered X string.")
        middles = [e.middle for e in x_tile.edges]
        counts = Counter()
        for i, start in enumerate(middles):
            for end in middles[i + 1:]:
                dx, dy = end.x - start.x, end.y - start.y
                if dy % 2:
                    counts[(dx, dy)] += 1
        return all(count % 2 == 0 for count in counts.values())

    @staticmethod
    def check(x_tile: "Tile", z_tile: "Tile") -> bool:
        """Check both mutuality and ordered parity conditions."""
        return Tile.check_mutuality(x_tile, z_tile) and Tile.check_parity(x_tile)
