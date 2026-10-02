from enum import Enum
from collections import Counter

from .directional_word import DirectionalWord
from .geometry import Edge, Vertex


class TileType(Enum):
    X = 0
    Z = 1


class Tile():
    def __init__(self, type: TileType, word: DirectionalWord):
        self.type = type
        self.B = word.B
        self.word = word

    def get_x_edges(self) -> tuple[Edge, ...]:
        return self.word.get_x_edges()

    def mutual_z_edges(self) -> tuple[Edge, ...]:
        c = 2 * (self.B - 1)
        return tuple(Edge(Vertex(c - e.origin.x, c - e.origin.y),
                          e.orientation.opposite) for e in self.get_x_edges())

    def get_z_edges(self) -> tuple[Edge, ...]:
        dual = self.word.get_z_edges()
        target = self.mutual_z_edges()
        offset = Vertex(min(e.origin.x for e in target) - min(e.origin.x for e in dual),
                        min(e.origin.y for e in target) - min(e.origin.y for e in dual))
        return tuple(e.translated(offset) for e in dual)
    
    def get_edges(self) -> tuple[Edge, ...]:
        if self.type is TileType.X:
            return self.get_x_edges()
        else:
            return self.get_z_edges()
    
    def fits_box(self) -> bool:
        c = 2 * (self.B - 1)
        return all(0 <= e.origin.x <= c and 0 <= e.origin.y <= c
                   and e.origin.x % 2 == 0 and e.origin.y % 2 == 0
                   for e in self.get_edges())
    
    @staticmethod
    def check_mutuality(x_tile: "Tile", z_tile: "Tile") -> bool:
        return (x_tile.B == z_tile.B and x_tile.fits_box() and z_tile.fits_box()
                and set(x_tile.mutual_z_edges()) == set(z_tile.get_z_edges()))

    @staticmethod
    def check_parity(x_tile: "Tile") -> bool:
        if x_tile.type is not TileType.X:
            raise ValueError("The parity test uses the ordered X string.")
        middles = [e.middle for e in x_tile.get_edges()]
        counts = Counter()
        for i, start in enumerate(middles):
            for end in middles[i + 1:]:
                dx, dy = end.x - start.x, end.y - start.y
                if dy % 2:
                    counts[(dx, dy)] += 1
        return all(count % 2 == 0 for count in counts.values())

    @staticmethod
    def check(x_tile: "Tile", z_tile: "Tile") -> bool:
        return Tile.check_mutuality(x_tile, z_tile) and Tile.check_parity(x_tile)
