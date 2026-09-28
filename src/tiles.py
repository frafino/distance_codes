from .directional_word import DirectionalWord
from .geometry import Edge, Vertex
from enum import Enum


class TileType(Enum):
    X = 0
    Z = 1


class Tile():
    def __init__(self, type : TileType, word : DirectionalWord):
        self.type = type
        self.B = word.B
        
        if self.type is TileType.X:
            raw = word.get_x_edges()
            self.path_start = Vertex(-min(e.origin.x for e in raw), -min(e.origin.y for e in raw))
        else:
            raw = word.get_z_edges()
            self.path_start = Vertex(self.B - 1 - max(e.origin.x for e in raw) + 0.5, self.B - 1 - max(e.origin.y for e in raw) + 0.5)

        self.support = tuple(edge.translated(self.offset) for edge in raw)


class TilePair():
    def __init__(self, x_tile : Tile, z_tile : Tile):
        self.x_tile = x_tile
        self.z_tile = z_tile

    def check_mutuality(self) -> bool:
        B = self.x_tile.B
        if B != self.z_tile.B:
            return False
        
        corresponding_edges = set()

        for x_edge in self.x_tile.support:
            # X tile has a vertical/horizontal edge in (a,b) --> Z tile has a horizontal/vertical edge in (B-1-a, B-1-b)
            a_prime = B - 1 - x_edge.origin.x
            b_prime = B - 1 - x_edge.origin.y
            orientation_prime = x_edge.orientation.opposite

            corresponding_edge = Edge(Vertex(a_prime, b_prime), orientation_prime)
            corresponding_edges.add(corresponding_edge)
        
        return corresponding_edges == set(self.z_tile.support)

    
    def check_parity(self) -> bool:
        middles = [edge.middle for edge in self.x_tile.support]
        counts = {}

        for i in range(len(middles)):
            for j in range(i + 1, len(middles)):
                dx = int(2 * (middles[j].x - middles[i].x))
                dy = int(2 * (middles[j].y - middles[i].y))

                if dy % 2:
                    vector = (dx, dy)
                    counts[vector] = counts.get(vector, 0) + 1

        return all(count % 2 == 0 for count in counts.values())
        
    def check(self) -> bool:
        return self.check_mutuality() and self.check_parity()