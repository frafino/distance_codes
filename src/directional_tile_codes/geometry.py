from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class Vertex:
    x: int
    y: int

    def move(self, direction: "Direction") -> "Vertex":
        """Return the vertex reached by one step in the given direction."""
        return Vertex(self.x + direction.dx, self.y + direction.dy)

    def add(self, vertex: "Vertex") -> "Vertex":
        """Return the coordinate-wise sum of two vertices."""
        return Vertex(self.x + vertex.x, self.y + vertex.y)


class Direction(Enum):
    N = (0, 2)
    E = (2, 0)
    S = (0, -2)
    W = (-2, 0)

    @property
    def dx(self) -> int:
        """Horizontal displacement of the direction."""
        return self.value[0]

    @property
    def dy(self) -> int:
        """Vertical displacement of the direction."""
        return self.value[1]

    @property
    def opposite(self) -> "Direction":
        """Return the opposite direction."""
        return {
            Direction.N: Direction.S,
            Direction.S: Direction.N,
            Direction.E: Direction.W,
            Direction.W: Direction.E,
        }[self]


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

    @property
    def opposite(self) -> "Orientation":
        """Return the other orientation."""
        return Orientation(1 - self.value)


@dataclass(frozen=True)
class Edge:
    origin: Vertex  # bottom/left adjacent vertex if the orientation is vertical/horizontal
    orientation: Orientation

    @classmethod
    def between(cls, vertex_1: Vertex, vertex_2: Vertex) -> "Edge":
        """Return the edge connecting two adjacent vertices."""
        dy = vertex_2.y - vertex_1.y
        
        if dy == 0:
            origin = Vertex(min(vertex_1.x, vertex_2.x), vertex_1.y)
            return cls(origin, Orientation.HORIZONTAL)
        else:
            origin = Vertex(vertex_1.x, min(vertex_1.y, vertex_2.y))
            return cls(origin, Orientation.VERTICAL)

    @property
    def middle(self) -> Vertex:
        """Midpoint of the edge, where the data qubit sits."""
        if self.orientation is Orientation.HORIZONTAL:
            return Vertex(self.origin.x + 1, self.origin.y)
        else:
            return Vertex(self.origin.x, self.origin.y + 1)

    def translated(self, offset: Vertex) -> "Edge":
        """Return the edge shifted by the given offset."""
        return Edge(self.origin.add(offset), self.orientation)


    def perpendicular(self) -> "Edge":
        """Return the perpendicular edge crossing this one at its midpoint."""
        if self.orientation is Orientation.HORIZONTAL:
            origin = self.origin.add(Vertex(1, -1))
            orientation = Orientation.VERTICAL
        else:
            origin = self.origin.add(Vertex(-1, 1)) 
            orientation = Orientation.HORIZONTAL
        return Edge(origin, orientation)
