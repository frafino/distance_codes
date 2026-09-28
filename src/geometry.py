from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class Vertex:
    x: int
    y: int

    def move(self, direction: "Direction") -> "Vertex":
        return Vertex(self.x + direction.dx, self.y + direction.dy)

    def add(self, vertex: "Vertex") -> "Vertex":
        return Vertex(self.x + vertex.x, self.y + vertex.y)

    @property
    def primal(self) -> "Vertex":
        return Vertex(2 * self.x, 2 * self.y)

    @property
    def dual(self) -> "Vertex":
        return self.primal.add(Vertex(1, -1))

class Direction(Enum):
    N = (0, 2)
    E = (2, 0)
    S = (0, -2)
    W = (-2, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]

    @property
    def opposite(self) -> "Direction":
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
        return Orientation(1 - self.value)


@dataclass(frozen=True)
class Edge:
    origin: Vertex  # bottom/left adjacent vertex if the orientation is vertical/horizontal
    orientation: Orientation

    @classmethod
    def between(cls, vertex_1: Vertex, vertex_2: Vertex) -> "Edge":
        dx = vertex_2.x - vertex_1.x
        dy = vertex_2.y - vertex_1.y

        if abs(dx) + abs(dy) != 2: 
            raise ValueError("The two vertices are not neighbours.")
        
        if dy == 0:
            origin = Vertex(min(vertex_1.x, vertex_2.x), vertex_1.y)
            return cls(origin, Orientation.HORIZONTAL)
        else:
            origin = Vertex(vertex_1.x, min(vertex_1.y, vertex_2.y))
            return cls(origin, Orientation.VERTICAL)

    @property
    def middle(self) -> Vertex:
        if self.orientation is Orientation.HORIZONTAL:
            return Vertex(self.origin.x + 1, self.origin.y)
        else:
            return Vertex(self.origin.x, self.origin.y + 1)

    def translated(self, offset: Vertex) -> "Edge":
        return Edge(self.origin.add(offset), self.orientation)