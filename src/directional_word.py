from .geometry import *

class DirectionalWord:
    def __init__(self, sequence):
        self.sequence = tuple(sequence)

    @property
    def weight(self) -> int:
        return len(self.sequence)

    def inverse(self) -> "DirectionalWord":
        inverse_sequence = tuple(direction.opposite for direction in reversed(self.sequence))
        return DirectionalWord(inverse_sequence)

    def get_x_edges(self, start: Vertex = Vertex(0, 0)) -> tuple[Edge, ...]:
        x_edges = []
        current_vertex = start

        for direction in self.sequence:
            next_vertex = current_vertex.move(direction)

            edge = Edge.between(current_vertex, next_vertex)
            x_edges.append(edge)

            current_vertex = next_vertex

        return tuple(x_edges)

    def get_z_edges(self, start: Vertex = Vertex(0, 0)) -> tuple[Edge, ...]:
        z_edges = []
        for edge in self.get_x_edges(start):
            a, b = edge.origin.x, edge.origin.y

            if edge.orientation is Orientation.HORIZONTAL:
                z_edges.append(Edge(Vertex(a, b), Orientation.VERTICAL))
            else:
                z_edges.append(Edge(Vertex(a + 1, b - 1), Orientation.HORIZONTAL))

        return tuple(z_edges)

    @property
    def B(self):
        edges = self.get_x_edges()
        max_hor = 0
        max_ver = 0

        for edge in edges:
            if edge.orientation is Orientation.HORIZONTAL:
                max_hor = max(max_hor, edge.origin.y)
            else:
                max_ver = max(max_ver, edge.origin.x)

        return max(max_hor, max_ver)
    
