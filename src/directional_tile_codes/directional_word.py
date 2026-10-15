from .geometry import Direction, Edge, Vertex

# TODO: Check if it makes sense to have get_x_edges like in Tile.

class DirectionalWord:
    def __init__(self, word):
        if isinstance(word, str):
            try:
                word = tuple(Direction[c] for c in word.upper())
            except KeyError:
                raise ValueError(":( Use only N, E, S, W.")

        self.word = tuple(word)

        edges = self.get_x_edges()
        if len(set(edges)) != len(edges):
            raise ValueError(":( Repeated edges.")

    @property
    def weight(self) -> int:
        return len(self.word)

    def inverse(self) -> "DirectionalWord":
        inverse_word = tuple(direction.opposite for direction in reversed(self.word))
        return DirectionalWord(inverse_word)

    def get_x_walk(self) -> tuple[tuple[Edge, ...], Vertex]:
        start = Vertex(0, 0)
        current_vertex = start
        edges = []

        for direction in self.word:
            next_vertex = current_vertex.move(direction)
            edges.append(Edge.between(current_vertex, next_vertex))
            current_vertex = next_vertex

        offset = Vertex(
            -min(edge.origin.x for edge in edges),
            -min(edge.origin.y for edge in edges),
        )

        return (
            tuple(edge.translated(offset) for edge in edges),
            start.add(offset),
        )

    def get_z_walk(self) -> tuple[tuple[Edge, ...], Vertex]:
        start = Vertex(1, -1)
        current_vertex = start
        edges = []

        for direction in self.word:
            next_vertex = current_vertex.move(direction)
            dual_edge = Edge.between(current_vertex, next_vertex)
            edges.append(dual_edge.perpendicular())
            current_vertex = next_vertex

        offset = Vertex(
            -min(edge.origin.x for edge in edges),
            -min(edge.origin.y for edge in edges),
        )

        return (
            tuple(edge.translated(offset) for edge in edges),
            start.add(offset),
        )


    def get_x_edges(self) -> tuple[Edge, ...]:
        edges, _ = self.get_x_walk()
        return edges


    def get_z_edges(self) -> tuple[Edge, ...]:
        edges, _ = self.get_z_walk()
        return edges
    
    @property
    def B(self):
        edges = self.get_x_edges() + self.get_z_edges()
        max_hor = max(edge.origin.x for edge in edges)
        max_ver = max(edge.origin.y for edge in edges)

        return max(max_hor, max_ver) // 2 + 1
    