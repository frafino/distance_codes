from .geometry import Direction, Edge, Vertex

class DirectionalWord:
    def __init__(self, word):
        if isinstance(word, str):
            try:
                word = tuple(Direction[c] for c in word.upper())
            except KeyError as exc:
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

    def get_x_edges(self) -> tuple[Edge, ...]:
        x_edges = []
        current_vertex = Vertex(0, 0)

        for direction in self.word:
            next_vertex = current_vertex.move(direction)
            edge = Edge.between(current_vertex, next_vertex)
            x_edges.append(edge)
            current_vertex = next_vertex

        offset = Vertex(-min(edge.origin.x for edge in x_edges), -min(edge.origin.y for edge in x_edges))

        return tuple([edge.translated(offset) for edge in x_edges])

    def get_z_edges(self) -> tuple[Edge, ...]:
        z_edges = []
        current_vertex = Vertex(1, -1)
        
        for direction in self.word:
            next_vertex = current_vertex.move(direction)
            edge = Edge.between(current_vertex, next_vertex)
            edge = Edge.perpendicular(edge)
            z_edges.append(edge)
            current_vertex = next_vertex

        offset = Vertex(-min(edge.origin.x for edge in z_edges), -min(edge.origin.y for edge in z_edges))

        return tuple([edge.translated(offset) for edge in z_edges])
    
    @property
    def B(self):
        # I am returning it as the paper convention, not in the full lattice
        edges = self.get_x_edges() + self.get_z_edges()
        max_hor = max(edge.origin.x for edge in edges)
        max_ver = max(edge.origin.y for edge in edges)

        return max(max_hor, max_ver) // 2 + 1
    