from .geometry import Vertex
from .directional_tile_code import DirTileCode

from enum import Enum 

class LocationType(Enum):
    DATA = 0
    ROUTING = 1
    X_CHECK = 2
    Z_CHECK = 3
    OUTSIDE = 4

class Location():
    def __init__(self, x : int, y : int, type : LocationType):
        self.vertex = Vertex(x, y)
        self.type = type

class Layout():
    def __init__(self, code : DirTileCode):
        self.data, _, _ = code.parity_checks
        self.code = code
        self.space = code.empty_mask()
        self.x_checks = code.x_check_qubits
        self.z_checks = code.z_check_qubits

        