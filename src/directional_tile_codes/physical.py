from enum import Enum

import stim

from .geometry import Vertex
from .directional_tile_code import DirTileCode

# TODO: locations is confusing -> sites

class SiteType(Enum):
    DATA = 0
    ROUTING = 1
    X_CHECK = 2
    Z_CHECK = 3


class Site:
    def __init__(self, type: SiteType):
        self.type = type


class Layout:
    def __init__(self, code: DirTileCode):
        self.code = code
        self.word = code.word
        self.locations: dict[Vertex, Site] = {}

        data_edges, _, _ = code.parity_checks

        for edge in data_edges:
            self.add_site(edge.middle, SiteType.DATA)

        for vertex in sorted(code.x_check_qubits, key=lambda v: (v.x, v.y)):
            self.add_site(vertex, SiteType.X_CHECK)

        for vertex in sorted(code.z_check_qubits, key=lambda v: (v.x, v.y)):
            self.add_site(vertex, SiteType.Z_CHECK)

        self.checks = [
            site for site in self.locations.values()
            if site.type in (SiteType.X_CHECK, SiteType.Z_CHECK)
        ]

        self.add_routing_sites()
        self.qubits = {loc: i for i, loc in enumerate(self.locations)}

    def add_site(self, vertex: Vertex, type: SiteType):
        self.locations[vertex] = Site(type)

    def get_pairs(self, locations, direction):
        dx, dy = direction.dx // 2, direction.dy // 2
        pairs = {}

        for loc, site in locations.items():
            if site.type in (SiteType.X_CHECK, SiteType.Z_CHECK):
                pairs[loc] = Vertex(loc.x + dx, loc.y + dy)
            elif site.type is SiteType.DATA:
                pairs[Vertex(loc.x - dx, loc.y - dy)] = loc

        return list(pairs.items())

    def add_routing_sites(self):
        locations = self.locations.copy()

        for direction in self.word.word:
            pairs = self.get_pairs(locations, direction)

            for first, second in pairs:
                for loc in (first, second):
                    if loc not in locations:
                        self.add_site(loc, SiteType.ROUTING)
                        locations[loc] = self.locations[loc]

            for first, second in pairs:
                locations[first], locations[second] = (locations[second], locations[first])

    def syndrom_extraction(self, rounds: int = 1) -> stim.Circuit:
        circuit = stim.Circuit()
        locations = self.locations.copy()

        for loc, qubit in self.qubits.items():
            circuit.append("QUBIT_COORDS", [qubit], [loc.x, loc.y])

        # Reset routing qubits
        for loc, site in locations.items():
            if site.type is SiteType.ROUTING:
                circuit.append("R", [self.qubits[loc]])

        for round_index in range(rounds):
            # Reset check qubits
            for loc, site in locations.items():
                if site.type is SiteType.X_CHECK:
                    circuit.append("RX", [self.qubits[loc]])
                elif site.type is SiteType.Z_CHECK:
                    circuit.append("R", [self.qubits[loc]])
            circuit.append("TICK")

            word = self.word if round_index % 2 == 0 else self.word.inverse()

            for direction in word.word:
                operations = self.get_pairs(locations, direction)

                for first, second in operations:
                    q1, q2 = self.qubits[first], self.qubits[second]
                    first_type = locations[first].type
                    second_type = locations[second].type

                    # Apply CXSWAP between X_CHECK-DATA and Z-CHECK-DATA
                    if second_type is SiteType.DATA and first_type is SiteType.X_CHECK:
                        circuit.append("CXSWAP", [q1, q2])
                    elif second_type is SiteType.DATA and first_type is SiteType.Z_CHECK:
                        circuit.append("CXSWAP", [q2, q1])
                    # Apply SWAP between ROUTING-DATA and ROUTING-CHECK
                    else:
                        circuit.append("SWAP", [q1, q2])

                # Update locations's types
                for first, second in operations:
                    locations[first], locations[second] = (locations[second], locations[first])
                circuit.append("TICK")

            # Measure X_CHECK and Z_CHECK
            positions = {site: loc for loc, site in locations.items()}
            for site in self.checks:
                gate = "MX" if site.type is SiteType.X_CHECK else "M"
                circuit.append(gate, [self.qubits[positions[site]]])

            # Add detectors
            num_checks = len(self.checks)
            if round_index > 0:
                for j in range(num_checks):
                    circuit.append(
                        "DETECTOR",
                        [
                            stim.target_rec(j - num_checks),
                            stim.target_rec(j - 2 * num_checks),
                        ],
                    )

            circuit.append("TICK")

        self.final_locations = locations
        return circuit
