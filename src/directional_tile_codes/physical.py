from enum import Enum
from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

import stim
from qldpc import codes
from qldpc.circuits import MeasurementRecord, QubitIDs, SyndromeMeasurementStrategy
from qldpc.circuits.common import restrict_to_qubits
from qldpc.objects import Pauli

from .geometry import Vertex
from .directional_tile_code import DirectionalTileCode

class SiteType(Enum):
    DATA = 0
    ROUTING = 1
    X_CHECK = 2
    Z_CHECK = 3

CHECK_TYPES = (SiteType.X_CHECK, SiteType.Z_CHECK)

@dataclass(eq=False, frozen=True)
class Site: 
    # movable identity with a fixed role, not a physical location 
    type: SiteType


class Layout:
    def __init__(self, code: DirectionalTileCode):
        self.code = code
        self.word = code.word
        self.sites: dict[Vertex, Site] = {}

        data_edges, x_checks, z_checks = code.parity_checks

        self.data_positions = tuple(edge.middle for edge in data_edges)
        self.check_positions = (tuple(anchor.add(code.x_tile.check_qubit) for anchor in x_checks) + tuple(anchor.add(code.z_tile.check_qubit) for anchor in z_checks))

        for position in self.data_positions:
            self.add_site(position, SiteType.DATA)
        for index, position in enumerate(self.check_positions):
            type = SiteType.X_CHECK if index < len(x_checks) else SiteType.Z_CHECK
            self.add_site(position, type)
        self.checks = tuple(self.sites[pos] for pos in self.check_positions)

        self.add_routing_sites()
        self.routing_positions = tuple(sorted((pos for pos, site in self.sites.items() if site.type is SiteType.ROUTING), key=lambda pos: (pos.x, pos.y)))

    def add_site(self, pos: Vertex, type: SiteType):
        self.sites[pos] = Site(type)

    def get_pairs(self, sites: Mapping[Vertex, Site], direction):
        dx, dy = direction.dx // 2, direction.dy // 2
        pairs = {}

        for pos, site in sites.items():
            if site.type in CHECK_TYPES:
                pairs[pos] = Vertex(pos.x + dx, pos.y + dy)
            elif site.type is SiteType.DATA:
                pairs[Vertex(pos.x - dx, pos.y - dy)] = pos

        return list(pairs.items())
    
    def add_routing_sites(self):
        sites = self.sites.copy()

        for direction in self.word.word:
            pairs = self.get_pairs(sites, direction)

            for pair in pairs:
                for pos in pair:
                    if pos not in sites:
                        self.add_site(pos, SiteType.ROUTING)
                        sites[pos] = self.sites[pos]

            for first, second in pairs:
                sites[first], sites[second] = (sites[second], sites[first])

    def get_qubit_ids(self, shift: int = 0) -> QubitIDs:
        return QubitIDs.from_code(self.code.css_code, num_ancillas=len(self.routing_positions), shift=shift)

    def position_to_id(self, qubit_ids: QubitIDs) -> dict[Vertex, int]:
        return dict(
            zip(
                self.data_positions + self.check_positions + self.routing_positions,
                qubit_ids.data + qubit_ids.check + qubit_ids.ancilla,
                strict=True
            )
        )
    
    def syndrom_extraction(self, qubit_ids: QubitIDs, include_coordinates: bool = True) -> tuple[stim.Circuit, MeasurementRecord]:
        circuit = stim.Circuit()
        sites = self.sites.copy()
        physical_ids = self.position_to_id(qubit_ids) # dict[vertex] = id
        record: dict[object, list[int]] = {}

        if include_coordinates:
            for pos, qubit_id in physical_ids.items():
                circuit.append("QUBIT_COORDS", [qubit_id], [pos.x, pos.y])

        for pass_index, word in enumerate((self.word, self.word.inverse())):
            # Reset routing qubits
            for pos, site in sites.items():
                if site.type is SiteType.ROUTING:
                    circuit.append("R", [physical_ids[pos]])

            # Reset check qubits
            for pos, site in sites.items():
                if site.type is SiteType.X_CHECK:
                    circuit.append("RX", [physical_ids[pos]])
                elif site.type is SiteType.Z_CHECK:
                    circuit.append("R", [physical_ids[pos]])
            circuit.append("TICK")

            for direction in word.word:
                pairs = self.get_pairs(sites, direction)

                for first, second in pairs:
                    first_type, second_type = sites[first].type, sites[second].type
                    q1, q2 = physical_ids[first], physical_ids[second]

                    # Apply CXSWAP between X_CHECK-DATA and Z-CHECK-DATA
                    if second_type is SiteType.DATA and first_type is SiteType.X_CHECK:
                        circuit.append("CXSWAP", [q1, q2])
                    elif second_type is SiteType.DATA and first_type is SiteType.Z_CHECK:
                        circuit.append("CXSWAP", [q2, q1])
                    # Apply SWAP between ROUTING-DATA and ROUTING-CHECK
                    else:
                        circuit.append("SWAP", [q1, q2])

                # Update sites's types
                for first, second in pairs:
                    sites[first], sites[second] = (sites[second], sites[first])
                circuit.append("TICK")

            positions = {site: position for position, site in sites.items()}
            for check_id, check in zip(qubit_ids.check, self.checks, strict=True):
                measurement_index = circuit.num_measurements
                gate = "MX" if check.type is SiteType.X_CHECK else "M"
                circuit.append(gate, [physical_ids[positions[check]]])
                # Keys denote stabilizer identities, not the measured locations.
                key = ("forward", check_id) if pass_index == 0 else check_id
                record[key] = [measurement_index]
            circuit.append("TICK")

        measurement_record = MeasurementRecord(record)
        measurement_record.validate_num_measurements(circuit.num_measurements)
        for index, check_id in enumerate(qubit_ids.check):
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(("forward", check_id)), measurement_record.get_target_rec(check_id)], [0, 1, index]
            )
        return circuit, measurement_record


class DirectionalSyndromeMeasurement(SyndromeMeasurementStrategy):
    def __init__(self, code: DirectionalTileCode, *, include_coordinates: bool = False):
        self.layout = Layout(code)
        self.include_coordinates = include_coordinates

    def get_qubit_ids(self, shift: int = 0) -> QubitIDs:
        return self.layout.get_qubit_ids(shift=shift)

    @restrict_to_qubits
    def get_circuit(self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None) -> tuple[stim.Circuit, MeasurementRecord]:
        hx, hz = self.layout.code.parity_check_matrices
        # If this fails there is a mismatch of the measurement indices or qubit indices
        if not (np.array_equal(code.get_matrix(Pauli.X), hx) and np.array_equal(code.get_matrix(Pauli.Z), hz)):
            raise ValueError("CSS matrices do not match this layout's ordered checks and data")
        qubit_ids = QubitIDs.validated(self.get_qubit_ids() if qubit_ids is None else qubit_ids, code)
        return self.layout.syndrom_extraction(qubit_ids, include_coordinates=self.include_coordinates)
