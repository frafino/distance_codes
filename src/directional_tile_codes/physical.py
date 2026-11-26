from enum import Enum
from dataclasses import dataclass, replace
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
    type: SiteType


@dataclass
class CheckRoute:
    stabilizer_index: int
    sites: list[Site | None]
    positions: list[Vertex]
    active: list[bool]

    @property
    def window(self) -> tuple[int, int]:
        """Indices of the first and last data interaction in the route (end exclusive)."""
        hits = [i for i, site in enumerate(self.sites) if site is not None and site.type is SiteType.DATA]
        if not hits:
            raise ValueError("A stabilizer has empty data support")
        return hits[0], hits[-1] + 1


@dataclass(frozen=True)
class Gate:
    first: Vertex
    second: Vertex
    first_site: Site
    second_site: Site

    @property
    def kind(self) -> str:
        """Type of interaction: check_data, check_routing or data_routing."""
        if self.first_site.type in CHECK_TYPES:
            if self.second_site.type is SiteType.DATA:
                return "check_data"
            else:
                return "check_routing"
        return "data_routing"


class PhysicalLayout:
    def __init__(self, code: DirectionalTileCode, *, optimize: bool = False):
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
        self.check_sites = tuple(self.sites[pos] for pos in self.check_positions)


        self.data_sites = tuple(self.sites[pos] for pos in self.data_positions)
        w = self.word.weight
        routes = {i: CheckRoute(stabilizer_index=i, sites=[], positions=[pos], active=[True] * w) for i, pos in enumerate(self.check_positions)}
        trace = self.trace_routes(routes)
        self.reference = trace[3]

        # Control: if the referece is correct using the parity check matrices 
        hx, hz = code.parity_check_matrices
        expected = np.concatenate((np.asarray(hx), np.asarray(hz)))
        for signature in self.reference:
            for i, row in enumerate(expected):
                observed = [step[i] for step in signature if step[i] is not None]
                wanted = [site for site, bit in zip(self.data_sites, row) if bit]
                if len(observed) != len(wanted) or set(observed) != set(wanted):
                    raise ValueError("Baseline walk does not measure the parity-check matrix :(")
        
        self.apply(trace)
        self.optimization_log = []
        if optimize:
            self.shorten_windows()

    def add_site(self, pos: Vertex, type: SiteType):
        """Add a site of the given type at a position."""
        self.sites[pos] = Site(type)

    @staticmethod
    def get_pairs(sites: Mapping[Vertex, Site], direction, active_checks=None):
        """Return the interacting position pairs for one step in the given direction."""
        if active_checks is None: 
            active_checks = {s for s in sites.values() if s.type in CHECK_TYPES}
        
        dx, dy = direction.dx // 2, direction.dy // 2
        pairs = set()

        for pos, site in sites.items():
            if site in active_checks:
                pairs.add((pos, Vertex(pos.x + dx, pos.y + dy)))
            elif site.type is SiteType.DATA:
                pairs.add((Vertex(pos.x - dx, pos.y - dy), pos))

        return sorted(pairs, key=lambda pair: (pair[0].x, pair[0].y))

    def trace_routes(self, routes):
        """Simulate the forward and backward passes and return sites, routes, schedule and data signatures."""
        # Control: if check start overlaps with a data or check start
        initial = dict(zip(self.data_positions, self.data_sites, strict=True))
        for i, check in enumerate(self.check_sites):
            pos = routes[i].positions[0]
            if pos in initial:
                raise ValueError("Check start overlaps a data or check start :(")
            initial[pos] = check

        sites = initial.copy()
        check_index = {check: i for i, check in enumerate(self.check_sites)}
        traced = {i: replace(r, sites=[], positions=[r.positions[0]]) for i, r in routes.items()}
        layers, signatures = [], []
        for backward, word in enumerate((self.word, self.word.inverse())):
            layer, signature = [], []
            for step, direction in enumerate(word.word):
                k = self.word.weight - 1 - step if backward else step
                active = {self.check_sites[i] for i, r in routes.items() if r.active[k]} # checks that are active at step k
                pairs = self.get_pairs(sites, direction, active)
                gates = []
                used, partners = set(), [None] * len(self.check_sites)

                for first, second in pairs:
                    if first in used or second in used:
                        raise ValueError("Two gates use the same coordinate in one layer :(")
                    used.update((first, second))

                    # Add needed routing qubits 
                    for pos in (first, second):
                        if pos not in sites:
                            site = Site(SiteType.ROUTING)
                            sites[pos] = initial[pos] = site
                    a, b = sites[first], sites[second]

                    # Control: if a parked check is involved in an interaction
                    if b.type in CHECK_TYPES or (a.type in CHECK_TYPES and a not in active):
                        raise ValueError("A moving qubit overlaps a parked check")

                    gates.append(Gate(first, second, a, b))

                    if a in check_index:
                        partners[check_index[a]] = b

                # Save the interactions that need to remain untouched
                signature.append(tuple(site if site is not None and site.type is SiteType.DATA else None for site in partners))

                # Save the site the check is interacting with
                if not backward:
                    for i, route in traced.items():
                        route.sites.append(partners[i])

                # SWAP
                for first, second in pairs:
                    sites[first], sites[second] = sites[second], sites[first]
                positions = {site: pos for pos, site in sites.items()}
                
                # Save the position of the check
                if not backward:
                    for i, route in traced.items():
                        route.positions.append(positions[self.check_sites[i]])
                
                layer.append(gates)

            layers.append(layer)
            signatures.append(signature)
        
        if sites != initial:
            raise ValueError("Forward/backward motion does not restore all identities :(")
        
        if hasattr(self, "reference") and signatures != self.reference:
            raise ValueError("A data interaction identity or its time changed :(")
        
        return initial, traced, layers, signatures

    def apply(self, trace):
        """Store a traced layout as the current sites, routes and schedule."""
        self.sites, self.routes, self.schedule, _ = trace
        self.check_positions = tuple(self.routes[i].positions[0] for i in range(len(self.check_sites)))
        self.routing_positions = tuple(sorted((p for p, s in self.sites.items() if s.type is SiteType.ROUTING), key=lambda p: (p.x, p.y)))

    def proposal(self, i, part):
        """Return a route with the inactive prefix and/or suffix outside its data window removed."""
        r = self.routes[i]
        first, end = r.window
        active = r.active.copy()
        start = r.positions[0]

        if part in ("both", "prefix"):
            start = r.positions[first] 
            active[:first] = [False] * first

        if part in ("both", "suffix"):
            active[end:] = [False] * (len(active) - end)

        return replace(r, positions=[start], active=active)

    def shorten_windows(self, order="shortest"):
        """Greedily shorten check routes to reduce moves and routing qubits."""
        if order not in ("shortest", "index", "longest"):
            raise ValueError("order must be shortest, index, or longest")
        indices = list(self.routes)
        if order != "index":
            indices.sort(key=lambda i: (self.routes[i].window[1] - self.routes[i].window[0], i), reverse=order == "longest")

        original = self.trace_routes(self.routes)
        log_start = len(self.optimization_log)
        try:
            changed = True
            while changed:
                changed = False
                for i in indices:
                    for part in ("both", "prefix", "suffix"):
                        proposals, pending = {}, [i]
                        occupied = {r.positions[0]: j for j, r in self.routes.items()}
                        while pending:
                            j = pending.pop()
                            if j in proposals:
                                continue

                            # Propose a new route
                            r = self.proposal(j, part if j == i else "prefix")
                            proposals[j] = r

                            # Push the route of the check start blocking the proposal
                            blocker = occupied.get(r.positions[0])
                            if blocker is not None and blocker != j:
                                pending.append(blocker)

                        removed = sum(sum(self.routes[j].active) - sum(r.active) for j, r in proposals.items())
                        if not removed:
                            continue

                        candidate = self.routes | proposals
                        try:
                            trace = self.trace_routes(candidate)
                        except ValueError:
                            continue

                        before = len(self.routing_positions)
                        self.apply(trace)
                        self.optimization_log.append(dict(checks=tuple(proposals), 
                                                          part=part, 
                                                          steps_removed=removed,
                                                          routing_removed=before - len(self.routing_positions)))
                        changed = True

            # Control: if detector is deterministic
            circuit, record = self.syndrom_extraction()
            model = circuit.detector_error_model()
        except Exception:
            self.apply(original)
            del self.optimization_log[log_start:]
            raise
        return self

    def trajectories(self):
        """Return the path of every site over the full schedule."""
        sites = self.sites.copy()
        paths = {site: [pos] for pos, site in sites.items()}
        for layer in self.schedule[0] + self.schedule[1]:
            for gate in layer:
                p, q = gate.first, gate.second
                sites[p], sites[q] = sites[q], sites[p]
            for pos, site in sites.items():
                paths[site].append(pos)
        return paths

    def routing_counts(self):
        """Return counts of qubits, moves and gates in the layout."""
        circuit, _ = self.syndrom_extraction()
        gates = {name: sum(len(op.targets_copy()) // 2 for op in circuit if op.name == name) for name in ("SWAP", "CXSWAP")}
        return dict(data=len(self.data_positions), checks=len(self.check_positions),
                    routing=len(self.routing_positions), physical=len(self.sites),
                    check_moves_per_pass=sum(sum(r.active) for r in self.routes.values()),
                    check_data=sum(g.kind == "check_data" for p in self.schedule for layer in p for g in layer),
                    check_routing=sum(g.kind == "check_routing" for p in self.schedule for layer in p for g in layer),
                    data_routing=sum(g.kind == "data_routing" for p in self.schedule for layer in p for g in layer),
                    **gates)

    def get_qubit_ids(self, shift: int = 0) -> QubitIDs:
        """Return the qldpc QubitIDs for data, check and routing qubits."""
        return QubitIDs.from_code(self.code.css_code, num_ancillas=len(self.routing_positions), shift=shift)

    def position_to_id(self, qubit_ids: QubitIDs) -> dict[Vertex, int]:
        """Map each physical position to its qubit index."""
        return dict(
            zip(
                self.data_positions + self.check_positions + self.routing_positions,
                qubit_ids.data + qubit_ids.check + qubit_ids.ancilla,
                strict=True
            )
        )
    
    def syndrom_extraction(self, qubit_ids: QubitIDs | None = None) -> tuple[stim.Circuit, MeasurementRecord]:
        """Build the stim circuit for one forward and one backward pass with its measurement record."""
        circuit = stim.Circuit()
        sites = self.sites.copy()
        qubit_ids = self.get_qubit_ids() if qubit_ids is None else qubit_ids
        physical_ids = self.position_to_id(qubit_ids) # dict[vertex] = id
        record: dict[object, list[int]] = {}

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

            for layer in self.schedule[pass_index]:
                for gate in layer:
                    first, second = gate.first, gate.second
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
                for gate in layer:
                    first, second = gate.first, gate.second
                    sites[first], sites[second] = (sites[second], sites[first])
                circuit.append("TICK")

            positions = {site: position for position, site in sites.items()}
            for check_id, check in zip(qubit_ids.check, self.check_sites, strict=True):
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
    rounds_per_call = 2

    def __init__(self, code: DirectionalTileCode, *, optimize: bool = False):
        self.layout = PhysicalLayout(code, optimize=optimize)

    def get_qubit_ids(self, shift: int = 0) -> QubitIDs:
        """Return the qldpc QubitIDs for data, check and routing qubits."""
        return self.layout.get_qubit_ids(shift=shift)

    @restrict_to_qubits
    def get_circuit(self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None) -> tuple[stim.Circuit, MeasurementRecord]:
        """Return the syndrome extraction circuit and measurement record for the memory experiment."""
        # Possible mismatch of the measurement indices or qubit indices
        hx, hz = self.layout.code.parity_check_matrices
        if not (np.array_equal(code.get_matrix(Pauli.X), hx) and np.array_equal(code.get_matrix(Pauli.Z), hz)):
            raise ValueError("CSS matrices do not match this layout's ordered checks and data")

        qubit_ids = QubitIDs.validated(self.get_qubit_ids() if qubit_ids is None else qubit_ids, code)
        return self.layout.syndrom_extraction(qubit_ids)
