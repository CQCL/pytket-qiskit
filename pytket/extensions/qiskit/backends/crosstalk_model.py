# Copyright 2019-2023 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Tuple, Union, Dict, Optional
from dataclasses import dataclass

from qiskit.providers.aer.noise import NoiseModel  # type: ignore
from qiskit_aer.noise.errors.standard_errors import amplitude_damping_error, phase_damping_error  # type: ignore

import numpy as np
from scipy.linalg import fractional_matrix_power  # type: ignore

from pytket.circuit import (
    Circuit,
    Qubit,
    Node,
    Command,
    OpType,
    Op,
    Unitary1qBox,
    Unitary2qBox,
    Unitary3qBox,
)
from pytket.backends.backendinfo import BackendInfo
from pytket.extensions.qiskit.qiskit_convert import _gate_str_2_optype


@dataclass
class FractionalUnitary:
    """
    Wrapper for a fractional unitary gate
    :param cmd: the fractional UnitaryBox wrapped in a pytket Command
    :param n_fractions: the number of fractional gates
        used to compose the original unitary gate.
    """

    cmd: Command
    n_fractions: float


@dataclass
class NoiseGate:
    """
    Wrapper for a gate that simulates noise
    :param cmd: gate wrapped in a pytket Command
    :param type: one of zz_crosstalks, single_q_phase, two_q_induced_phase
        and non_markovian.
    """

    cmd: Command
    type: str


Instruction = Union[FractionalUnitary, Command, NoiseGate]
Slice = List[Instruction]
EPS = 1e-9


@dataclass
class CrosstalkParams:
    """
    Stores various parameters for modelling crosstalk noise

    :param zz_crosstalks: symmetric crosstalks between qubit pairs
    :type zz_crosstalks: `Dict[Tuple[Qubit, Qubit], float]`
    :param single_q_phase_errors: dict specify the single qubit phase error
        on each qubit
    :type single_q_phase_errors: `Dict[Qubit, float]`
    :param two_q_induced_phase_errors: keys of dictionary specify the control
        and target qubit index, while the values are tuples with the spectator
        qubit index and the amount of phase error to be applied.
    :type two_q_induced_phase_errors: `Dict[Tuple[Qubit, Qubit], Tuple[Qubit, float]]`
    :param non_markovian_noise: List storing the non-Markovian noise parameters.
        Each tuple in the list contains the qubit index and the zx, zz noise parameters.
    :type non_markovian_noise: `List[Tuple[Qubit, float, float]]`
    :param virtual_z: If True, then don't break any single qubit Z gate into
        unitary fractions, instead add the full unitary.
    :type bool
    :param N: hyperparameter specifies splices per second.
        1/N must divide all gate times.
    :type: float
    :param gate_times: python dict to store the gate time information.
    :type gate_times: `Dict[Tuple[OpType, Tuple[Qubit, ...]], float]`
    :param phase_damping_error: dict specify amplitude phase damping error
        on each qubit
    :type phase_damping_error: `Dict[Qubit, float]`
    :param amplitude_damping_error: dict pecify amplitude damping error
        on each qubit
    """

    zz_crosstalks: Dict[Tuple[Qubit, Qubit], float]
    single_q_phase_errors: Dict[Qubit, float]
    two_q_induced_phase_errors: Dict[Tuple[Qubit, Qubit], Tuple[Qubit, float]]
    non_markovian_noise: List[Tuple[Qubit, float, float]]
    virtual_z: bool
    N: float
    gate_times: Dict[Tuple[OpType, Tuple[Qubit, ...]], float]
    phase_damping_error: Dict[Qubit, float]
    amplitude_damping_error: Dict[Qubit, float]

    def get_noise_model(self) -> NoiseModel:
        """Construct a NoiseModel from phase_damping_error
        and amplitude_damping_error"""
        noise_model = NoiseModel()
        for q, phase in self.phase_damping_error.items():
            noise_model.add_quantum_error(
                phase_damping_error(phase / self.N), ["unitary"], [q.index[0]]
            )
        for q, amp in self.amplitude_damping_error.items():
            noise_model.add_quantum_error(
                amplitude_damping_error(amp / self.N),
                ["unitary"],
                [q.index[0]],
                warnings=False,
            )
        return noise_model


class NoisyCircuitBuilder:
    """Builder used to generate a noisy circuit"""

    Ibox = Unitary1qBox(np.eye(2))  # type: ignore
    SUPPORTED_TYPES = {
        OpType.Unitary1qBox,
        OpType.Unitary2qBox,
        OpType.Unitary3qBox,
        OpType.Measure,
        OpType.Reset,
    }

    def __init__(
        self,
        circ: Circuit,
        ct_params: CrosstalkParams,
    ) -> None:
        """Construct a builder to generate noisy circuit
        :param circ: the original circuit.
        :type circ: `Circuit`
        :param N: hyperparameter N
        :type N: float
        :param ct_params: crosstalk parameters.
        :type ct_params: `CrosstalkParams`
        """
        self.circ = circ
        self.all_qubits = set(circ.qubits)
        self.N = ct_params.N
        self.ct_params = ct_params
        self.two_level_map = {}
        for i, (q, _, _) in enumerate(self.ct_params.non_markovian_noise):
            two_level_q = Qubit("two-level", i)
            self.two_level_map[q] = two_level_q
        self.reset()

    def reset(self) -> None:
        """Clear the build cache"""
        self._slices: List[Slice] = []

    @staticmethod
    def _get_qubits(inst: Instruction) -> List[Qubit]:
        if isinstance(inst, Command):
            return inst.qubits
        else:
            return inst.cmd.qubits

    def _append(
        self,
        inst: Instruction,
        frontier: Dict[Qubit, int],
    ) -> None:
        """Append a command to the splices in Tetris style"""
        args = self._get_qubits(inst)
        pivot_q = max(args, key=lambda q: frontier[q])
        slice_idx = frontier[pivot_q]
        # add cmd to the idx_th slice
        n_slices = len(self._slices)
        assert slice_idx <= n_slices
        if slice_idx == n_slices:
            self._slices.append([inst])
        else:
            self._slices[slice_idx].append(inst)
        # update frontier
        for q in args:
            frontier[q] = slice_idx + 1

    def _fill_gaps(self, frontier: Dict[Qubit, int]) -> None:
        """Fill the gaps in the slices with identity `Unitary1qBox`es"""
        for idx, s in enumerate(self._slices):
            slice_qubits = set().union(*[self._get_qubits(inst) for inst in s])
            gap_qs = self.all_qubits - slice_qubits
            for q in gap_qs:
                # only fill up to the frontier
                if idx < frontier[q]:
                    s.append(Command(self.Ibox, [q]))

    def sort_and_fill_gaps(self) -> None:
        """Sort splices so each slice only contains independent instructions"""
        old_slices = self._slices.copy()
        self._slices = []
        frontier = {q: 0 for q in self.all_qubits}
        for s in old_slices:
            for inst in s:
                self._append(inst, frontier)
        self._fill_gaps(frontier)

    @staticmethod
    def _get_ubox(u: np.ndarray) -> Union[Unitary1qBox, Unitary2qBox, Unitary3qBox]:
        """Return a UnitaryxqBox for a given unitary"""
        if u.shape[0] == 2:
            return Unitary1qBox(u)
        if u.shape[0] == 4:
            return Unitary2qBox(u)
        if u.shape[0] == 8:
            return Unitary3qBox(u)
        raise ValueError(f"Unsupported unitary shape: {u.shape}")

    def unitary_factorisation(self) -> None:
        """For each unitary U with time D, factorise it into N*D unitaries u_i,
        such that u_0;u1;...;u_(N*D-1) = U.
        Store the factorised circuit as a list of slices
        """
        for cmd in self.circ:
            if cmd.op.type in [OpType.Measure, OpType.Reset]:
                self._slices.append([cmd])
            else:
                gt: Optional[float]
                if self.ct_params.virtual_z and cmd.op.type == OpType.Z:
                    gt = 1 / self.N
                else:
                    gt = self.ct_params.gate_times.get((cmd.op.type, tuple(cmd.qubits)))
                if gt is None:
                    raise ValueError(
                        f"No gate time for OpType {cmd.op.type} on qubits {cmd.qubits}"
                    )
                u = cmd.op.get_unitary()
                n_fractions = round(self.N * gt)
                if abs((self.N * gt) - n_fractions) > 1e-9:
                    raise ValueError(
                        f"Command {cmd} cannot be factorised into equal slices"
                    )
                power = 1 / n_fractions
                u_i = fractional_matrix_power(u, power)
                for _ in range(n_fractions):
                    u_i_box = self._get_ubox(u_i)
                    self._slices.append(
                        [FractionalUnitary(Command(u_i_box, cmd.args), n_fractions)]
                    )

    def _add_zz_crosstalks(self, noise_slice: Slice) -> None:
        for (q0, q1), zz in self.ct_params.zz_crosstalks.items():
            if q0 in self.all_qubits and q1 in self.all_qubits:
                Z = zz / self.N
                if abs(Z) > EPS:
                    noise_slice.append(
                        NoiseGate(
                            Command(Op.create(OpType.ZZPhase, Z), [q0, q1]),
                            "zz_crosstalks",
                        )
                    )

    def _add_single_q_phase(self, noise_slice: Slice) -> None:
        for q, z in self.ct_params.single_q_phase_errors.items():
            if q in self.all_qubits:
                Z = z / self.N
                if abs(Z) > EPS:
                    noise_slice.append(
                        NoiseGate(
                            Command(Op.create(OpType.Rz, Z), [q]), "single_q_phase"
                        )
                    )

    def _add_two_q_induced_phase(
        self, unitary_slice: Slice, noise_slice: Slice
    ) -> None:
        for inst in unitary_slice:
            if (
                isinstance(inst, FractionalUnitary)
                and inst.cmd.op.type == OpType.Unitary2qBox
            ):
                qubits = inst.cmd.qubits
                value = self.ct_params.two_q_induced_phase_errors.get(
                    (qubits[0], qubits[1])
                )
                if value is None:
                    # Don't add and error if the two-q interaction is not specified
                    continue
                Z = value[1] / inst.n_fractions
                if abs(Z) > EPS:
                    noise_slice.append(
                        NoiseGate(
                            Command(Op.create(OpType.Rz, Z), [value[0]]),
                            "two_q_induced_phase",
                        )
                    )

    def _add_non_markovian(self, noise_slice: Slice) -> None:
        for q, zx, zz in self.ct_params.non_markovian_noise:
            two_level_q = self.two_level_map[q]
            ZX = zx / self.N
            ZZ = zz / self.N
            if abs(ZZ) > EPS:
                noise_slice.append(
                    NoiseGate(
                        Command(Op.create(OpType.ZZPhase, ZZ), [two_level_q, q]),
                        "non_markovian",
                    )
                )
            if abs(ZX) > EPS:
                noise_slice.extend(
                    [
                        NoiseGate(
                            Command(Op.create(OpType.H), [q]),
                            "non_markovian",
                        ),
                        NoiseGate(
                            Command(Op.create(OpType.ZZPhase, ZX), [two_level_q, q]),
                            "non_markovian",
                        ),
                        NoiseGate(
                            Command(Op.create(OpType.H), [q]),
                            "non_markovian",
                        ),
                    ]
                )

    def add_noise(self) -> None:
        """Add noise gates between slices"""
        i = 1
        base_noise_slice: Slice = []
        self._add_zz_crosstalks(base_noise_slice)
        self._add_single_q_phase(base_noise_slice)
        self._add_non_markovian(base_noise_slice)
        while i < len(self._slices):
            noise_slice: Slice = []
            if self.circ.n_qubits > 2:
                self._add_two_q_induced_phase(self._slices[i - 1], noise_slice)
            noise_slice.extend(base_noise_slice)
            self._slices.insert(i, noise_slice)
            i = i + 2

    def build(self) -> None:
        """Build the noisy circuit as slices"""
        self.reset()
        self.unitary_factorisation()
        self.sort_and_fill_gaps()
        self.add_noise()

    def get_circuit(self) -> Circuit:
        """Convert the slices into a circuit"""
        d = Circuit()
        for q in self.circ.qubits:
            d.add_qubit(q)
        for q in self.two_level_map.values():
            d.add_qubit(q)
        for b in self.circ.bits:
            d.add_bit(b)
        for s in self._slices:
            for inst in s:
                if isinstance(inst, Command):
                    d.add_gate(inst.op, inst.args)
                else:
                    d.add_gate(inst.cmd.op, inst.cmd.args)
        return d

    def get_slices(self) -> List[Slice]:
        """Return the internally stored slices"""
        return self._slices


def get_gate_times_from_backendinfo(
    backend_info: BackendInfo,
) -> Dict[Tuple[OpType, Tuple[Qubit, ...]], float]:
    """Convert the gate time information stored in a `BackendInfo`
    into the format required by `NoisyCircuitBuilder`"""
    if (
        "characterisation" not in backend_info.misc
        or "GateTimes" not in backend_info.misc["characterisation"]
    ):
        raise ValueError("'GateTimes' is not present in the provided 'BackendInfo'")
    gate_times: Dict[Tuple[OpType, Tuple[Qubit, ...]], float] = {}
    for gt in backend_info.misc["characterisation"]["GateTimes"]:
        # GateTimes are nanoseconds
        gate_times[_gate_str_2_optype[gt[0]], tuple([Node(q) for q in gt[1]])] = (
            gt[2] / 1e9
        )
    return gate_times
