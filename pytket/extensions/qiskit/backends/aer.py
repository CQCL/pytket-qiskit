# Copyright Quantinuum
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

import itertools
import json
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from logging import warning
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from qiskit_aer import Aer  # type: ignore
from qiskit_aer.noise import NoiseModel  # type: ignore

from pytket.architecture import Architecture, FullyConnected
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, Node, OpType, Qubit
from pytket.passes import (
    AutoRebase,
    BasePass,
    CliffordSimp,
    CustomPassMap,
    DecomposeBoxes,
    FullPeepholeOptimise,
    GreedyPauliSimp,
    RemoveBarriers,
    SequencePass,
    SynthesiseTket,
)
from pytket.pauli import Pauli, QubitPauliString
from pytket.predicates import (
    ConnectivityPredicate,
    DefaultRegisterPredicate,
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoBarriersPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import KwargTypes
from qiskit import transpile  # type: ignore
from qiskit.quantum_info.operators import Pauli as qk_Pauli  # type: ignore
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import (  # type: ignore
    SparsePauliOp,
)

from .._metadata import __extension_version__
from ..qiskit_convert import (
    _gate_str_2_optype,
    tk_to_qiskit,
)
from ..result_convert import qiskit_result_to_backendresult
from .crosstalk_model import (
    CrosstalkParams,
    NoisyCircuitBuilder,
)
from .ibm_utils import _STATUS_MAP, _batch_circuits, _gen_lightsabre_transformation

if TYPE_CHECKING:
    from qiskit_aer import AerJob
    from qiskit_aer.backends.aerbackend import (  # type: ignore
        AerBackend as QiskitAerBackend,
    )


def _default_q_index(q: Qubit) -> int:
    if q.reg_name != "q" or len(q.index) != 1:
        raise ValueError("Non-default qubit register")
    return int(q.index[0])


def _tket_gate_set_from_qiskit_backend(
    qiskit_aer_backend: "QiskitAerBackend",
) -> set[OpType]:
    config = qiskit_aer_backend.configuration()
    gate_set = {
        _gate_str_2_optype[gate_str]
        for gate_str in config.basis_gates
        if gate_str in _gate_str_2_optype
    }

    gate_set.add(OpType.Barrier)

    if "unitary" in config.basis_gates:
        gate_set.add(OpType.Unitary1qBox)
        gate_set.add(OpType.Unitary2qBox)
        gate_set.add(OpType.Unitary3qBox)

    gate_set.add(OpType.Reset)
    gate_set.add(OpType.Measure)
    gate_set.add(OpType.Conditional)

    # special case mapping TK1 to U
    gate_set.add(OpType.TK1)
    return gate_set


def qiskit_aer_backend(backend_name: str) -> "QiskitAerBackend":
    """Find a qiskit backend with the given name.

    If more than one backend with the given name is available, emit a warning
    and return the first one in the list returned by `Aer.backends()`.
    """
    candidates = [b for b in Aer.backends() if b.name == backend_name]
    n_candidates = len(candidates)
    if n_candidates == 0:
        raise ValueError(f"No backend with name '{backend_name}' is available.")
    if n_candidates > 1:
        warnings.warn(  # noqa: B028
            f"More than one backend with name '{backend_name}' \
is available. Picking one."
        )
    return candidates[0]


class _AerBaseBackend(Backend):
    """Common base class for all Aer simulator backends"""

    _qiskit_backend: "QiskitAerBackend"
    _backend_info: BackendInfo
    _memory: bool
    _required_predicates: list[Predicate]
    _noise_model: NoiseModel | None = None
    _has_arch: bool = False
    _needs_transpile: bool = False

    # Map from (job ID, circuit index) to (number of qubits, postprocessing circuit),
    # i.e. from the first two components of the ResultHandle to the last two.
    _circuit_data: dict[
        tuple[int | float | complex | str | bool | bytes, int], tuple[int, str]
    ] = {}  # noqa: RUF012

    @property
    def required_predicates(self) -> list[Predicate]:
        return self._required_predicates

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str, int, int, str)

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    def rebase_pass(self) -> BasePass:
        return AutoRebase(
            self._backend_info.gate_set,
        )

    def _arch_dependent_default_compilation_pass(
        self,
        arch: Architecture,
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        assert optimisation_level in range(4)
        arch_specific_passes = [
            AutoRebase({OpType.CX, OpType.TK1}),
            CustomPassMap(_gen_lightsabre_transformation(arch), label="lightsabrepass"),
        ]
        if optimisation_level == 0:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    self.rebase_pass(),
                    *arch_specific_passes,
                    self.rebase_pass(),
                ],
            )
        if optimisation_level == 1:
            return SequencePass(
                [
                    DecomposeBoxes(),
                    SynthesiseTket(),
                    *arch_specific_passes,
                    SynthesiseTket(),
                ],
            )
        if optimisation_level == 2:  # noqa: PLR2004
            return SequencePass(
                [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(),
                    *arch_specific_passes,
                    CliffordSimp(False),
                    SynthesiseTket(),
                ],
            )
        return SequencePass(
            [
                DecomposeBoxes(),
                RemoveBarriers(),
                AutoRebase(
                    {
                        OpType.Z,
                        OpType.X,
                        OpType.Y,
                        OpType.S,
                        OpType.Sdg,
                        OpType.V,
                        OpType.Vdg,
                        OpType.H,
                        OpType.CX,
                        OpType.CY,
                        OpType.CZ,
                        OpType.SWAP,
                        OpType.Rz,
                        OpType.Rx,
                        OpType.Ry,
                        OpType.T,
                        OpType.Tdg,
                        OpType.ZZMax,
                        OpType.ZZPhase,
                        OpType.XXPhase,
                        OpType.YYPhase,
                        OpType.PhasedX,
                    }
                ),
                GreedyPauliSimp(thread_timeout=timeout, only_reduce=True, trials=10),
                *arch_specific_passes,
                self.rebase_pass(),
                SynthesiseTket(),
            ],
        )

    def _arch_independent_default_compilation_pass(
        self,
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        assert optimisation_level in range(4)
        if optimisation_level == 0:
            return SequencePass([DecomposeBoxes(), self.rebase_pass()])
        if optimisation_level == 1:
            return SequencePass([DecomposeBoxes(), SynthesiseTket()])
        if optimisation_level == 2:  # noqa: PLR2004
            return SequencePass([DecomposeBoxes(), FullPeepholeOptimise()])
        return SequencePass(
            [
                DecomposeBoxes(),
                RemoveBarriers(),
                AutoRebase(
                    {
                        OpType.Z,
                        OpType.X,
                        OpType.Y,
                        OpType.S,
                        OpType.Sdg,
                        OpType.V,
                        OpType.Vdg,
                        OpType.H,
                        OpType.CX,
                        OpType.CY,
                        OpType.CZ,
                        OpType.SWAP,
                        OpType.Rz,
                        OpType.Rx,
                        OpType.Ry,
                        OpType.T,
                        OpType.Tdg,
                        OpType.ZZMax,
                        OpType.ZZPhase,
                        OpType.XXPhase,
                        OpType.YYPhase,
                        OpType.PhasedX,
                    }
                ),
                GreedyPauliSimp(thread_timeout=timeout, only_reduce=True, trials=10),
                self.rebase_pass(),
                SynthesiseTket(),
            ],
        )

    def default_compilation_pass(
        self,
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        """
        See documentation for :py:meth:`IBMQBackend.default_compilation_pass`.
        """
        arch = self._backend_info.architecture
        if self._has_arch and arch.coupling:  # type: ignore
            return self._arch_dependent_default_compilation_pass(
                arch,  # type: ignore
                optimisation_level,
                timeout,
            )
        return self._arch_independent_default_compilation_pass(
            optimisation_level, timeout
        )

    def get_compiled_circuit(
        self, circuit: Circuit, optimisation_level: int = 2, timeout: int = 300
    ) -> Circuit:
        """
        Return a single circuit compiled with :py:meth:`default_compilation_pass`.

        :param optimisation_level: Allows values of 0, 1, 2 or 3, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :type optimisation_level: int, optional
        :param timeout: Only valid for optimisation level 3, gives a maximum time
            for running a single thread of the pass `GreedyPauliSimp`. Increase for
            optimising larger circuits.
        :type timeout: int, optional

        :return: An optimised quantum circuit
        :rtype: Circuit
        """
        return_circuit = circuit.copy()
        if optimisation_level == 3 and circuit.n_gates_of_type(OpType.Barrier) > 0:  # noqa: PLR2004
            warnings.warn(  # noqa: B028
                "Barrier operations in this circuit will be removed when using "
                "optimisation level 3."
            )
        self.default_compilation_pass(optimisation_level, timeout).apply(return_circuit)
        return return_circuit

    def get_compiled_circuits(
        self,
        circuits: Sequence[Circuit],
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> list[Circuit]:
        """Compile a sequence of circuits with :py:meth:`default_compilation_pass`
        and return the list of compiled circuits (does not act in place).

        As well as applying a degree of optimisation (controlled by the
        `optimisation_level` parameter), this method tries to ensure that the circuits
        can be run on the backend (i.e. successfully passed to
        :py:meth:`process_circuits`), for example by rebasing to the supported gate set,
        or routing to match the connectivity of the device. However, this is not always
        possible, for example if the circuit contains classical operations that are not
        supported by the backend. You may use :py:meth:`valid_circuit` to check whether
        the circuit meets the backend's requirements after compilation. This validity
        check is included in :py:meth:`process_circuits` by default, before any circuits
        are submitted to the backend.

        If the validity check fails, you can obtain more information about the failure
        by iterating through the predicates in the `required_predicates` property of the
        backend, and running the :py:meth:`verify` method on each in turn with your
        circuit.

        :param circuits: The circuits to compile.
        :type circuit: Sequence[Circuit]
        :param optimisation_level: The level of optimisation to perform during
            compilation. See :py:meth:`default_compilation_pass` for a description of
            the different levels (0, 1, 2 or 3). Defaults to 2.
        :type optimisation_level: int, optional
        :param timeout: Only valid for optimisation level 3, gives a maximum time
            for running a single thread of the pass `GreedyPauliSimp`. Increase for
            optimising larger circuits.
        :type timeout: int, optional
        :return: Compiled circuits.
        :rtype: List[Circuit]
        """
        return [
            self.get_compiled_circuit(c, optimisation_level, timeout) for c in circuits
        ]

    def process_circuits(  # noqa: PLR0912
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[int | None] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`, `postprocess`.
        """
        postprocess = kwargs.get("postprocess", False)

        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(  # noqa: SLF001
            n_shots,
            len(circuits),
            optional=True,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        if hasattr(self, "_crosstalk_params") and self._crosstalk_params is not None:
            noisy_circuits = []
            for c in circuits:
                noisy_circ_builder = NoisyCircuitBuilder(c, self._crosstalk_params)
                noisy_circ_builder.build()
                noisy_circuits.append(noisy_circ_builder.get_circuit())
            circuits = noisy_circuits

        handle_list: list[ResultHandle | None] = [None] * len(circuits)
        seed = kwargs.get("seed")
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        replace_implicit_swaps = self.supports_state or self.supports_unitary

        for (n_shots, batch), indices in zip(  # noqa: PLR1704
            circuit_batches, batch_order, strict=False
        ):
            qcs, ppcirc_strs, tkc_qubits_count = [], [], []
            for tkc in batch:
                if postprocess:
                    c0, ppcirc = prepare_circuit(tkc, allow_classical=False)
                    ppcirc_rep = ppcirc.to_dict()
                else:
                    c0, ppcirc_rep = tkc, None

                qc = tk_to_qiskit(c0, replace_implicit_swaps, perm_warning=False)

                if self.supports_state:
                    qc.save_state()

                elif self.supports_density_matrix:
                    qc.save_density_matrix()

                elif self.supports_unitary:
                    qc.save_unitary()

                qcs.append(qc)
                tkc_qubits_count.append(c0.n_qubits)
                ppcirc_strs.append(json.dumps(ppcirc_rep))

            if self._needs_transpile:
                qcs = transpile(qcs, self._qiskit_backend)

            job = self._qiskit_backend.run(
                qcs,
                shots=n_shots,
                memory=self._memory,
                seed_simulator=seed,
                noise_model=self._noise_model,
            )
            if type(seed) is int:
                seed += 1
            jobid = job.job_id()
            for i, ind in enumerate(indices):
                handle = ResultHandle(jobid, i, tkc_qubits_count[i], ppcirc_strs[i])
                handle_list[ind] = handle
                self._circuit_data[(jobid, i)] = (tkc_qubits_count[i], ppcirc_strs[i])
                self._cache[handle] = {"job": job}
        return cast("list[ResultHandle]", handle_list)

    def cancel(self, handle: ResultHandle) -> None:
        job: AerJob = self._cache[handle]["job"]
        cancelled = job.cancel()
        if not cancelled:
            warning(f"Unable to cancel job {cast('str', handle[0])}")  # noqa: LOG015, G004

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        job: AerJob = self._cache[handle]["job"]
        ibmstatus = job.status()
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus.value)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = handle[0]
            try:
                job: AerJob = self._cache[handle]["job"]
            except KeyError:
                raise CircuitNotRunError(handle)  # noqa: B904

            res = job.result()
            backresults = qiskit_result_to_backendresult(
                res,
                include_shots=self._supports_shots,
                include_counts=self._supports_counts,
                include_state=self._supports_state,
                include_unitary=self._supports_unitary,
                include_density_matrix=self._supports_density_matrix,
            )
            for circ_index, backres in enumerate(backresults):
                qubit_n, ppc = self._circuit_data[(jobid, circ_index)]
                self._cache[ResultHandle(jobid, circ_index, qubit_n, ppc)]["result"] = (
                    backres
                )

            return cast("BackendResult", self._cache[handle]["result"])

    def _snapshot_expectation_value(
        self,
        circuit: Circuit,
        hamiltonian: SparsePauliOp | qk_Pauli,
        valid_check: bool = True,
    ) -> complex:
        if valid_check:
            self._check_all_circuits([circuit], nomeasure_warn=False)

        circ_qbs = circuit.qubits
        q_indices = (_default_q_index(q) for q in circ_qbs)
        if not all(
            q_ind == i
            for q_ind, i in zip(q_indices, range(len(circ_qbs)), strict=False)
        ):
            raise ValueError(
                "Circuit must act on default register Qubits, contiguously from 0"  # noqa: ISC003
                + f" onwards. Circuit qubits were: {circ_qbs}"
            )
        qc = tk_to_qiskit(circuit)
        qc.save_expectation_value(hamiltonian, qc.qubits, "snap")
        job = self._qiskit_backend.run(qc)
        return cast(
            "complex",
            job.result().data(qc)["snap"],
        )

    def get_pauli_expectation_value(
        self,
        state_circuit: Circuit,
        pauli: QubitPauliString,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit using the built-in Aer
        snapshot functionality
        Requires a simple circuit with default register qubits.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :param pauli: Pauli operator
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :return: :math:`\\left<\\psi | P | \\psi \\right>`
        """
        if self._noise_model:
            raise RuntimeError(
                "Snapshot based expectation value not supported with noise model. "
                "Use shots."
            )
        if not self._supports_expectation:
            raise NotImplementedError("Cannot get expectation value from this backend")

        operator = qk_Pauli(_sparse_to_zx_tup(pauli, state_circuit.n_qubits))
        return self._snapshot_expectation_value(state_circuit, operator, valid_check)

    def get_operator_expectation_value(
        self,
        state_circuit: Circuit,
        operator: QubitPauliOperator,
        valid_check: bool = True,
    ) -> complex:
        """Calculates the expectation value of the given circuit with respect to the
        operator using the built-in Aer snapshot functionality
        Requires a simple circuit with default register qubits.

        :param state_circuit: Circuit that generates the desired state
            :math:`\\left|\\psi\\right>`.
        :param operator: Operator :math:`H`.
        :param valid_check: Explicitly check that the circuit satisfies all required
            predicates to run on the backend. Defaults to True
        :return: :math:`\\left<\\psi | H | \\psi \\right>`
        """
        if self._noise_model:
            raise RuntimeError(
                "Snapshot based expectation value not supported with noise model. "
                "Use shots."
            )
        if not self._supports_expectation:
            raise NotImplementedError("Cannot get expectation value from this backend")

        sparse_op = _qubitpauliop_to_sparsepauliop(operator, state_circuit.n_qubits)
        return self._snapshot_expectation_value(state_circuit, sparse_op, valid_check)


@dataclass(frozen=True)
class NoiseModelCharacterisation:
    """Class to hold information from the processing of the noise model"""

    architecture: Architecture
    node_errors: dict[Node, dict[OpType, float]] | None = None
    edge_errors: dict[tuple[Node, Node], dict[OpType, float]] | None = None
    readout_errors: dict[Node, list[list[float]]] | None = None
    averaged_node_errors: dict[Node, float] | None = None
    averaged_edge_errors: dict[tuple[Node, Node], float] | None = None
    averaged_readout_errors: dict[Node, float] | None = None
    generic_q_errors: dict[str, Any] | None = None


def _map_trivial_noise_model_to_none(
    noise_model: NoiseModel | None,
) -> NoiseModel | None:
    if noise_model and all(value == [] for value in noise_model.to_dict().values()):
        return None
    return noise_model


def _get_characterisation_of_noise_model(
    noise_model: NoiseModel | None, gate_set: set[OpType]
) -> NoiseModelCharacterisation:
    if noise_model is None:
        return NoiseModelCharacterisation(architecture=Architecture([]))
    return _process_noise_model(noise_model, gate_set)


class AerBackend(_AerBaseBackend):
    """
    Backend for running simulations on the Qiskit Aer QASM simulator.

    :param noise_model: Noise model to apply during simulation. Defaults to None.
    :param simulation_method: Simulation method, see
        https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
        for available values. Defaults to "automatic".
    :param crosstalk_params: Apply crosstalk noise simulation to the circuits before
        execution. `noise_model` will be overwritten if this is given. Default to None.
    :param n_qubits: The maximum number of qubits supported by the backend.
    """

    _persistent_handles: bool = False
    _supports_shots: bool = True
    _supports_counts: bool = True
    _supports_expectation: bool = True
    _expectation_allows_nonhermitian: bool = False

    _memory: bool = True

    _qiskit_backend_name: str = "aer_simulator"
    _allowed_special_gates: set[OpType] = {  # noqa: RUF012
        OpType.Measure,
        OpType.Barrier,
        OpType.Reset,
        OpType.RangePredicate,
    }

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        simulation_method: str = "automatic",
        crosstalk_params: CrosstalkParams | None = None,
        n_qubits: int = 40,
    ):
        super().__init__()
        self._qiskit_backend = qiskit_aer_backend(self._qiskit_backend_name)
        self._qiskit_backend.set_options(method=simulation_method)
        gate_set: set[OpType] = _tket_gate_set_from_qiskit_backend(
            self._qiskit_backend
        ).union(self._allowed_special_gates)

        self._crosstalk_params = crosstalk_params
        if self._crosstalk_params is not None:
            self._noise_model = self._crosstalk_params.get_noise_model()
            self._backend_info = BackendInfo(
                name=type(self).__name__,
                device_name=self._qiskit_backend_name,
                version=__extension_version__,
                architecture=Architecture([]),
                gate_set=gate_set,
            )
        else:
            self._noise_model = _map_trivial_noise_model_to_none(noise_model)
            characterisation = _get_characterisation_of_noise_model(
                self._noise_model, gate_set
            )
            self._has_arch = bool(characterisation.architecture) and bool(
                characterisation.architecture.nodes
            )

            self._backend_info = BackendInfo(
                name=type(self).__name__,
                device_name=self._qiskit_backend_name,
                version=__extension_version__,
                architecture=(
                    characterisation.architecture
                    if self._has_arch
                    else FullyConnected(n_qubits)
                ),
                gate_set=gate_set,
                supports_midcircuit_measurement=True,  # is this correct?
                supports_fast_feedforward=True,
                all_node_gate_errors=characterisation.node_errors,
                all_edge_gate_errors=characterisation.edge_errors,
                all_readout_errors=characterisation.readout_errors,
                averaged_node_gate_errors=characterisation.averaged_node_errors,
                averaged_edge_gate_errors=characterisation.averaged_edge_errors,
                averaged_readout_errors=characterisation.averaged_readout_errors,
                misc={"characterisation": characterisation.generic_q_errors},
            )

        self._required_predicates = [
            NoSymbolsPredicate(),
            GateSetPredicate(self._backend_info.gate_set),
            MaxNQubitsPredicate(n_qubits),
        ]
        if self._crosstalk_params is not None:
            self._required_predicates.extend(
                [
                    NoClassicalControlPredicate(),
                    DefaultRegisterPredicate(),
                    NoBarriersPredicate(),
                ]
            )

        if self._has_arch:
            # architecture is non-trivial
            self._required_predicates.append(
                ConnectivityPredicate(self._backend_info.architecture)  # type: ignore
            )


class AerStateBackend(_AerBaseBackend):
    """
    Backend for running simulations on the Qiskit Aer Statevector simulator.

    :param n_qubits: The maximum number of qubits supported by the backend.
    """

    _persistent_handles: bool = False
    _supports_state: bool = True
    _supports_expectation: bool = True
    _expectation_allows_nonhermitian: bool = False

    _noise_model: NoiseModel | None = None
    _memory: bool = False

    _qiskit_backend_name: str = "aer_simulator_statevector"

    def __init__(
        self,
        n_qubits: int = 40,
    ) -> None:
        super().__init__()
        self._qiskit_backend = qiskit_aer_backend(self._qiskit_backend_name)
        self._backend_info = BackendInfo(
            name=type(self).__name__,
            device_name=self._qiskit_backend_name,
            version=__extension_version__,
            architecture=FullyConnected(n_qubits),
            gate_set=_tket_gate_set_from_qiskit_backend(self._qiskit_backend),
            supports_midcircuit_measurement=True,
            supports_reset=True,
            supports_fast_feedforward=True,
            misc={"characterisation": None},
        )
        self._required_predicates = [
            GateSetPredicate(self._backend_info.gate_set),
        ]


class AerUnitaryBackend(_AerBaseBackend):
    """Backend for running simulations on the Qiskit Aer Unitary simulator.

    :param n_qubits: The maximum number of qubits supported by the backend.
    """

    _persistent_handles: bool = False
    _supports_unitary: bool = True

    _memory: bool = False
    _noise_model: NoiseModel | None = None
    _needs_transpile: bool = True

    _qiskit_backend_name: str = "aer_simulator_unitary"

    def __init__(self, n_qubits: int = 40) -> None:
        super().__init__()
        self._qiskit_backend = qiskit_aer_backend(self._qiskit_backend_name)
        self._backend_info = BackendInfo(
            name=type(self).__name__,
            device_name=self._qiskit_backend_name,
            version=__extension_version__,
            architecture=FullyConnected(n_qubits),
            gate_set=_tket_gate_set_from_qiskit_backend(self._qiskit_backend),
            supports_midcircuit_measurement=True,  # is this correct?
            misc={"characterisation": None},
        )
        self._required_predicates = [
            NoClassicalControlPredicate(),
            NoFastFeedforwardPredicate(),
            GateSetPredicate(self._backend_info.gate_set),
        ]


class AerDensityMatrixBackend(_AerBaseBackend):
    """
    Backend for running simulations on the Qiskit Aer density matrix simulator.

    :param noise_model: Noise model to apply during simulation. Defaults to None.
    :param n_qubits: The maximum number of qubits supported by the backend.
    """

    _supports_density_matrix: bool = True
    _supports_state: bool = False
    _memory: bool = False
    _noise_model: NoiseModel | None = None
    _needs_transpile: bool = True
    _supports_expectation: bool = True

    _qiskit_backend_name: str = "aer_simulator_density_matrix"

    _allowed_special_gates: set[OpType] = {  # noqa: RUF012
        OpType.Measure,
        OpType.Barrier,
        OpType.Reset,
        OpType.RangePredicate,
    }

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        n_qubits: int = 40,
    ) -> None:
        super().__init__()
        self._qiskit_backend = qiskit_aer_backend(self._qiskit_backend_name)

        gate_set: set[OpType] = _tket_gate_set_from_qiskit_backend(
            self._qiskit_backend
        ).union(self._allowed_special_gates)
        self._noise_model = _map_trivial_noise_model_to_none(noise_model)
        characterisation: NoiseModelCharacterisation = (
            _get_characterisation_of_noise_model(self._noise_model, gate_set)
        )
        self._has_arch: bool = bool(characterisation.architecture) and bool(
            characterisation.architecture.nodes
        )

        self._backend_info = BackendInfo(
            name=type(self).__name__,
            device_name=self._qiskit_backend_name,
            version=__extension_version__,
            architecture=(
                FullyConnected(n_qubits)
                if not self._has_arch
                else characterisation.architecture
            ),
            gate_set=_tket_gate_set_from_qiskit_backend(self._qiskit_backend),
            supports_midcircuit_measurement=True,
            supports_reset=True,
            supports_fast_feedforward=True,
            all_node_gate_errors=characterisation.node_errors,
            all_edge_gate_errors=characterisation.edge_errors,
            all_readout_errors=characterisation.readout_errors,
            averaged_node_gate_errors=characterisation.averaged_node_errors,
            averaged_edge_gate_errors=characterisation.averaged_edge_errors,
            averaged_readout_errors=characterisation.averaged_readout_errors,
            misc={"characterisation": characterisation.generic_q_errors},
        )
        self._required_predicates = [
            GateSetPredicate(self._backend_info.gate_set),
        ]


def _process_noise_model(
    noise_model: NoiseModel, gate_set: set[OpType]
) -> NoiseModelCharacterisation:
    # obtain approximations for gate errors from noise model by using probability of
    #  "identity" error
    assert OpType.CX in gate_set
    # TODO explicitly check for and separate 1 and 2 qubit gates
    errors = [
        e
        for e in noise_model.to_dict()["errors"]
        if e["type"] == "qerror" or e["type"] == "roerror"
    ]

    node_errors: dict[Node, dict[OpType, float]] = defaultdict(dict)
    link_errors: dict[tuple[Node, Node], dict[OpType, float]] = defaultdict(dict)
    readout_errors: dict[Node, list[list[float]]] = {}

    generic_single_qerrors_dict: dict = defaultdict(list)
    generic_2q_qerrors_dict: dict = defaultdict(list)

    qubits_set: set = set()
    # remember which qubits have explicit link errors
    qubits_with_link_errors: set = set()

    coupling_map = []
    for error in errors:
        name = error["operations"]
        if len(name) > 1:
            raise RuntimeWarning("Error applies to multiple gates.")
        if "gate_qubits" not in error:
            raise RuntimeWarning(
                "Please define NoiseModel without using the"
                " add_all_qubit_quantum_error()"
                " or add_all_qubit_readout_error() method."
            )
        name = name[0]

        qubits = error["gate_qubits"][0]
        gate_fid = error["probabilities"][0]
        if len(qubits) == 1:
            [q] = qubits
            optype = _gate_str_2_optype[name]
            qubits_set.add(q)
            if error["type"] == "qerror":
                node_errors[Node(q)].update({optype: float(1 - gate_fid)})
                generic_single_qerrors_dict[q].append(
                    [error["instructions"], error["probabilities"]]
                )
            elif error["type"] == "roerror":
                readout_errors[Node(q)] = cast(
                    "list[list[float]]", error["probabilities"]
                )
            else:
                raise RuntimeWarning("Error type not 'qerror' or 'roerror'.")
        elif len(qubits) == 2:  # noqa: PLR2004
            # note that if multiple multi-qubit errors are added to the CX gate,
            #  the resulting noise channel is composed and reflected in probabilities
            [q0, q1] = qubits
            optype = _gate_str_2_optype[name]
            link_errors.update()
            link_errors[(Node(q0), Node(q1))].update({optype: float(1 - gate_fid)})
            qubits_with_link_errors.add(q0)
            qubits_with_link_errors.add(q1)
            # to simulate a worse reverse direction square the fidelity
            link_errors[(Node(q1), Node(q0))].update({optype: float(1 - gate_fid**2)})
            generic_2q_qerrors_dict[(q0, q1)].append(
                [error["instructions"], error["probabilities"]]
            )
            coupling_map.append(qubits)

    # free qubits (ie qubits with no link errors) have full connectivity
    free_qubits = qubits_set - qubits_with_link_errors

    for q in free_qubits:
        for lq in qubits_with_link_errors:
            coupling_map.append([q, lq])
            coupling_map.append([lq, q])

    for pair in itertools.permutations(free_qubits, 2):
        coupling_map.append(pair)  # noqa: PERF402

    generic_q_errors = {
        "GenericOneQubitQErrors": [
            [k, v] for k, v in generic_single_qerrors_dict.items()
        ],
        "GenericTwoQubitQErrors": [
            [list(k), v] for k, v in generic_2q_qerrors_dict.items()
        ],
    }

    averaged_node_errors: dict[Node, float] = {
        k: sum(v.values()) / len(v) for k, v in node_errors.items()
    }
    averaged_link_errors = {k: sum(v.values()) / len(v) for k, v in link_errors.items()}
    averaged_readout_errors = {
        k: (v[0][1] + v[1][0]) / 2.0 for k, v in readout_errors.items()
    }

    return NoiseModelCharacterisation(
        node_errors=dict(node_errors),
        edge_errors=dict(link_errors),
        readout_errors=readout_errors,
        averaged_node_errors=averaged_node_errors,
        averaged_edge_errors=averaged_link_errors,
        averaged_readout_errors=averaged_readout_errors,
        generic_q_errors=generic_q_errors,
        architecture=Architecture(coupling_map),
    )


def _sparse_to_zx_tup(
    pauli: QubitPauliString, n_qubits: int
) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(n_qubits, dtype=np.bool_)
    z = np.zeros(n_qubits, dtype=np.bool_)
    for q, p in pauli.map.items():
        i = _default_q_index(q)
        z[i] = p in (Pauli.Z, Pauli.Y)
        x[i] = p in (Pauli.X, Pauli.Y)
    return (z, x)


def _qubitpauliop_to_sparsepauliop(
    operator: QubitPauliOperator, n_qubits: int
) -> SparsePauliOp:
    strings, coeffs = [], []
    for term, coeff in operator._dict.items():  # noqa: SLF001
        termmap = term.map
        strings.append(
            "".join(
                termmap.get(Qubit(i), Pauli.I).name for i in reversed(range(n_qubits))
            )
        )
        coeffs.append(coeff)

    return SparsePauliOp(strings, coeffs)
