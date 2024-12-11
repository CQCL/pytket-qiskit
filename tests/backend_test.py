# Copyright 2019-2024 Quantinuum
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
import cmath
import json
import math
import os
from collections import Counter
from typing import cast

import numpy as np
import pytest
from hypothesis import given, strategies
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore
from qiskit.circuit import Parameter  # type: ignore
from qiskit_aer import Aer  # type: ignore
from qiskit_aer.noise import ReadoutError  # type: ignore
from qiskit_aer.noise.errors import depolarizing_error, pauli_error  # type: ignore
from qiskit_aer.noise.noise_model import NoiseModel  # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

from pytket.architecture import Architecture, FullyConnected
from pytket.backends import (
    CircuitNotRunError,
    CircuitNotValidError,
    CircuitStatus,
    ResultHandle,
    StatusEnum,
)
from pytket.backends.backend import ResultHandleTypeError
from pytket.circuit import (
    BasisOrder,
    CircBox,
    Circuit,
    OpType,
    QControlBox,
    Qubit,
    Unitary2qBox,
    reg_eq,
)
from pytket.extensions.qiskit import (
    AerBackend,
    AerDensityMatrixBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQBackend,
    IBMQEmulatorBackend,
    process_characterisation,
    qiskit_to_tk,
    tk_to_qiskit,
)
from pytket.extensions.qiskit.backends.crosstalk_model import (
    CrosstalkParams,
    FractionalUnitary,
    NoisyCircuitBuilder,
)
from pytket.extensions.qiskit.backends.ibm_utils import _gen_lightsabre_transformation
from pytket.mapping import LexiLabellingMethod, LexiRouteRoutingMethod, MappingManager
from pytket.passes import CliffordSimp, FlattenRelabelRegistersPass, SequencePass
from pytket.pauli import Pauli, QubitPauliString
from pytket.predicates import (
    CompilationUnit,
    ConnectivityPredicate,
    NoMidMeasurePredicate,
)
from pytket.transform import Transform
from pytket.utils.expectations import (
    get_operator_expectation_value,
    get_pauli_expectation_value,
)
from pytket.utils.operators import QubitPauliOperator
from pytket.utils.results import compare_statevectors, compare_unitaries

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of IBMQ account)"


def circuit_gen(measure: bool = False) -> Circuit:
    c = Circuit(2, 2)
    c.H(0)
    c.CX(0, 1)
    if measure:
        c.measure_all()
    return c


def get_test_circuit(measure: bool) -> QuantumCircuit:
    qr = QuantumRegister(5)
    cr = ClassicalRegister(5)
    qc = QuantumCircuit(qr, cr)
    # qc.h(qr[0])
    qc.x(qr[0])
    qc.x(qr[2])
    qc.cx(qr[1], qr[0])
    # qc.h(qr[1])
    qc.cx(qr[0], qr[3])
    qc.cz(qr[2], qr[0])
    qc.cx(qr[1], qr[3])
    # qc.rx(PI/2,qr[3])
    qc.z(qr[2])
    if measure:
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        qc.measure(qr[3], cr[3])
    return qc


def test_statevector() -> None:
    c = circuit_gen()
    b = AerStateBackend()
    state = b.run_circuit(c).get_state()
    assert np.allclose(state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    c.add_phase(0.5)
    state1 = b.run_circuit(c).get_state()
    assert np.allclose(state1, state * 1j, atol=1e-10)


def test_statevector_sim_with_permutation() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/35
    b = AerStateBackend()
    c = Circuit(3).X(0).SWAP(0, 1).SWAP(0, 2)
    qubits = c.qubits
    sv = b.run_circuit(c).get_state()
    # convert swaps to implicit permutation
    c.replace_SWAPs()
    assert c.implicit_qubit_permutation() == {
        qubits[0]: qubits[1],
        qubits[1]: qubits[2],
        qubits[2]: qubits[0],
    }
    sv1 = b.run_circuit(c).get_state()
    assert np.allclose(sv, sv1, atol=1e-10)


def test_sim() -> None:
    c = circuit_gen(True)
    b = AerBackend()
    b.run_circuit(c, n_shots=1024).get_shots()


def test_measures() -> None:
    n_qbs = 12
    c = Circuit(n_qbs, n_qbs)
    x_qbs = [2, 5, 7, 11]
    for i in x_qbs:
        c.X(i)
    c.measure_all()
    b = AerBackend()
    shots = b.run_circuit(c, n_shots=10).get_shots()
    all_ones = True
    all_zeros = True
    for i in x_qbs:
        all_ones = all_ones and bool(np.all(shots[:, i]))
    for i in range(n_qbs):
        if i not in x_qbs:
            all_zeros = all_zeros and (not np.any(shots[:, i]))
    assert all_ones
    assert all_zeros


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_noise(brisbane_backend: IBMQBackend) -> None:
    noise_model = NoiseModel.from_backend(brisbane_backend._backend)
    n_qbs = 5
    c = Circuit(n_qbs, n_qbs)
    x_qbs = [2, 0, 4]
    for i in x_qbs:
        c.X(i)
    c.measure_all()
    b = AerBackend(noise_model)
    n_shots = 50
    c = b.get_compiled_circuit(c)
    shots = b.run_circuit(c, n_shots=n_shots, seed=4).get_shots()
    zer_exp = []
    one_exp = []
    for i in range(n_qbs):
        expectation = np.sum(shots[:, i]) / n_shots
        if i in x_qbs:
            one_exp.append(expectation)
        else:
            zer_exp.append(expectation)

    assert min(one_exp) > max(zer_exp)

    c2 = (
        Circuit(4, 4)
        .H(0)
        .CX(0, 2)
        .CX(3, 1)
        .T(2)
        .CX(0, 1)
        .CX(0, 3)
        .CX(2, 1)
        .measure_all()
    )

    c2 = b.get_compiled_circuit(c2)
    shots = b.run_circuit(c2, n_shots=10, seed=5).get_shots()
    assert shots.shape == (10, 4)


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_process_characterisation(brisbane_backend: IBMQBackend) -> None:
    char = process_characterisation(brisbane_backend._backend)
    arch: Architecture = char.get("Architecture", Architecture([]))
    node_errors: dict = char.get("NodeErrors", {})
    link_errors: dict = char.get("EdgeErrors", {})

    assert len(arch.nodes) == 127
    assert len(arch.coupling) == 144
    assert len(node_errors) == 127
    assert len(link_errors) == 288


def test_process_characterisation_no_noise_model() -> None:
    my_noise_model = NoiseModel()
    back = AerBackend(my_noise_model)
    assert back.backend_info.get_misc("characterisation") is None

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0)
    c = back.get_compiled_circuit(c)
    assert back.valid_circuit(c)


def test_process_characterisation_incomplete_noise_model() -> None:
    my_noise_model = NoiseModel()

    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [1])
    my_noise_model.add_quantum_error(depolarizing_error(0.1, 1), ["u3"], [3])
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Z", 0.65)]), ["u2"], [0]
    )
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Y", 0.65)]), ["u1"], [2]
    )

    back = AerBackend(my_noise_model)

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0).measure_all()
    c = back.get_compiled_circuit(c)
    assert back.valid_circuit(c)

    arch = back.backend_info.architecture
    assert isinstance(arch, Architecture)
    nodes = arch.nodes
    assert set(arch.coupling) == {
        (nodes[0], nodes[1]),
        (nodes[0], nodes[2]),
        (nodes[0], nodes[3]),
        (nodes[1], nodes[2]),
        (nodes[1], nodes[3]),
        (nodes[2], nodes[0]),
        (nodes[2], nodes[1]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[0]),
        (nodes[3], nodes[1]),
        (nodes[3], nodes[2]),
    }


def test_circuit_compilation_complete_noise_model() -> None:
    my_noise_model = NoiseModel()
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 2])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [1, 2])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [1, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [2, 3])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [0])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [2])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [3])

    back = AerBackend(my_noise_model)

    c = Circuit(4).CX(0, 1).H(2).CX(2, 1).H(3).CX(0, 3).H(1).X(0).measure_all()
    c = back.get_compiled_circuit(c)
    assert back.valid_circuit(c)


def test_process_characterisation_complete_noise_model() -> None:
    my_noise_model = NoiseModel()

    readout_error_0 = 0.2
    readout_error_1 = 0.3
    my_noise_model.add_readout_error(
        [
            [1 - readout_error_0, readout_error_0],
            [readout_error_0, 1 - readout_error_0],
        ],
        [0],
    )
    my_noise_model.add_readout_error(
        [
            [1 - readout_error_1, readout_error_1],
            [readout_error_1, 1 - readout_error_1],
        ],
        [1],
    )

    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [0])
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Z", 0.65)]), ["u2"], [0]
    )
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Y", 0.65)]), ["u1"], [0]
    )

    back = AerBackend(my_noise_model)
    char = back.backend_info.get_misc("characterisation")

    node_errors = cast(dict, back.backend_info.all_node_gate_errors)
    link_errors = cast(dict, back.backend_info.all_edge_gate_errors)
    arch = back.backend_info.architecture
    assert node_errors is not None
    assert link_errors is not None
    assert arch is not None

    gqe2 = {tuple(qs): errs for qs, errs in char["GenericTwoQubitQErrors"]}
    gqe1 = {q: errs for q, errs in char["GenericOneQubitQErrors"]}

    assert round(gqe2[(0, 1)][0][1][15], 5) == 0.0375
    assert round(gqe2[(0, 1)][0][1][0], 5) == 0.4375
    assert gqe1[0][0][1][3] == 0.125
    assert gqe1[0][0][1][0] == 0.625
    assert gqe1[0][1][1][0] == 0.35
    assert gqe1[0][1][1][1] == 0.65
    assert gqe1[0][2][1][0] == 0.35
    assert gqe1[0][2][1][1] == 0.65
    assert node_errors[arch.nodes[0]][OpType.U3] == 0.375
    assert round(link_errors[(arch.nodes[0], arch.nodes[1])][OpType.CX], 4) == 0.5625
    assert (
        round(link_errors[(arch.nodes[1], arch.nodes[0])][OpType.CX], 8) == 0.80859375
    )
    readout_errors = cast(dict, back.backend_info.all_readout_errors)
    assert readout_errors[arch.nodes[0]] == [
        [0.8, 0.2],
        [0.2, 0.8],
    ]
    assert readout_errors[arch.nodes[1]] == [
        [0.7, 0.3],
        [0.3, 0.7],
    ]


def test_process_model() -> None:
    noise_model = NoiseModel()
    # add readout error to qubits 0, 1, 2
    error_ro = ReadoutError([[0.8, 0.2], [0.2, 0.8]])
    for i in range(3):
        noise_model.add_readout_error(error_ro, [i])
    # add depolarizing error to qubits 3, 4, 5
    error_dp_sq = depolarizing_error(0.5, 1)
    for i in range(3, 6):
        noise_model.add_quantum_error(error_dp_sq, ["u3"], [i])
    error_dp_mq = depolarizing_error(0.6, 2)
    # add coupling errors
    noise_model.add_quantum_error(error_dp_mq, ["cx"], [0, 7])
    noise_model.add_quantum_error(error_dp_mq, ["cx"], [1, 2])
    noise_model.add_quantum_error(error_dp_mq, ["cx"], [8, 9])

    # check basic information has been captured
    b = AerBackend(noise_model)
    assert b.backend_info.architecture is not None
    nodes = b.backend_info.architecture.nodes
    assert len(nodes) == 9
    assert "characterisation" in b.backend_info.misc
    assert "GenericOneQubitQErrors" in b.backend_info.misc["characterisation"]
    assert "GenericTwoQubitQErrors" in b.backend_info.misc["characterisation"]
    node_gate_errors = cast(dict, b.backend_info.all_node_gate_errors)
    assert nodes[3] in node_gate_errors
    edge_gate_errors = cast(dict, b.backend_info.all_edge_gate_errors)
    assert (nodes[7], nodes[8]) in edge_gate_errors


def test_cancellation_aer() -> None:
    b = AerBackend()
    c = circuit_gen(True)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_cancellation_ibmq(brisbane_backend: IBMQBackend) -> None:
    b = brisbane_backend
    c = circuit_gen(True)
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, 10)
    b.cancel(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_machine_debug(brisbane_backend: IBMQBackend) -> None:
    backend = brisbane_backend
    backend._MACHINE_DEBUG = True
    try:
        c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
        with pytest.raises(CircuitNotValidError) as errorinfo:
            handles = backend.process_circuits([c, c.copy()], n_shots=2)
        assert "in submitted does not satisfy GateSetPredicate" in str(errorinfo.value)
        c = backend.get_compiled_circuit(c)
        handles = backend.process_circuits([c, c.copy()], n_shots=4)
        from pytket.extensions.qiskit.backends.ibm import _DEBUG_HANDLE_PREFIX

        assert all(
            cast(str, hand[0]).startswith(_DEBUG_HANDLE_PREFIX) for hand in handles
        )

        correct_counts = {(0, 0): 4}

        res = backend.run_circuit(c, n_shots=4)
        assert res.get_counts() == correct_counts

        # check that generating new shots still works
        res = backend.run_circuit(c, n_shots=4)
        assert res.get_counts() == correct_counts
    finally:
        # ensure shared backend is reset for other tests
        backend._MACHINE_DEBUG = False


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_nshots_batching(brisbane_backend: IBMQBackend) -> None:
    backend = brisbane_backend
    backend._MACHINE_DEBUG = True
    try:
        c1 = Circuit(2, 2).H(0).CX(0, 1).measure_all()
        c2 = Circuit(2, 2).Rx(0.5, 0).CX(0, 1).measure_all()
        c3 = Circuit(2, 2).H(1).CX(0, 1).measure_all()
        c4 = Circuit(2, 2).Rx(0.5, 0).CX(0, 1).CX(1, 0).measure_all()
        cs = [c1, c2, c3, c4]
        n_shots = [10, 12, 10, 13]
        cs = backend.get_compiled_circuits(cs)
        handles = backend.process_circuits(cs, n_shots=n_shots)

        from pytket.extensions.qiskit.backends.ibm import _DEBUG_HANDLE_PREFIX

        assert all(
            cast(str, hand[0]) == _DEBUG_HANDLE_PREFIX + suffix
            for hand, suffix in zip(
                handles,
                [f"{(10, 0)}", f"{(12, 1)}", f"{(10, 0)}", f"{(13, 2)}"],
            )
        )
    finally:
        # ensure shared backend is reset for other tests
        backend._MACHINE_DEBUG = False


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_nshots(
    brisbane_emulator_backend: IBMQEmulatorBackend,
) -> None:
    for b in [AerBackend(), brisbane_emulator_backend]:
        circuit = Circuit(1).X(0)
        circuit.measure_all()
        n_shots = [1, 2, 3]
        circ_comp = b.get_compiled_circuit(circuit)
        results = b.get_results(b.process_circuits([circ_comp] * 3, n_shots=n_shots))
        assert [sum(r.get_counts().values()) for r in results] == n_shots


def test_pauli_statevector() -> None:
    c = Circuit(2)
    c.Rz(0.5, 0)
    Transform.OptimisePostRouting().apply(c)
    b = AerStateBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    assert get_pauli_expectation_value(c, zi, b) == 1
    c.X(0)
    assert get_pauli_expectation_value(c, zi, b) == -1


def test_pauli_sim() -> None:
    c = Circuit(2, 2)
    c.Rz(0.5, 0)
    Transform.OptimisePostRouting().apply(c)
    b = AerBackend()
    zi = QubitPauliString(Qubit(0), Pauli.Z)
    energy = get_pauli_expectation_value(c, zi, b, 8000)
    assert abs(energy - 1) < 0.001
    c.X(0)
    energy = get_pauli_expectation_value(c, zi, b, 8000)
    assert abs(energy + 1) < 0.001


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_default_pass(brisbane_backend: IBMQBackend) -> None:
    b = brisbane_backend
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c.measure_all()
        comp_pass.apply(c)
        for pred in b.required_predicates:
            assert pred.verify(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_aer_default_pass(brisbane_backend: IBMQBackend) -> None:
    noise_model = NoiseModel.from_backend(brisbane_backend._backend)
    for nm in [None, noise_model]:
        b = AerBackend(nm)
        for ol in range(3):
            comp_pass = b.default_compilation_pass(ol)
            c = Circuit(3, 3)
            c.H(0)
            c.CX(0, 1)
            c.CSWAP(1, 0, 2)
            c.ZZPhase(0.84, 2, 0)
            c.add_gate(OpType.TK1, [0.2, 0.3, 0.4], [0])
            comp_pass.apply(c)
            c.measure_all()
            for pred in b.required_predicates:
                assert pred.verify(c)


def test_routing_measurements() -> None:
    qc = get_test_circuit(True)
    physical_c = qiskit_to_tk(qc)
    sim = AerBackend()
    original_results = sim.run_circuit(physical_c, n_shots=10, seed=4).get_shots()
    coupling = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
    arc = Architecture(coupling)
    mm = MappingManager(arc)
    mm.route_circuit(physical_c, [LexiLabellingMethod(), LexiRouteRoutingMethod()])
    Transform.DecomposeSWAPtoCX(arc).apply(physical_c)
    Transform.DecomposeCXDirected(arc).apply(physical_c)
    Transform.OptimisePostRouting().apply(physical_c)
    assert (
        sim.run_circuit(physical_c, n_shots=10).get_shots() == original_results
    ).all()


def test_routing_no_cx() -> None:
    circ = Circuit(2, 2)
    circ.H(1)
    circ.Rx(0.2, 0)
    circ.measure_all()
    coupling = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
    arc = Architecture(coupling)
    mm = MappingManager(arc)
    mm.route_circuit(circ, [LexiRouteRoutingMethod()])
    assert len(circ.get_commands()) == 4


def test_counts() -> None:
    qc = get_test_circuit(True)
    circ = qiskit_to_tk(qc)
    sim = AerBackend()
    counts = sim.run_circuit(circ, n_shots=10, seed=4).get_counts()
    assert counts == {(1, 0, 1, 1, 0): 10}


def test_ilo() -> None:
    b = AerBackend()
    bs = AerStateBackend()
    bu = AerUnitaryBackend()
    c = Circuit(2)
    c.X(1)
    res_s = bs.run_circuit(c)
    res_u = bu.run_circuit(c)
    assert np.allclose(res_s.get_state(), np.asarray([0, 1, 0, 0]))
    assert np.allclose(res_s.get_state(basis=BasisOrder.dlo), np.asarray([0, 0, 1, 0]))
    assert np.allclose(
        res_u.get_unitary(),
        np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
    )
    assert np.allclose(
        res_u.get_unitary(basis=BasisOrder.dlo),
        np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    )
    c.measure_all()
    res = b.run_circuit(c, n_shots=2)
    assert (res.get_shots() == np.asarray([[0, 1], [0, 1]])).all()
    assert (res.get_shots(basis=BasisOrder.dlo) == np.asarray([[1, 0], [1, 0]])).all()
    assert res.get_counts() == {(0, 1): 2}
    assert res.get_counts(basis=BasisOrder.dlo) == {(1, 0): 2}


def test_ubox() -> None:
    # https://github.com/CQCL/pytket-extensions/issues/342
    u = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )
    ubox = Unitary2qBox(u)
    c = Circuit(2)
    c.add_unitary2qbox(ubox, 0, 1)
    b = AerUnitaryBackend()
    h = b.process_circuit(c)
    r = b.get_result(h)
    u1 = r.get_unitary()
    assert np.allclose(u, u1)


def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    b = AerStateBackend()
    c = Circuit(4)
    c.X(0)
    c.CX(0, 1)
    c.CX(1, 0)
    c.CX(1, 3)
    c.CX(3, 1)
    c.X(2)
    cu = CompilationUnit(c)
    CliffordSimp(True).apply(cu)
    c1 = cu.circuit
    assert c1.n_gates_of_type(OpType.CX) == 2

    c, c1 = b.get_compiled_circuits([c, c1])

    handles = b.process_circuits([c, c1])
    res_c = b.run_circuit(c)
    res_c1 = b.run_circuit(c1)
    s_ilo = res_c1.get_state(basis=BasisOrder.ilo)
    correct_ilo = res_c.get_state(basis=BasisOrder.ilo)

    assert np.allclose(s_ilo, correct_ilo)
    s_dlo = res_c1.get_state(basis=BasisOrder.dlo)
    correct_dlo = res_c.get_state(basis=BasisOrder.dlo)
    assert np.allclose(s_dlo, correct_dlo)

    qbs = c.qubits
    for result in b.get_results(handles):
        assert (
            result.get_state([qbs[1], qbs[2], qbs[3], qbs[0]]).real.tolist().index(1.0)  # type: ignore
            == 6
        )
        assert (
            result.get_state([qbs[2], qbs[1], qbs[0], qbs[3]]).real.tolist().index(1.0)  # type: ignore
            == 9
        )
        assert (
            result.get_state([qbs[2], qbs[3], qbs[0], qbs[1]]).real.tolist().index(1.0)  # type: ignore
            == 12
        )

    bu = AerUnitaryBackend()
    res_c = bu.run_circuit(c)
    res_c1 = bu.run_circuit(c1)
    u_ilo = res_c1.get_unitary(basis=BasisOrder.ilo)
    correct_ilo = res_c.get_unitary(basis=BasisOrder.ilo)
    assert np.allclose(u_ilo, correct_ilo)
    u_dlo = res_c1.get_unitary(basis=BasisOrder.dlo)
    correct_dlo = res_c.get_unitary(basis=BasisOrder.dlo)
    assert np.allclose(u_dlo, correct_dlo)


def test_pauli() -> None:
    for b in [AerBackend(), AerStateBackend()]:
        c = Circuit(2)
        c.Rz(0.5, 0)
        c = b.get_compiled_circuit(c)
        zi = QubitPauliString(Qubit(0), Pauli.Z)
        assert cmath.isclose(get_pauli_expectation_value(c, zi, b), 1)
        c.X(0)
        assert cmath.isclose(get_pauli_expectation_value(c, zi, b), -1)


def test_operator() -> None:
    for b in [AerBackend(), AerStateBackend()]:
        c = circuit_gen()
        zz = QubitPauliOperator(
            {QubitPauliString([Qubit(0), Qubit(1)], [Pauli.Z, Pauli.Z]): 1.0}
        )
        assert cmath.isclose(get_operator_expectation_value(c, zz, b), 1.0)
        c.X(0)
        assert cmath.isclose(get_operator_expectation_value(c, zz, b), -1.0)


# TKET-1432 this was either too slow or consumed too much memory when bugged
@pytest.mark.flaky(reruns=3, reruns_delay=10)
def test_expectation_bug() -> None:
    backend = AerStateBackend()
    # backend.compile_circuit(circuit)
    circuit = Circuit(16)
    with open("big_hamiltonian.json") as f:
        hamiltonian = QubitPauliOperator.from_list(json.load(f))
    exp = backend.get_operator_expectation_value(circuit, hamiltonian)
    assert np.isclose(exp, 1.4325392)


def test_aer_result_handle() -> None:
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()

    b = AerBackend()

    handles = b.process_circuits([c, c.copy()], n_shots=2)

    ids, indices, _, _ = zip(*(han for han in handles))

    assert all(isinstance(idval, str) for idval in ids)
    assert indices == (0, 1)

    assert len(b.get_result(handles[0]).get_shots()) == 2

    with pytest.raises(ResultHandleTypeError) as errorinfo:
        _ = b.get_result(ResultHandle("43"))
    assert "ResultHandle('43',) does not match expected identifier types" in str(
        errorinfo.value
    )

    wronghandle = ResultHandle("asdf", 3, 0, "jsonstr")

    with pytest.raises(CircuitNotRunError) as errorinfoCirc:
        _ = b.get_result(wronghandle)
    assert (
        f"Circuit corresponding to {wronghandle!r} "
        + "has not been run by this backend instance."
        in str(errorinfoCirc.value)
    )


def test_aerstate_result_handle() -> None:
    c = circuit_gen()
    b1 = AerStateBackend()
    h1 = b1.process_circuits([c])[0]
    state = b1.get_result(h1).get_state()
    status = b1.circuit_status(h1)
    assert status == CircuitStatus(StatusEnum.COMPLETED, "job has successfully run")
    assert np.allclose(state, [np.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
    b2 = AerUnitaryBackend()
    unitary = b2.run_circuit(c).get_unitary()
    assert np.allclose(
        unitary,
        np.sqrt(0.5)
        * np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]),
    )


def test_cache() -> None:
    b = AerBackend()
    c = circuit_gen()
    c = b.get_compiled_circuit(c)
    h = b.process_circuits([c], 2)[0]
    b.get_result(h).get_shots()
    assert h in b._cache
    b.pop_result(h)
    assert h not in b._cache
    assert not b._cache

    b.run_circuit(c, n_shots=2).get_counts()
    b.run_circuit(c.copy(), n_shots=2).get_counts()
    b.empty_cache()
    assert not b._cache


def test_mixed_circuit() -> None:
    c = Circuit()
    qr = c.add_q_register("q", 2)
    ar = c.add_c_register("a", 1)
    br = c.add_c_register("b", 1)
    c.H(qr[0])
    c.Measure(qr[0], ar[0])
    c.X(qr[1], condition=reg_eq(ar, 0))
    c.Measure(qr[1], br[0])
    backend = AerBackend()
    c = backend.get_compiled_circuit(c)
    counts = backend.run_circuit(c, n_shots=1024).get_counts()
    for key in counts:
        assert key in {(0, 1), (1, 0)}


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_aer_placed_expectation(brisbane_backend: IBMQBackend) -> None:
    # bug TKET-695
    n_qbs = 3
    c = Circuit(n_qbs, n_qbs)
    c.X(0)
    c.CX(0, 2)
    c.CX(1, 2)
    c.H(1)
    # c.measure_all()
    b = AerBackend()
    operator = QubitPauliOperator(
        {
            QubitPauliString(Qubit(0), Pauli.Z): 1.0,
            QubitPauliString(Qubit(1), Pauli.X): 0.5,
        }
    )
    assert b.get_operator_expectation_value(c, operator) == (-0.5 + 0j)

    noise_model = NoiseModel.from_backend(brisbane_backend._backend)

    noise_b = AerBackend(noise_model)

    with pytest.raises(RuntimeError) as errorinfo:
        noise_b.get_operator_expectation_value(c, operator)
        assert "not supported with noise model" in str(errorinfo.value)

    c.rename_units({Qubit(1): Qubit("node", 1)})
    with pytest.raises(ValueError) as errorinfoCirc:
        b.get_operator_expectation_value(c, operator)
        assert "default register Qubits" in str(errorinfoCirc.value)


def test_operator_expectation_value() -> None:
    c = Circuit(2).X(0).V(0).V(1).S(0).S(1).H(0).H(1).S(0).S(1)
    op = QubitPauliOperator(
        {
            QubitPauliString([], []): 0.5,
            QubitPauliString([Qubit(0)], [Pauli.Z]): -0.5,
        }
    )
    b = AerBackend()
    c1 = b.get_compiled_circuit(c)
    e = AerBackend().get_operator_expectation_value(c1, op)
    assert np.isclose(e, 1.0)


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_emulator(
    brisbane_emulator_backend: IBMQEmulatorBackend,
) -> None:
    assert brisbane_emulator_backend._noise_model is not None
    b_ibm = brisbane_emulator_backend._ibmq
    b_aer = AerBackend()
    for ol in range(3):
        comp_pass = brisbane_emulator_backend.default_compilation_pass(ol)
        c = Circuit(3, 3)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c_cop = c.copy()
        comp_pass.apply(c_cop)
        c.measure_all()
        for bac in (brisbane_emulator_backend, b_ibm):
            assert all(pred.verify(c_cop) for pred in bac.required_predicates)

        c_cop_2 = c.copy()
        c_cop_2 = b_aer.get_compiled_circuit(c_cop_2, ol)
        if ol == 0:
            assert not all(
                pred.verify(c_cop_2)
                for pred in brisbane_emulator_backend.required_predicates
            )

    circ = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    copy_circ = circ.copy()
    brisbane_emulator_backend.rebase_pass().apply(copy_circ)
    assert brisbane_emulator_backend.required_predicates[1].verify(copy_circ)
    circ = brisbane_emulator_backend.get_compiled_circuit(circ)
    b_noi = AerBackend(noise_model=brisbane_emulator_backend._noise_model)
    emu_counts = brisbane_emulator_backend.run_circuit(
        circ, n_shots=10, seed=10
    ).get_counts()
    aer_counts = b_noi.run_circuit(circ, n_shots=10, seed=10).get_counts()
    # Even with the same seed, the results may differ.
    assert sum(emu_counts.values()) == sum(aer_counts.values())


@given(
    n_shots=strategies.integers(min_value=1, max_value=10),
    n_bits=strategies.integers(min_value=0, max_value=10),
)
def test_shots_bits_edgecases(n_shots: int, n_bits: int) -> None:
    c = Circuit(n_bits, n_bits)
    c.measure_all()
    aer_backend = AerBackend()

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = aer_backend.process_circuit(c, n_shots)
    res = aer_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = aer_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


def test_simulation_method() -> None:
    state_backends = [AerBackend(), AerBackend(simulation_method="statevector")]
    stabilizer_backend = AerBackend(simulation_method="stabilizer")

    clifford_circ = Circuit(2).H(0).CX(0, 1).measure_all()
    clifford_T_circ = Circuit(2).H(0).T(1).CX(0, 1).measure_all()

    for b in state_backends + [stabilizer_backend]:
        counts = b.run_circuit(clifford_circ, n_shots=4).get_counts()
        assert sum(val for _, val in counts.items()) == 4

    for b in state_backends:
        counts = b.run_circuit(clifford_T_circ, n_shots=4).get_counts()
        assert sum(val for _, val in counts.items()) == 4

    with pytest.raises(CircuitNotValidError) as warninfo:
        # check for the error thrown when non-clifford circuit used with
        # stabilizer backend
        stabilizer_backend.run_circuit(clifford_T_circ, n_shots=4).get_counts()
        assert (
            "Circuit with index 0 in submitted does not satisfy GateSetPredicate"
            in str(warninfo.value)
        )


def test_aer_expanded_gates() -> None:
    c = Circuit(3).CX(0, 1)
    c.add_gate(OpType.ZZPhase, 0.1, [0, 1])
    c.add_gate(OpType.CY, [0, 1])
    c.add_gate(OpType.CCX, [0, 1, 2])

    backend = AerBackend()
    assert backend.valid_circuit(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_mid_measure(brisbane_backend: IBMQBackend) -> None:
    c = Circuit(3, 3).H(1).CX(1, 2).Measure(0, 0).Measure(1, 1)
    c.add_barrier([0, 1, 2])

    c.CX(1, 0).H(0).Measure(2, 2)

    b = brisbane_backend
    ps = b.default_compilation_pass(0)
    ps.apply(c)
    assert not NoMidMeasurePredicate().verify(c)
    assert b.valid_circuit(c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_conditional(brisbane_backend: IBMQBackend) -> None:
    c = Circuit(3, 2).H(1).CX(1, 2).Measure(0, 0).Measure(1, 1)
    c.add_barrier([0, 1, 2])
    ar = c.add_c_register("a", 1)
    c.CX(1, 0).H(0).X(2, condition=reg_eq(ar, 0)).Measure(Qubit(2), ar[0])

    b = brisbane_backend
    compiled = b.get_compiled_circuit(c)
    assert b.backend_info.supports_fast_feedforward
    assert not NoMidMeasurePredicate().verify(compiled)
    assert b.valid_circuit(compiled)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_compile_x(brisbane_backend: IBMQBackend) -> None:
    # TKET-1028
    b = brisbane_backend
    c = Circuit(1).X(0)
    for ol in range(3):
        c1 = c.copy()
        c1 = b.get_compiled_circuit(c1, optimisation_level=ol)
        assert c1.n_gates == 1


def lift_perm(p: dict[int, int]) -> np.ndarray:
    """
    Given a permutation of {0,1,...,n-1} return the 2^n by 2^n permuation matrix
    representing the permutation of qubits (big-endian convention).
    """
    n = len(p)
    pm = np.zeros((1 << n, 1 << n), dtype=complex)
    for i in range(1 << n):
        j = 0
        mask = 1 << n
        for q in range(n):
            mask >>= 1
            if (i & mask) != 0:
                j |= 1 << (n - 1 - p[q])
        pm[j][i] = 1
    return pm


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_compilation_correctness(brisbane_backend: IBMQBackend) -> None:
    # of routing
    c = Circuit(7)
    c.H(0).H(1).H(2)
    c.CX(0, 1).CX(1, 2)
    c.Rx(0.25, 1).Ry(0.75, 1).Rz(0.5, 2)
    c.CCX(2, 1, 0)
    c.CY(1, 0).CY(2, 1)
    c.H(0).H(1).H(2)
    c.Rz(0.125, 0)
    c.X(1)
    c.Rz(0.125, 2).X(2).Rz(0.25, 2)
    c.SX(3).Rz(0.125, 3).SX(3)
    c.CX(0, 3).CX(0, 4)
    c.remove_blank_wires()
    FlattenRelabelRegistersPass().apply(c)
    c_pred = ConnectivityPredicate(
        cast(Architecture, brisbane_backend.backend_info.architecture)
    )
    for ol in range(3):
        p = brisbane_backend.default_compilation_pass(optimisation_level=ol)
        cu = CompilationUnit(c)
        p.apply(cu)
        assert c_pred.verify(cu.circuit)


# pytket-extensions issue #69
def test_symbolic_rebase() -> None:
    circ = QuantumCircuit(2)
    circ.rx(Parameter("a"), 0)
    circ.ry(Parameter("b"), 1)
    circ.cx(0, 1)

    pytket_circ = qiskit_to_tk(circ)

    # rebase pass could not handle symbolic parameters originally and would fail here:
    AerBackend().rebase_pass().apply(pytket_circ)

    assert len(pytket_circ.free_symbols()) == 2


def _tk1_to_rotations(a: float, b: float, c: float) -> Circuit:
    """Translate tk1 to a RzRxRz so AerUnitaryBackend can simulate"""
    circ = Circuit(1)
    circ.Rz(c, 0).Rx(b, 0).Rz(a, 0)
    return circ


def _verify_single_q_rebase(
    backend: AerUnitaryBackend, a: float, b: float, c: float
) -> bool:
    """Compare the unitary of a tk1 gate to the unitary of the translated circuit"""
    rotation_circ = _tk1_to_rotations(a, b, c)
    u_before = backend.run_circuit(rotation_circ).get_unitary()
    circ = Circuit(1)
    circ.add_gate(OpType.TK1, [a, b, c], [0])
    backend.rebase_pass().apply(circ)
    u_after = backend.run_circuit(circ).get_unitary()
    return np.allclose(u_before, u_after)


def test_rebase_phase() -> None:
    backend = AerUnitaryBackend()
    for a in [0.6, 0, 1, 2, 3]:
        for b in [0.7, 0, 0.5, 1, 1.5]:
            for c in [0.8, 0, 1, 2, 3]:
                assert _verify_single_q_rebase(backend, a, b, c)
                assert _verify_single_q_rebase(backend, -a, -b, -c)
                assert _verify_single_q_rebase(backend, 2 * a, 3 * b, 4 * c)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess() -> None:
    b = IBMQBackend(
        "ibm_brisbane",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.X(0).X(1).measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[3])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    b.cancel(h)


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_postprocess_emu(brisbane_emulator_backend: IBMQEmulatorBackend) -> None:
    assert brisbane_emulator_backend.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.X(0).X(1).measure_all()
    c = brisbane_emulator_backend.get_compiled_circuit(c)
    h = brisbane_emulator_backend.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast(str, h[3])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = brisbane_emulator_backend.get_result(h)
    counts = r.get_counts()
    assert sum(counts.values()) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_available_devices(qiskit_runtime_service: QiskitRuntimeService) -> None:
    backend_info_list = IBMQBackend.available_devices(instance="ibm-q/open/main")
    assert len(backend_info_list) > 0

    # Check consistency with pytket-qiskit and qiskit runtime service
    assert len(backend_info_list) == len(qiskit_runtime_service.backends())

    backend_info_list = IBMQBackend.available_devices(service=qiskit_runtime_service)
    assert len(backend_info_list) > 0
    backend_info_list = IBMQBackend.available_devices()
    assert len(backend_info_list) > 0


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_backendinfo_serialization1(
    brisbane_emulator_backend: IBMQEmulatorBackend,
) -> None:
    # https://github.com/CQCL/tket/issues/192
    backend_info_json = brisbane_emulator_backend.backend_info.to_dict()
    s = json.dumps(backend_info_json)
    backend_info_json1 = json.loads(s)
    assert backend_info_json == backend_info_json1


def test_backendinfo_serialization2() -> None:
    # https://github.com/CQCL/tket/issues/192
    my_noise_model = NoiseModel()
    my_noise_model.add_readout_error(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        [0],
    )
    my_noise_model.add_readout_error(
        [
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        [1],
    )
    my_noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cx"], [0, 1])
    my_noise_model.add_quantum_error(depolarizing_error(0.5, 1), ["u3"], [0])
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Z", 0.65)]), ["u2"], [0]
    )
    my_noise_model.add_quantum_error(
        pauli_error([("X", 0.35), ("Y", 0.65)]), ["u1"], [0]
    )
    backend = AerBackend(my_noise_model)
    backend_info_json = backend.backend_info.to_dict()
    s = json.dumps(backend_info_json)
    backend_info_json1 = json.loads(s)
    assert backend_info_json == backend_info_json1


def test_sim_qubit_order() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/54
    backend = AerStateBackend()
    circ = Circuit()
    circ.add_q_register("a", 1)
    circ.add_q_register("b", 1)
    circ.X(Qubit("a", 0))
    s = backend.run_circuit(circ).get_state()
    assert np.isclose(abs(s[2]), 1.0)


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_required_predicates(
    brisbane_emulator_backend: IBMQEmulatorBackend,
) -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/93
    circ = Circuit(8)  # 8 qubit circuit in IBMQ gateset
    circ.X(0).CX(0, 1).CX(0, 2).CX(0, 3).CX(0, 4).CX(0, 5).CX(0, 6).CX(
        0, 7
    ).measure_all()
    with pytest.raises(CircuitNotValidError) as errorinfo:
        brisbane_emulator_backend.run_circuit(circ, n_shots=100)
        assert (
            "pytket.backends.backend_exceptions.CircuitNotValidError:"
            + "Circuit with index 0 in submitted does"
            + "not satisfy MaxNQubitsPredicate(5)"
            in str(errorinfo)
        )


@pytest.mark.flaky(reruns=3, reruns_delay=10)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ecr_gate_compilation(ibm_brisbane_backend: IBMQBackend) -> None:
    assert ibm_brisbane_backend.backend_info.gate_set >= {
        OpType.X,
        OpType.SX,
        OpType.Rz,
        OpType.ECR,
    }
    # circuit for an un-routed GHZ state
    circ = (
        Circuit(7)
        .H(0)
        .CX(0, 1)
        .CX(0, 2)
        .CX(0, 3)
        .CX(0, 4)
        .CX(0, 5)
        .CX(0, 6)
        .measure_all()
    )
    for optimisation_level in range(3):
        compiled_circ = ibm_brisbane_backend.get_compiled_circuit(
            circ, optimisation_level
        )
        assert ibm_brisbane_backend.valid_circuit(compiled_circ)


def test_crosstalk_noise_model() -> None:
    circ = Circuit(3).X(0).CX(0, 1).CX(1, 2).measure_all()
    zz_crosstalks = {
        (Qubit(0), Qubit(1)): 0.0003,
        (Qubit(0), Qubit(2)): 0.0013,
        (Qubit(1), Qubit(2)): 0.002,
    }
    single_q_phase_errors = {
        Qubit(0): 0.00498,
        Qubit(1): 0.0021,
        Qubit(2): 0.0021,
    }
    two_q_induced_phase_errors = {
        (Qubit(0), Qubit(1)): (Qubit(2), 0.0033),
        (Qubit(1), Qubit(0)): (Qubit(2), 0.0033),
        (Qubit(0), Qubit(2)): (Qubit(1), 0.0033),
        (Qubit(2), Qubit(0)): (Qubit(1), 0.0033),
        (Qubit(1), Qubit(2)): (Qubit(0), 0.0033),
        (Qubit(2), Qubit(1)): (Qubit(0), 0.0033),
    }
    non_markovian_noise = [
        (Qubit(0), 0.007, 0.007),
        (Qubit(1), 0.004, 0.006),
        (Qubit(2), 0.005, 0.006),
    ]
    phase_damping_error = {
        Qubit(0): 0.05,
        Qubit(1): 0.05,
        Qubit(2): 0.05,
    }
    amplitude_damping_error = {
        Qubit(0): 0.05,
        Qubit(1): 0.05,
        Qubit(2): 0.05,
    }

    N = 10
    gate_times = {}
    for q in circ.qubits:
        gate_times[(OpType.X, tuple([q]))] = 0.1
    for q0 in circ.qubits:
        for q1 in circ.qubits:
            if q0 != q1:
                gate_times[(OpType.CX, (q0, q1))] = 0.5

    ctparams = CrosstalkParams(
        zz_crosstalks,
        single_q_phase_errors,
        two_q_induced_phase_errors,
        non_markovian_noise,
        False,
        N,
        gate_times,
        phase_damping_error,
        amplitude_damping_error,
    )
    # test manual construction
    noisy_circ_builder = NoisyCircuitBuilder(circ, ctparams)
    noisy_circ_builder.build()
    slices = noisy_circ_builder.get_slices()
    n_fractions = 0
    for s in slices:
        for inst in s:
            if isinstance(inst, FractionalUnitary):
                n_fractions = n_fractions + 1
    assert n_fractions == 11

    # test processing circuit
    aer = AerBackend(crosstalk_params=ctparams)
    compiled_circ = aer.get_compiled_circuit(circ, optimisation_level=0)
    h = aer.process_circuit(compiled_circ, n_shots=100)
    res = aer.get_result(h)
    res.get_counts()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ecr(ibm_brisbane_backend: IBMQBackend) -> None:
    ghz5 = Circuit(5)
    ghz5.H(0).CX(0, 1).CX(0, 2).CX(0, 3).CX(0, 4)
    ghz5.measure_all()
    ibm_ghz5 = ibm_brisbane_backend.get_compiled_circuit(ghz5)

    compiled_ghz5 = ibm_brisbane_backend.get_compiled_circuit(ibm_ghz5)

    ibm_brisbane_backend.valid_circuit(compiled_ghz5)

    handle = ibm_brisbane_backend.process_circuit(compiled_ghz5, n_shots=1000)
    ibm_brisbane_backend.cancel(handle)


# helper function for testing
def _get_qiskit_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Given a QuantumCircuit, use aer_simulator_statevector to compute its
    statevector, return the vector with its endianness adjusted"""
    back = Aer.get_backend("aer_simulator_statevector")
    qc.save_state()
    job = back.run(qc)
    return np.array(job.result().data()["statevector"].reverse_qargs().data)


# The three tests below and helper function above relate to this issue.
# https://github.com/CQCL/pytket-qiskit/issues/99
def test_statevector_simulator_gateset_deterministic() -> None:
    sv_backend = AerStateBackend()
    sv_supported_gates = sv_backend.backend_info.gate_set
    assert OpType.Reset and OpType.Measure in sv_supported_gates
    assert OpType.Conditional in sv_supported_gates
    # This circuit is deterministic in the sense that it prepares a
    # non-mixed state starting from the "all-0" state.
    # In general circuits with measures/resets won't be deterministic
    circ = Circuit(3, 1)
    circ.CCX(*range(3))
    circ.U1(1 / 4, 2)
    circ.H(2)
    circ.Measure(2, 0)
    circ.CZ(0, 1, condition_bits=[0], condition_value=1)
    circ.Reset(2)
    compiled_circ = sv_backend.get_compiled_circuit(circ)
    assert sv_backend.valid_circuit(compiled_circ)
    tket_statevector = sv_backend.run_circuit(compiled_circ).get_state()
    qc = tk_to_qiskit(compiled_circ)
    qiskit_statevector = _get_qiskit_statevector(qc)
    assert compare_statevectors(tket_statevector, qiskit_statevector)


def test_statevector_non_deterministic() -> None:
    circ = Circuit(2, 1)
    circ.H(0).H(1)
    circ.Measure(0, 0)
    circ.CX(1, 0, condition_bits=[0], condition_value=1)
    sv_backend = AerStateBackend()
    statevector = sv_backend.run_circuit(circ).get_state()
    # Possible results: 1/sqrt(2)(|00>+|01>) or 1/sqrt(2)(|01>+|10>)
    result1 = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    result2 = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])
    assert compare_statevectors(statevector, result1) or compare_statevectors(
        statevector, result2
    )


def test_unitary_backend_transpiles() -> None:
    """regression test for https://github.com/CQCL/pytket-qiskit/issues/142"""
    backend = AerUnitaryBackend()
    n_ancillas = 5  # using n_ancillas <=4 doees not raise an error
    n_spins = 1
    circ = Circuit(n_ancillas + n_spins)
    trgt = Circuit(n_spins)
    trgt.X(0)

    circ.add_qcontrolbox(
        QControlBox(CircBox(trgt), n_ancillas), list(range(n_ancillas + n_spins))
    )

    compiled_circ = backend.get_compiled_circuit(circ, optimisation_level=0)
    # using optimisation_level >= 1 does not raise an error
    r = backend.run_circuit(compiled_circ)
    u = r.get_unitary()
    # check that the lower-right 2x2 submatrix of the unitary is the matrix of
    # the X gate.
    assert np.isclose(u[62:64, 62:64], np.asarray(([0.0, 1.0], [1.0, 0.0]))).all()


def test_barriers_in_aer_simulators() -> None:
    """Test for barrier support in aer simulators
    https://github.com/CQCL/pytket-qiskit/issues/186"""

    circ = Circuit(2).H(0).CX(0, 1).add_barrier([0, 1])

    state_backend = AerStateBackend()
    shots_backend = AerBackend()
    unitary_backend = AerUnitaryBackend()

    test_state = circ.get_statevector()
    test_unitary = circ.get_unitary()

    backends = (state_backend, unitary_backend, shots_backend)

    for backend in backends:
        assert OpType.Barrier in backend.backend_info.gate_set
        assert backend.valid_circuit(circ)

    state_result = state_backend.run_circuit(circ).get_state()
    unitary_result = unitary_backend.run_circuit(circ).get_unitary()

    assert compare_statevectors(test_state, state_result)
    assert compare_unitaries(test_unitary, unitary_result)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_ibmq_local_emulator(
    brisbane_emulator_backend: IBMQEmulatorBackend,
) -> None:
    b = brisbane_emulator_backend
    circ = Circuit(2).H(0).CX(0, 1).measure_all()
    circ1 = b.get_compiled_circuit(circ)
    h = b.process_circuit(circ1, n_shots=100)
    r = b.get_result(h)
    counts = r.get_counts()
    # Most results should be (0,0) or (1,1):
    assert sum(c0 != c1 for c0, c1 in counts) < 25


# https://github.com/CQCL/pytket-qiskit/issues/231
def test_noiseless_density_matrix_simulation() -> None:
    density_matrix_backend = AerDensityMatrixBackend()
    assert density_matrix_backend.supports_density_matrix is True

    assert isinstance(density_matrix_backend.backend_info.architecture, FullyConnected)

    circ1 = Circuit(3).X(0).X(1).CCX(0, 1, 2)

    output_state = np.array([0] * 7 + [1])

    result1 = density_matrix_backend.run_circuit(circ1)
    noiseless_dm1 = result1.get_density_matrix()

    assert noiseless_dm1.shape == (8, 8)
    assert np.allclose(noiseless_dm1, np.outer(output_state, output_state.conj()))
    # Check purity to verify that we have a pure state
    np.isclose(np.trace(noiseless_dm1**2).real, 1)

    # Example with resets and conditional gates
    # Prepares a state deterministically if input is a computational basis state
    circ2 = Circuit(3, 1)
    circ2.CCX(*range(3))
    circ2.U1(1 / 4, 2)
    circ2.H(2)
    circ2.Measure(2, 0)
    circ2.CZ(0, 1, condition_bits=[0], condition_value=1)
    circ2.Reset(2)

    result2 = density_matrix_backend.run_circuit(circ2)
    assert result1.get_density_matrix().shape == (8, 8)
    state_backend = AerStateBackend()
    statevector = state_backend.run_circuit(circ2).get_state()
    noiseless_dm2 = result2.get_density_matrix()
    assert np.allclose(noiseless_dm2, np.outer(statevector, statevector.conj()))
    # Check purity to verify that we have a pure state
    assert np.isclose(np.trace(noiseless_dm2**2).real, 1)


# https://github.com/CQCL/pytket-qiskit/issues/231
def test_noisy_density_matrix_simulation() -> None:
    # Test that __init__ works with a very simple noise model
    noise_model = NoiseModel()
    noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cz"], [0, 1])
    noise_model.add_quantum_error(depolarizing_error(0.6, 2), ["cz"], [1, 2])

    noisy_density_sim = AerDensityMatrixBackend(noise_model)
    assert isinstance(noisy_density_sim.backend_info.architecture, Architecture)
    assert len(noisy_density_sim.backend_info.architecture.nodes) == 3

    circ = Circuit(3)
    circ.X(0)
    circ.X(1)
    circ.SX(1)
    circ.CZ(0, 1)
    circ.CZ(1, 2)
    assert noisy_density_sim.valid_circuit(circ)

    result = noisy_density_sim.run_circuit(circ)
    noisy_dm = result.get_density_matrix()
    assert noisy_dm.shape == (8, 8)
    # Check purity to verify mixed state
    assert np.trace(noisy_dm**2).real < 0.99


def test_mc_gate_on_aer() -> None:
    """Test for cm gates support in aer simulators
    https://github.com/CQCL/pytket-qiskit/issues/368"""
    b = AerBackend()
    c = Circuit(3, 3)
    c.X(0).X(1)
    c.H(2)
    c.add_gate(OpType.CnZ, [0, 1, 2])
    c.H(2)
    c.measure_all()
    r = b.run_circuit(c, n_shots=10)
    assert r.get_counts() == Counter({(1, 1, 1): 10})


def test_optimisation_level_3_compilation() -> None:
    b = AerBackend()
    a = Architecture([(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7)])
    b._has_arch = True
    b._backend_info.architecture = a

    c = Circuit(6)
    for _ in range(6):
        for i in range(4):
            for j in range(i + 1, 4):
                c.CX(i, j)
                c.Rz(0.23, j)
                c.S(j)
            c.H(i)

    compiled_2 = b.get_compiled_circuit(c, 2)
    compiled_3 = b.get_compiled_circuit(c, 3)
    compiled_3_timeout = b.get_compiled_circuit(c, 3, timeout=0)

    assert compiled_2.n_2qb_gates() == 78
    assert compiled_2.n_gates == 205
    assert compiled_2.depth() == 147
    assert compiled_3.n_2qb_gates() == 61
    assert compiled_3.n_gates == 164
    assert compiled_3.depth() == 114
    assert compiled_3_timeout.n_2qb_gates() == 69
    assert compiled_3_timeout.n_gates == 171
    assert compiled_3_timeout.depth() == 125


def test_optimisation_level_3_serialisation() -> None:
    b = AerBackend()
    a = Architecture([(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7)])
    b._has_arch = True
    b._backend_info.architecture = a

    p_dict = b.default_compilation_pass(3).to_dict()
    passlist = SequencePass.from_dict(
        p_dict,
        {
            "lightsabrepass": _gen_lightsabre_transformation(
                b._backend_info.architecture
            )
        },
    )
    assert p_dict == passlist.to_dict()


def test_process_circuits_n_qubits() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/420
    circs = [Circuit(1).X(0).measure_all(), Circuit(2).X(0).measure_all()]
    b = AerBackend()
    hs = b.process_circuits(circs, n_shots=10)
    rs = b.get_results(hs)
    assert rs[0].get_counts() == Counter({(1,): 10})
    assert rs[1].get_counts() == Counter({(1, 0): 10})
