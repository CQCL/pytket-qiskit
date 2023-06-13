# Copyright 2020-2023 Cambridge Quantum Computing
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

import os
from typing import Optional, Any

import numpy as np
import pytest

from qiskit import QuantumCircuit, execute  # type: ignore
from qiskit.opflow import CircuitStateFn, CircuitSampler  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit.utils import QuantumInstance  # type: ignore
from qiskit.algorithms import Grover, AmplificationProblem  # type: ignore
from qiskit.transpiler.exceptions import TranspilerError  # type: ignore
from qiskit.transpiler.passes import Unroller  # type: ignore
from qiskit_aer import Aer  # type: ignore

from qiskit_ibm_provider import IBMProvider # type: ignore 

from pytket.extensions.qiskit import (
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQEmulatorBackend,
)
from pytket.extensions.qiskit.tket_backend import TketBackend
from pytket.circuit import OpType  # type: ignore
from pytket.architecture import Architecture, FullyConnected  # type: ignore

from .mock_pytket_backend import MockShotBackend

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires IBMQ configuration)"


@pytest.fixture
def provider() -> Optional["IBMProvider"]:
    if skip_remote_tests:
        return None
    else:
        return IBMProvider(instance="ibm-q")


def circuit_gen(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    if measure:
        qc.measure_all()
    return qc


def test_samples() -> None:
    qc = circuit_gen(True)
    b = AerBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = execute(qc, tb, shots=100, memory=True)
        shots = job.result().get_memory()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in shots))
        counts = job.result().get_counts()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in counts.keys()))


def test_state() -> None:
    qc = circuit_gen()
    b = AerStateBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        assert QuantumInstance(tb).is_statevector
        job = execute(qc, tb)
        state = job.result().get_statevector()
        qb = Aer.get_backend("aer_simulator_statevector")
        qc1 = qc.copy()
        qc1.save_state()
        job2 = execute(qc1, qb)
        state2 = job2.result().get_statevector()
        assert np.allclose(state, state2)


def test_unitary() -> None:
    qc = circuit_gen()
    b = AerUnitaryBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = execute(qc, tb)
        u = job.result().get_unitary()
        qb = Aer.get_backend("aer_simulator_unitary")
        qc1 = qc.copy()
        qc1.save_unitary()
        job2 = execute(qc1, qb)
        u2 = job2.result().get_unitary()
        assert np.allclose(u, u2)


def test_cancel() -> None:
    b = AerBackend()
    tb = TketBackend(b)
    qc = circuit_gen()
    job = execute(qc, tb, shots=1024)
    job.cancel()
    assert job.status() in [JobStatus.CANCELLED, JobStatus.DONE]


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_qiskit_counts(belem_emulator_backend: IBMQEmulatorBackend) -> None:
    num_qubits = 2
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    circfn = CircuitStateFn(qc)

    s = CircuitSampler(TketBackend(belem_emulator_backend))

    res = s.sample_circuits([circfn])

    res_dictstatefn = res[id(circfn)][0]

    assert res_dictstatefn.num_qubits == num_qubits


def test_architectures() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/14
    arch_list = [None, Architecture([[0, 1], [1, 2]]), FullyConnected(3)]
    qc = circuit_gen(True)
    for arch in arch_list:
        # without architecture
        b = MockShotBackend(arch=arch)
        tb = TketBackend(b, b.default_compilation_pass())
        job = execute(qc, tb, shots=100, memory=True)
        shots = job.result().get_memory()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in shots))
        counts = job.result().get_counts()
        assert all(((r[0] == "1" and r[1] == r[2]) for r in counts.keys()))


def test_grover() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/15
    b = MockShotBackend()
    backend = TketBackend(b, b.default_compilation_pass())
    qinstance = QuantumInstance(backend)
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)

    def is_good_state(bitstr: Any) -> bool:
        return sum(map(int, bitstr)) == 2

    problem = AmplificationProblem(oracle=oracle, is_good_state=is_good_state)
    grover = Grover(quantum_instance=qinstance)
    result = grover.amplify(problem)
    assert result.top_measurement == "11"


def test_unsupported_gateset() -> None:
    # Working with gatesets that are unsupported by qiskit requires
    # providing QuantumInstance with a custom pass manager.
    b = MockShotBackend(gate_set={OpType.Rz, OpType.PhasedX, OpType.ZZMax})
    backend = TketBackend(b, b.default_compilation_pass())
    qinstance = QuantumInstance(backend)
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)

    def is_good_state(bitstr: Any) -> bool:
        return sum(map(int, bitstr)) == 2

    problem = AmplificationProblem(oracle=oracle, is_good_state=is_good_state)
    grover = Grover(quantum_instance=qinstance)
    # Qiskit will attempt to rebase a Grover op into the MockShotBackend gateset.
    # However, Rz, PhasedX and ZZMax gateset isn't supported by qiskit.
    # (tested with qiskit 0.39.1)
    with pytest.raises(TranspilerError) as e:
        result = grover.amplify(problem)
    err_msg = "Unable to map"
    assert err_msg in str(e.value)

    # By providing an Unroller pass, the QuantumInstance will rebase the Grover op into
    # the u3, CX gateset instead.
    unroll_pass = Unroller(["u3", "cx"])
    qinstance = QuantumInstance(backend, pass_manager=unroll_pass)
    grover = Grover(quantum_instance=qinstance)
    problem = AmplificationProblem(oracle=oracle, is_good_state=is_good_state)
    result = grover.amplify(problem)
    assert result.top_measurement == "11"
