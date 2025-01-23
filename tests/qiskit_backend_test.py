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

import os

import numpy as np
import pytest
from qiskit import QuantumCircuit  # type: ignore
from qiskit.primitives import BackendSamplerV2  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit_aer import Aer  # type: ignore

from pytket.architecture import Architecture, FullyConnected
from pytket.circuit import Circuit
from pytket.extensions.qiskit import (
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQEmulatorBackend,
)
from pytket.extensions.qiskit.tket_backend import TketBackend

from .mock_pytket_backend import MockShotBackend

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires IBM configuration)"


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
        job = tb.run(qc, shots=100, memory=True)
        shots = job.result().get_memory()
        assert all((r[0] == "1" and r[1] == r[2]) for r in shots)
        counts = job.result().get_counts()
        assert all((r[0] == "1" and r[1] == r[2]) for r in counts)


def test_maxnqubits() -> None:
    backend = AerBackend(n_qubits=1)
    with pytest.raises(Exception):
        backend.run_circuit(
            circuit=Circuit(2).CX(0, 1).measure_all(),
            n_shots=1,
        )


def test_state() -> None:
    qc = circuit_gen()
    b = AerStateBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = tb.run(qc)
        state = job.result().get_statevector()
        qb = Aer.get_backend("aer_simulator_statevector")
        qc1 = qc.copy()
        qc1.save_state()
        job2 = qb.run(qc1)
        state2 = job2.result().get_statevector()
        assert np.allclose(state, state2)


def test_unitary() -> None:
    qc = circuit_gen()
    b = AerUnitaryBackend()
    for comp in (None, b.default_compilation_pass()):
        tb = TketBackend(b, comp)
        job = tb.run(qc)
        u = job.result().get_unitary()
        qb = Aer.get_backend("aer_simulator_unitary")
        qc1 = qc.copy()
        qc1.save_unitary()
        job2 = qb.run(qc1)
        u2 = job2.result().get_unitary()
        assert np.allclose(u, u2)


def test_cancel() -> None:
    b = AerBackend()
    tb = TketBackend(b)
    qc = circuit_gen()
    job = tb.run(qc, shots=1024)
    job.cancel()
    assert job.status() in [JobStatus.CANCELLED, JobStatus.DONE]


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_qiskit_counts(brisbane_emulator_backend: IBMQEmulatorBackend) -> None:
    num_qubits = 2
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    s = BackendSamplerV2(
        backend=TketBackend(
            brisbane_emulator_backend,
            comp_pass=brisbane_emulator_backend.default_compilation_pass(
                optimisation_level=0
            ),
        )
    )

    job = s.run([qc], shots=10)
    res = job.result()

    assert res[0].metadata["shots"] == 10
    assert all(n in range(4) for n in res[0].data["meas"].get_int_counts())


def test_architectures() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/14
    arch_list = [None, Architecture([(0, 1), (1, 2)]), FullyConnected(3)]
    qc = circuit_gen(True)
    for arch in arch_list:
        # without architecture
        b = MockShotBackend(arch=arch)
        tb = TketBackend(b, b.default_compilation_pass())
        job = tb.run(qc, shots=100, memory=True)
        shots = job.result().get_memory()
        assert all((r[0] == "1" and r[1] == r[2]) for r in shots)
        counts = job.result().get_counts()
        assert all((r[0] == "1" and r[1] == r[2]) for r in counts)
