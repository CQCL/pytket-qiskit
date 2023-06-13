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
import pytest

from qiskit_ibm_provider import IBMProvider  # type: ignore
from pytket.extensions.qiskit import IBMQBackend, IBMQEmulatorBackend


@pytest.fixture(autouse=True, scope="session")
def setup_qiskit_account() -> None:
    if os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None:
        # The remote tests require an active IBMQ account
        # We check if an IBMQ account is already saved, otherwise we try
        # to enable one using the token in the env variable:
        # PYTKET_REMOTE_QISKIT_TOKEN
        # Note: The IBMQ account will only be enabled for the current session
        try:
            IBMProvider()
        except:
            token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
            if token:
                IBMProvider.save_account(token, overwrite=True)
                IBMProvider()


@pytest.fixture(scope="module")
def manila_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibmq_manila",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def lima_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibmq_lima",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def qasm_simulator_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibmq_qasm_simulator",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def simulator_stabilizer_backend() -> IBMQBackend:
    return IBMQBackend(
        "simulator_stabilizer",
        instance="ibm-q/open/main",
        monitor=False,
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def manila_emulator_backend() -> IBMQEmulatorBackend:
    return IBMQEmulatorBackend(
        "ibmq_manila",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def belem_emulator_backend() -> IBMQEmulatorBackend:
    return IBMQEmulatorBackend(
        "ibmq_belem",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def ibm_provider() -> IBMProvider:
    token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")

    try:
        return IBMProvider(instance="ibm-q/open/main")
    except:
        token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
        return IBMProvider(token=token, instance="ibm-q/open/main", overwrite=True)
