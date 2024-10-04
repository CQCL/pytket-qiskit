# Copyright 2020-2024 Quantinuum
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
from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

from pytket.extensions.qiskit import (
    IBMQBackend,
    IBMQEmulatorBackend,
)


@pytest.fixture(autouse=True, scope="session")
def setup_qiskit_account() -> None:
    # The remote tests require an active IBMQ account
    # We check if an IBMQ account is already saved, otherwise we try
    # to enable one using the token in the env variable:
    # PYTKET_REMOTE_QISKIT_TOKEN
    # Note: The IBMQ account will only be enabled for the current session
    if (
        os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None
        and not QiskitRuntimeService.saved_accounts()
    ):
        token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
        if token:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum", token=token, overwrite=True
            )


@pytest.fixture(scope="module")
def brisbane_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibm_brisbane",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def brisbane_emulator_backend() -> IBMQEmulatorBackend:
    return IBMQEmulatorBackend(
        "ibm_brisbane",
        instance="ibm-q/open/main",
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )


@pytest.fixture(scope="module")
def qiskit_runtime_service() -> QiskitRuntimeService:
    token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")

    try:
        return QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")
    except:  # noqa: E722
        token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
        return QiskitRuntimeService(
            channel="ibm_quantum", token=token, instance="ibm-q/open/main"
        )


@pytest.fixture(scope="module")
def ibm_brisbane_backend() -> IBMQBackend:
    return IBMQBackend(
        backend_name="ibm_brisbane",
        monitor=False,
        token=os.getenv("PYTKET_REMOTE_QISKIT_TOKEN"),
    )
