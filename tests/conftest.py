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

import pytest
from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

from pytket.extensions.qiskit import (
    IBMQBackend,
    IBMQEmulatorBackend,
)

INSTANCE = "crn:v1:bluemix:public:quantum-computing:eu-de:a/18f63f4565ef4a40851959792418cbf2:2a6bcfe2-0f5b-4c25-acd0-c13793935eb5::"


@pytest.fixture(autouse=True, scope="session")
def setup_qiskit_account() -> None:
    # The remote tests require an active IBMQ account
    # We check if an IBMQ account is already saved, otherwise we try
    # to enable one using the token in the env variable:
    # PYTKET_REMOTE_IBM_CLOUD_TOKEN
    # Note: The IBMQ account will only be enabled for the current session
    if (
        os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None
        and not QiskitRuntimeService.saved_accounts()
    ):
        token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")
        if token:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform", token=token, overwrite=True
            )


@pytest.fixture(scope="module")
def brussels_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibm_brussels",
        instance=INSTANCE,
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )


@pytest.fixture(scope="module")
def brussels_emulator_backend() -> IBMQEmulatorBackend:
    return IBMQEmulatorBackend(
        "ibm_brussels",
        instance=INSTANCE,
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )


@pytest.fixture(scope="module")
def qiskit_runtime_service() -> QiskitRuntimeService:
    token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")

    try:
        return QiskitRuntimeService(channel="ibm_quantum_platform", instance=INSTANCE)
    except:  # noqa: E722
        token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")
        return QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=INSTANCE
        )


@pytest.fixture(scope="module")
def ibm_brussels_backend() -> IBMQBackend:
    return IBMQBackend(
        backend_name="ibm_brussels",
        monitor=False,
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )
