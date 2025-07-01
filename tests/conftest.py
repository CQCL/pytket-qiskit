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


@pytest.fixture(autouse=True, scope="session")
def setup_qiskit_account() -> None:
    # The remote tests require an active IBMQ account
    # We check if an IBMQ account is already saved, otherwise we try
    # to enable one using the token in the env variable:
    # PYTKET_REMOTE_IBM_CLOUD_TOKEN
    # and the instance in the env variable:
    # PYTKET_REMOTE_IBM_CLOUD_INSTANCE
    # Note: The IBMQ account will only be enabled for the current session
    if (
        os.getenv("PYTKET_RUN_REMOTE_TESTS") is not None
        and not QiskitRuntimeService.saved_accounts()
    ):
        instance = os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE")
        token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")
        if token:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum_platform", instance=instance, token=token, overwrite=True
            )


@pytest.fixture(scope="module")
def brussels_backend() -> IBMQBackend:
    return IBMQBackend(
        "ibm_brussels",
        instance=os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE"),
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )


@pytest.fixture(scope="module")
def brussels_emulator_backend() -> IBMQEmulatorBackend:
    return IBMQEmulatorBackend(
        "ibm_brussels",
        instance=os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE"),
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )


@pytest.fixture(scope="module")
def qiskit_runtime_service() -> QiskitRuntimeService:
    instance = os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE")
    token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")

    try:
        return QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)
    except:  # noqa: E722
        token = os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN")
        return QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=instance
        )


@pytest.fixture(scope="module")
def ibm_brussels_backend() -> IBMQBackend:
    return IBMQBackend(
        backend_name="ibm_brussels",
        monitor=False,
        instance=os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE"),
        token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
    )
