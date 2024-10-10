# Copyright 2019 Cambridge Quantum Computing
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

"""Shared utility methods for ibm backends.
"""

import itertools
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Optional

import numpy as np

from pytket.backends.status import StatusEnum
from qiskit.providers import JobStatus  # type: ignore

if TYPE_CHECKING:
    from pytket.circuit import Circuit

_STATUS_MAP = {
    JobStatus.CANCELLED: StatusEnum.CANCELLED,
    JobStatus.ERROR: StatusEnum.ERROR,
    JobStatus.DONE: StatusEnum.COMPLETED,
    JobStatus.INITIALIZING: StatusEnum.SUBMITTED,
    JobStatus.VALIDATING: StatusEnum.SUBMITTED,
    JobStatus.QUEUED: StatusEnum.QUEUED,
    JobStatus.RUNNING: StatusEnum.RUNNING,
    "CANCELLED": StatusEnum.CANCELLED,
    "ERROR": StatusEnum.ERROR,
    "DONE": StatusEnum.COMPLETED,
    "INITIALIZING": StatusEnum.SUBMITTED,
    "VALIDATING": StatusEnum.SUBMITTED,
    "QUEUED": StatusEnum.QUEUED,
    "RUNNING": StatusEnum.RUNNING,
}


def _batch_circuits(
    circuits: Sequence["Circuit"],
    n_shots: Sequence[Optional[int]],
) -> tuple[list[tuple[Optional[int], list["Circuit"]]], list[list[int]]]:
    """
    Groups circuits into sets of circuits with the same number of shots.

    Returns a tuple of circuit batches and their ordering.

    :param circuits: Circuits to be grouped.
    :param n_shots: Number of shots for each circuit.
    """
    # take care of None entries
    n_shots_int = list(map(lambda x: x if x is not None else -1, n_shots))

    order: Collection[int] = np.argsort(n_shots_int)
    batches: list[tuple[Optional[int], list[Circuit]]] = [
        (n, [circuits[i] for i in indices])
        for n, indices in itertools.groupby(order, key=lambda i: n_shots[i])
    ]
    batch_order: list[list[int]] = [
        list(indices)
        for n, indices in itertools.groupby(order, key=lambda i: n_shots[i])
    ]
    return batches, batch_order
