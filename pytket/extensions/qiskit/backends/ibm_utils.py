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
from typing import Collection, Optional, Sequence, Tuple, List, TYPE_CHECKING, Union

import numpy as np

from qiskit.providers import JobStatus  # type: ignore

from pytket.backends.status import StatusEnum

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
}


def _batch_circuits(
    circuits: Sequence["Circuit"],
    n_shots: Sequence[Optional[int]],
    seed: Union[int, float, str, None, Sequence[Optional[int]]],
) -> Tuple[List[Tuple[Optional[int], Optional[int], List["Circuit"]]], List[List[int]]]:
    """
    Groups circuits into sets of circuits with the same number of shots.

    Returns a tuple of circuit batches and their ordering.

    :param circuits: Circuits to be grouped.
    :type circuits: Sequence[Circuit]
    :param n_shots: Number of shots for each circuit.
    :type n_shots: Sequence[int]
    :param seed: RNG Seed for each circuit.
    :type seed: Union[int, None, Sequence[Optional[int]]]
    """

    n_seeds: list[Optional[int]] = []
    if type(seed) == list:
        n_seeds = seed
    elif type(seed) == int:
        n_seeds = [seed for _ in range(len(circuits))]
    elif seed == None:
        n_seeds = [None for _ in range(len(circuits))]
    else:
        raise ValueError(
            f"""unknown seed type, type should be None,
int, or list[int], type found {type(seed)}"""
        )

    assert len(n_seeds) == len(n_shots)
    assert len(n_seeds) == len(circuits)

    batches: List[Tuple[Optional[int], Optional[int], List["Circuit"]]] = []
    batch_order: List[List[int]] = []

    if all(seed == n_seeds[0] for seed in n_seeds):
        # take care of None entries
        n_shots_int = list(map(lambda x: x if x is not None else -1, n_shots))

        order: Collection[int] = np.argsort(n_shots_int)

        batches = [
            (n, n_seeds[0], [circuits[i] for i in indices])
            for n, indices in itertools.groupby(order, key=lambda i: n_shots[i])
        ]
        batch_order = [
            list(indices)
            for n, indices in itertools.groupby(order, key=lambda i: n_shots[i])
        ]
    else:

        for i in range(len(circuits)):
            batches.append((n_shots[i], n_seeds[i], [circuits[i]]))
            batch_order.append([i])

    return batches, batch_order
