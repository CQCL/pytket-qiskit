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

"""Shared utility methods for ibm backends."""

import itertools
from collections.abc import Collection, Sequence
from typing import Callable, Optional

import numpy as np

from pytket.architecture import Architecture
from pytket.backends.status import StatusEnum
from pytket.circuit import Circuit, Node
from pytket.passes import RebaseTket
from pytket.transform import Transform
from qiskit.passmanager.flow_controllers import ConditionalController  # type: ignore
from qiskit.providers import JobStatus  # type: ignore
from qiskit.transpiler import CouplingMap, PassManager  # type: ignore
from qiskit.transpiler.passes import SabreLayout, SetLayout  # type: ignore
from qiskit.transpiler.passmanager_config import PassManagerConfig  # type: ignore
from qiskit.transpiler.preset_passmanagers import common  # type: ignore

from ..qiskit_convert import qiskit_to_tk, tk_to_qiskit

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


def _architecture_to_couplingmap(architecture: Architecture) -> CouplingMap:
    """
    Converts a pytket Architecture object to a Qiskit CouplingMap object.

    :param architecture: Architecture to be converted
    :return: A Qiskit CouplingMap object corresponding to the same connectivity
    """
    # we can make some assumptions from how the Architecture object is
    # originally constructed from the Qiskit CouplingMap:
    # 1) All nodes are single indexed
    # 2) All nodes are default register
    # 3) Node with index "i" corresponds to integer "i" in the original coupling map
    # We confirm assumption 1) and 2) while producing the coupling map
    coupling_map: list[tuple[int, int]] = []
    for edge in architecture.coupling:
        assert len(edge[0].index) == 1
        assert len(edge[1].index) == 1
        assert edge[0].reg_name == "node"
        assert edge[1].reg_name == "node"
        coupling_map.append((edge[0].index[0], edge[1].index[0]))
    return CouplingMap(coupling_map)


def _gen_lightsabre_transformation(  # type: ignore
    architecture: Architecture, seed=0, attempts=20
) -> Callable[[Circuit], Circuit]:
    """
    Generates a function that can be passed to CustomPass for running
    LightSABRE routing.

    :param architecture: Architecture LightSABRE routes circuits to match
    :param seed: LightSABRE routing is stochastic, with this parameter setting the seed
    :param attempts: Number of generated random solutions to pick from.
    :return: A function that accepts a pytket Circuit and returns a new Circuit that
        has been routed to the architecture using LightSABRE
    """
    config: PassManagerConfig = PassManagerConfig(
        coupling_map=_architecture_to_couplingmap(architecture),
        routing_method="sabre",
        seed_transpiler=seed,
    )
    sabre_pass: PassManager = PassManager(
        [
            SetLayout(config.initial_layout),
            ConditionalController(
                [
                    SabreLayout(
                        config.coupling_map,
                        max_iterations=2,
                        seed=config.seed_transpiler,
                        swap_trials=attempts,
                        layout_trials=attempts,
                        skip_routing=False,
                    )
                ],
                condition=lambda property_set: not property_set["layout"],
            ),
            ConditionalController(
                common.generate_embed_passmanager(
                    config.coupling_map
                ).to_flow_controller(),
                condition=lambda property_set: property_set["final_layout"] is None,
            ),
        ]
    )

    def lightsabre(circuit: Circuit) -> Circuit:
        c: Circuit = qiskit_to_tk(
            sabre_pass.run(tk_to_qiskit(circuit, replace_implicit_swaps=True))
        )
        c.remove_blank_wires()
        c.rename_units({q: Node(q.index[0]) for q in c.qubits})
        RebaseTket().apply(c)
        Transform.DecomposeCXDirected(architecture).apply(c)
        return c

    return lightsabre
