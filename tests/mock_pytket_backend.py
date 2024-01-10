# Copyright 2019-2023 Cambridge Quantum Computing
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


from typing import Optional, Union, List, Sequence, Set, cast
import json

from pytket.circuit import Circuit, OpType
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backendinfo import BackendInfo
from pytket.architecture import Architecture, FullyConnected
from pytket.predicates import Predicate, GateSetPredicate
from pytket.passes import BasePass, CustomPass
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.backends.backendresult import BackendResult
from pytket.backends.backend import KwargTypes, ResultCache
from pytket.utils.outcomearray import OutcomeArray


class MockShotBackend(Backend):
    def __init__(
        self,
        arch: Optional[Union[Architecture, FullyConnected]] = None,
        gate_set: Optional[Set[OpType]] = None,
    ):
        """Mock shot backend for testing qiskit embedding. This should only be used
        in conjunction with the TketBackend. The readout bitstring will always be 1s.
        :param arch: The backend architecture
        :type arch: Optional[Union[Architecture, FullyConnected]]
        :param gate_set: The supported gateset, default to {OpType.CX, OpType.U3}
        :type gate_set: Optional[Set[OpType]]
        """
        self._id = 0
        self._arch = arch
        if gate_set:
            self._gate_set = gate_set
        else:
            self._gate_set = {OpType.CX, OpType.U3}

    @property
    def required_predicates(self) -> List[Predicate]:
        """Returns a GateSetPredicate constructed with the given gateset."""
        return [GateSetPredicate(self._gate_set)]

    def rebase_pass(self) -> BasePass:
        """Return a pass that does nothing to a circuit."""
        return CustomPass(lambda c: c)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        """Return a pass that does nothing to a circuit."""
        return self.rebase_pass()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (int, str)

    @property
    def backend_info(self) -> Optional[BackendInfo]:
        """Returns a BackendInfo constructed with the given architecture."""
        return BackendInfo(
            name="TketBackend",
            device_name="MockShotBackend",
            version="0.0.1",
            gate_set=self._gate_set,
            architecture=self._arch,
        )

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Optional[Union[int, Sequence[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """Mock processing the circuits."""
        handles = []
        for c in circuits:
            handles.append(ResultHandle(self._id, json.dumps(c.to_dict())))
            self._id = self._id + 1
        return handles

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        """Always StatusEnum.COMPLETED."""
        return CircuitStatus(StatusEnum.COMPLETED)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """Always return a single readout containing all 1s."""
        circ_rep = json.loads(cast(str, handle[1]))
        circ = Circuit.from_dict(circ_rep)
        shots_list = [[1] * circ.n_bits]
        outcome_arr = OutcomeArray.from_readouts(shots_list)
        return BackendResult(shots=outcome_arr, q_bits=circ.qubits, c_bits=circ.bits)

    def pop_result(self, handle: ResultHandle) -> Optional[ResultCache]:
        """Does nothing. Implementation is required by TketJob."""
        return None

    def cancel(self, handle: ResultHandle) -> None:
        """Does nothing. Implementation is required by TketJob."""
        return
