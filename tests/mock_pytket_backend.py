from typing import Optional, Union, List, Sequence, Set, cast
import json

from pytket.circuit import Circuit, OpType  # type: ignore
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backendinfo import BackendInfo
from pytket.architecture import Architecture, FullyConnected  # type: ignore
from pytket.predicates import Predicate, GateSetPredicate  # type: ignore
from pytket.passes import BasePass, CustomPass  # type: ignore
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
        """Mock shot backend for testing qiskit embedding.
        The readout bitstring will always be all 1s."""
        self._id = 0
        self._arch = arch
        if gate_set:
            self._gate_set = gate_set
        else:
            self._gate_set = {OpType.CX, OpType.U3}

    @property
    def required_predicates(self) -> List[Predicate]:
        return [GateSetPredicate(self._gate_set)]

    def rebase_pass(self) -> BasePass:
        return CustomPass(lambda c: c)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        return self.rebase_pass()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (int, str)

    @property
    def backend_info(self) -> Optional[BackendInfo]:
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
        handles = []
        for c in circuits:
            handles.append(ResultHandle(self._id, json.dumps(c.to_dict())))
            self._id = self._id + 1
        return handles

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        return CircuitStatus(StatusEnum.COMPLETED)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        circ_rep = json.loads(cast(str, handle[1]))
        circ = Circuit.from_dict(circ_rep)
        shots_list = [[1] * circ.n_bits]
        outcome_arr = OutcomeArray.from_readouts(shots_list)
        return BackendResult(shots=outcome_arr, q_bits=circ.qubits, c_bits=circ.bits)

    def pop_result(self, handle: ResultHandle) -> Optional[ResultCache]:
        return None

    def cancel(self, handle: ResultHandle) -> None:
        return
