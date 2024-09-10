# Copyright 2019-2024 Quantinuum
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

from collections import Counter
from typing import (
    Optional,
    Sequence,
    Any,
)

from qiskit_aer.noise.noise_model import NoiseModel  # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

from pytket.backends.backend import Backend
from pytket.backends.status import CircuitStatus

from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple, ResultHandle
from pytket.circuit import Circuit
from pytket.passes import BasePass
from pytket.predicates import Predicate
from pytket.utils.results import KwargTypes

from .aer import AerBackend
from .ibm import IBMQBackend


class IBMQEmulatorBackend(Backend):
    """A backend which uses the AerBackend to loaclly emulate the behaviour of
    IBMQBackend. Identical to :py:class:`IBMQBackend` except there is no `monitor`
    parameter. Performs the same compilation and predicate checks as IBMQBackend.
    Requires a valid IBM account.
    """

    _supports_shots = False
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = False
    _supports_expectation = False

    def __init__(
        self,
        backend_name: str,
        instance: Optional[str] = None,
        service: Optional["QiskitRuntimeService"] = None,
        token: Optional[str] = None,
    ):
        super().__init__()
        self._ibmq = IBMQBackend(
            backend_name=backend_name,
            instance=instance,
            service=service,
            token=token,
        )

        # Get noise model:
        self._noise_model = NoiseModel.from_backend(self._ibmq._backend)

        # Construct AerBackend based on noise model:
        self._aer = AerBackend(noise_model=self._noise_model)

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: dict[tuple[str, int], Counter] = dict()

    @property
    def backend_info(self) -> BackendInfo:
        return self._ibmq._backend_info

    @property
    def required_predicates(self) -> list[Predicate]:
        return self._ibmq.required_predicates

    def default_compilation_pass(
        self,
        optimisation_level: int = 2,
        placement_options: Optional[dict[str, Any]] = None,
    ) -> BasePass:
        """
        See documentation for :py:meth:`IBMQBackend.default_compilation_pass`.
        """
        return self._ibmq.default_compilation_pass(
            optimisation_level=optimisation_level, placement_options=placement_options
        )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return self._aer._result_id_type

    def rebase_pass(self) -> BasePass:
        return self._ibmq.rebase_pass()

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[Optional[int]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.
        Supported kwargs: `seed`, `postprocess`.
        """

        if valid_check:
            self._ibmq._check_all_circuits(circuits)
        return self._aer.process_circuits(
            circuits, n_shots=n_shots, valid_check=False, **kwargs
        )

    def cancel(self, handle: ResultHandle) -> None:
        self._aer.cancel(handle)

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        return self._aer.circuit_status(handle)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: none.
        """
        return self._aer.get_result(handle)
