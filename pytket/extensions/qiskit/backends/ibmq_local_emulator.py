# Copyright 2019-2024 Cambridge Quantum Computing
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
    Dict,
    Optional,
    List,
    Sequence,
    Tuple,
    Union,
)

from qiskit.providers.aer.noise.noise_model import NoiseModel  # type: ignore

from qiskit_ibm_provider import IBMProvider  # type: ignore

from pytket.backends import (
    Backend,
    ResultHandle,
    CircuitStatus,
)
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit
from pytket.passes import BasePass
from pytket.predicates import Predicate
from pytket.utils.results import KwargTypes

from .aer import AerBackend
from .ibm import IBMQBackend


class IBMQLocalEmulatorBackend(Backend):
    """A backend which uses the AerBackend to locally emulate the behaviour of
    IBMQBackend. Performs the same compilation and predicate checks as IBMQBackend.
    Requires a valid IBMQ account.

    """

    _supports_shots = False
    _supports_counts = True
    _supports_contextual_optimisation = False
    _persistent_handles = False
    _supports_expectation = False

    def __init__(
        self,
        backend_name: str,
        instance: Optional[str] = None,
        provider: Optional["IBMProvider"] = None,
        token: Optional[str] = None,
    ):
        """Construct an IBMQLocalEmulatorBackend. Identical to :py:class:`IBMQBackend`
        constructor, except there is no `monitor` parameter. See :py:class:`IBMQBackend`
        docs for more details.
        """
        super().__init__()
        self._ibmq = IBMQBackend(
            backend_name=backend_name,
            instance=instance,
            provider=provider,
            token=token,
        )

        # Get noise model:
        self._noise_model = NoiseModel.from_backend(self._ibmq._backend)

        # Construct AerBackend based on noise model:
        self._aer = AerBackend(noise_model=self._noise_model)

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], Counter] = dict()

    @property
    def backend_info(self) -> BackendInfo:
        return self._ibmq._backend_info

    @property
    def required_predicates(self) -> List[Predicate]:
        return self._ibmq.required_predicates

    def default_compilation_pass(
        self, optimisation_level: int = 2, placement_options: Optional[Dict] = None
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
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
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
