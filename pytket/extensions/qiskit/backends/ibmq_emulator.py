# Copyright 2019-2022 Cambridge Quantum Computing
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

from ast import literal_eval
from collections import Counter
import itertools
import json
from typing import (
    cast,
    Dict,
    Optional,
    List,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

from qiskit.providers.aer import AerSimulator  # type: ignore
from qiskit.providers.aer.noise.noise_model import NoiseModel  # type: ignore
from qiskit.providers.ibmq import AccountProvider  # type: ignore
from qiskit_ibm_runtime import (  # type: ignore
    QiskitRuntimeService,
    Session,
    Options,
    Sampler,
    RuntimeJob,
)

from pytket.backends import Backend, CircuitNotRunError, ResultHandle, CircuitStatus
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Bit, Circuit, OpType  # type: ignore
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit
from pytket.passes import BasePass  # type: ignore
from pytket.predicates import Predicate  # type: ignore
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes

from .ibm import IBMQBackend
from .ibm_utils import _STATUS_MAP, _batch_circuits


class IBMQEmulatorBackend(Backend):
    """A backend which uses the ibmq_qasm_simulator to emulate the behaviour of
    IBMQBackend. Performs the same compilation and predicate checks as IBMQBackend.
    Requires a valid IBMQ account.

    """

    _supports_shots = False
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = False
    _supports_expectation = False

    def __init__(
        self,
        backend_name: str,
        hub: Optional[str] = None,
        group: Optional[str] = None,
        project: Optional[str] = None,
        account_provider: Optional["AccountProvider"] = None,
        token: Optional[str] = None,
    ):
        """Construct an IBMQEmulatorBackend. Identical to :py:class:`IBMQBackend`
        constructor, except there is no `monitor` parameter. See :py:class:`IBMQBackend`
        docs for more details.
        """
        super().__init__()
        self._ibmq = IBMQBackend(
            backend_name=backend_name,
            hub=hub,
            group=group,
            project=project,
            account_provider=account_provider,
            token=token,
        )

        self._service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        self._session = Session(service=self._service, backend="ibmq_qasm_simulator")

        # Get noise model:
        aer_sim = AerSimulator.from_backend(self._ibmq._backend)
        self._noise_model = NoiseModel.from_backend(aer_sim)

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], Counter] = dict()

    @property
    def backend_info(self) -> BackendInfo:
        return self._ibmq._backend_info

    @property
    def required_predicates(self) -> List[Predicate]:
        return self._ibmq.required_predicates

    def default_compilation_pass(self, optimisation_level: int = 2) -> BasePass:
        return self._ibmq.default_compilation_pass(
            optimisation_level=optimisation_level
        )

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        # job ID, index, stringified sequence of measured bits, post-processing circuit
        return (str, int, str, str)

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
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        handle_list: List[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        batch_id = 0  # identify batches for debug purposes only
        for (n_shots, batch), indices in zip(circuit_batches, batch_order):
            for chunk in itertools.zip_longest(
                *([iter(zip(batch, indices))] * self._ibmq._max_per_job)
            ):
                filtchunk = list(filter(lambda x: x is not None, chunk))
                batch_chunk, indices_chunk = zip(*filtchunk)

                if valid_check:
                    self._check_all_circuits(batch_chunk)

                postprocess = kwargs.get("postprocess", False)

                qcs, c_bit_strs, ppcirc_strs = [], [], []
                for tkc in batch_chunk:
                    if postprocess:
                        c0, ppcirc = prepare_circuit(tkc, allow_classical=False)
                        ppcirc_rep = ppcirc.to_dict()
                    else:
                        c0, ppcirc_rep = tkc, None
                    qcs.append(tk_to_qiskit(c0))
                    measured_bits = sorted(
                        [cmd.args[1] for cmd in tkc if cmd.op.type == OpType.Measure]
                    )
                    c_bit_strs.append(
                        repr([(b.reg_name, b.index) for b in measured_bits])
                    )
                    ppcirc_strs.append(json.dumps(ppcirc_rep))
                options = Options()
                options.resilience_level = 0
                options.execution.shots = n_shots
                options.simulator.noise_model = self._noise_model
                options.seed_simulator = kwargs.get("seed")
                sampler = Sampler(session=self._session, options=options)
                job = sampler.run(circuits=qcs)
                job_id = job.job_id
                for i, ind in enumerate(indices_chunk):
                    handle_list[ind] = ResultHandle(
                        job_id, i, c_bit_strs[i], ppcirc_strs[i]
                    )
            batch_id += 1
        for handle in handle_list:
            assert handle is not None
            self._cache[handle] = dict()
        return cast(List[ResultHandle], handle_list)

    def _retrieve_job(self, jobid: str) -> RuntimeJob:
        return self._service.job(jobid)

    def cancel(self, handle: ResultHandle) -> None:
        jobid = cast(str, handle[0])
        job = self._retrieve_job(jobid)
        try:
            job.cancel()
        except Exception as e:
            warn(f"Unable to cancel job {jobid}: {e}")

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        self._check_handle_type(handle)
        jobid = cast(str, handle[0])
        job = self._service.job(jobid)
        ibmstatus = job.status()
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus.value)

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.
        Supported kwargs: `timeout`, `wait`.
        """
        self._check_handle_type(handle)
        if handle in self._cache:
            cached_result = self._cache[handle]
            if "result" in cached_result:
                return cast(BackendResult, cached_result["result"])
        jobid, index, c_bit_str, ppcirc_str = handle
        c_bits = [Bit(reg_name, index) for reg_name, index in literal_eval(c_bit_str)]
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        cache_key = (jobid, index)
        if cache_key not in self._ibm_res_cache:
            try:
                job = self._retrieve_job(jobid)
            except Exception as e:
                warn(f"Unable to retrieve job {jobid}: {e}")
                raise CircuitNotRunError(handle)

            res = job.result(timeout=kwargs.get("timeout", None))
            for circ_index, (r, d) in enumerate(zip(res.quasi_dists, res.metadata)):
                self._ibm_res_cache[(jobid, circ_index)] = Counter(
                    {n: int(0.5 + d["shots"] * p) for n, p in r.items()}
                )

        counts = self._ibm_res_cache[cache_key]  # Counter[int]
        # Convert to `OutcomeArray`:
        tket_counts: Counter = Counter()
        for outcome_key, sample_count in counts.items():
            array = OutcomeArray.from_ints(
                ints=[outcome_key],
                width=len(c_bits),
                big_endian=False,
            )
            tket_counts[array] = sample_count
        # Convert to `BackendResult`:
        result = BackendResult(c_bits=c_bits, counts=tket_counts, ppcirc=ppcirc)

        self._cache[handle] = {"result": result}
        return result
