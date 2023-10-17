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

import itertools
import logging
from ast import literal_eval
from collections import Counter
import json
from time import sleep
from typing import (
    cast,
    List,
    Optional,
    Dict,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
    Set,
    Any,
)
from warnings import warn

from qiskit_ibm_provider import IBMProvider  # type: ignore
from qiskit_ibm_provider.exceptions import IBMProviderError  # type: ignore
from qiskit.primitives import SamplerResult  # type: ignore


# RuntimeJob has no queue_position attribute, which is referenced
# via job_monitor see-> https://github.com/CQCL/pytket-qiskit/issues/48
# therefore we can't use job_monitor until fixed
# from qiskit.tools.monitor import job_monitor  # type: ignore
from qiskit.result.distributions import QuasiDistribution  # type: ignore
from qiskit_ibm_runtime import (  # type: ignore
    QiskitRuntimeService,
    Session,
    Options,
    Sampler,
    RuntimeJob,
)

from pytket.circuit import Circuit, OpType
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.extensions.qiskit.qiskit_convert import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket.extensions.qiskit._metadata import __extension_version__
from pytket.passes import (
    BasePass,
    auto_rebase_pass,
    KAKDecomposition,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
    NaivePlacementPass,
)
from pytket.predicates import (
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    GateSetPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    MaxNQubitsPredicate,
    Predicate,
)
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, _tk_gate_set
from pytket.architecture import FullyConnected
from pytket.placement import NoiseAwarePlacement
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes
from .ibm_utils import _STATUS_MAP, _batch_circuits
from .config import QiskitConfig

if TYPE_CHECKING:
    from qiskit_ibm_provider.ibm_backend import IBMBackend as _QiskIBMBackend  # type: ignore

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"


def _gen_debug_results(n_qubits: int, shots: int, index: int) -> SamplerResult:
    debug_dist = {n: 0.0 for n in range(pow(2, n_qubits))}
    debug_dist[0] = 1.0
    qd = QuasiDistribution(debug_dist)
    return SamplerResult(
        quasi_dists=[qd] * (index + 1),
        metadata=[{"header_metadata": {}, "shots": shots}] * (index + 1),
    )


class NoIBMQCredentialsError(Exception):
    """Raised when there is no IBMQ account available for the backend"""

    def __init__(self) -> None:
        super().__init__(
            "No IBMQ credentials found on disk, store your account using qiskit,"
            " or using :py:meth:`pytket.extensions.qiskit.set_ibmq_config` first."
        )


def _save_ibmq_auth(qiskit_config: Optional[QiskitConfig]) -> None:
    token = None
    if qiskit_config is not None:
        token = qiskit_config.ibmq_api_token
    try:
        if token is not None:
            IBMProvider.save_account(token, overwrite=True)
            IBMProvider()
        else:
            IBMProvider()
    except:
        if token is not None:
            IBMProvider.save_account(token, overwrite=True)
            IBMProvider()
        else:
            raise NoIBMQCredentialsError()
    if not QiskitRuntimeService.saved_accounts():
        if token is not None:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)


def _get_primitive_gates(gateset: Set[OpType]) -> Set[OpType]:
    if gateset >= {OpType.X, OpType.SX, OpType.Rz, OpType.CX}:
        return {OpType.X, OpType.SX, OpType.Rz, OpType.CX}
    elif gateset >= {OpType.X, OpType.SX, OpType.Rz, OpType.ECR}:
        return {OpType.X, OpType.SX, OpType.Rz, OpType.ECR}
    else:
        return gateset


class IBMQBackend(Backend):
    _supports_shots = False
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        backend_name: str,
        instance: Optional[str] = None,
        monitor: bool = True,
        provider: Optional["IBMProvider"] = None,
        token: Optional[str] = None,
    ):
        """A backend for running circuits on remote IBMQ devices.
        The provider arguments of `hub`, `group` and `project` can
        be specified here as parameters or set in the config file
        using :py:meth:`pytket.extensions.qiskit.set_ibmq_config`.
        This function can also be used to set the IBMQ API token.

        :param backend_name: Name of the IBMQ device, e.g. `ibmq_16_melbourne`.
        :type backend_name: str
        :param instance: String containing information about the hub/group/project.
        :type instance: str, optional
        :param monitor: Use the IBM job monitor. Defaults to True.
        :type monitor: bool, optional
        :raises ValueError: If no IBMQ account is loaded and none exists on the disk.
        :param provider: An IBMProvider
        :type provider: Optional[IBMProvider]
        :param token: Authentication token to use the `QiskitRuntimeService`.
        :type token: Optional[str]
        """
        super().__init__()
        self._pytket_config = QiskitConfig.from_default_config_file()
        self._provider = (
            self._get_provider(instance=instance, qiskit_config=self._pytket_config)
            if provider is None
            else provider
        )
        self._backend: "_QiskIBMBackend" = self._provider.get_backend(backend_name)
        config = self._backend.configuration()
        self._max_per_job = getattr(config, "max_experiments", 1)

        gate_set = _tk_gate_set(self._backend)
        self._backend_info = self._get_backend_info(self._backend)

        self._service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        self._session = Session(service=self._service, backend=backend_name)

        self._primitive_gates = _get_primitive_gates(gate_set)

        self._supports_rz = OpType.Rz in self._primitive_gates

        self._monitor = monitor

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: Dict[Tuple[str, int], Counter] = dict()

        self._MACHINE_DEBUG = False

    @staticmethod
    def _get_provider(
        instance: Optional[str],
        qiskit_config: Optional[QiskitConfig],
    ) -> "IBMProvider":
        _save_ibmq_auth(qiskit_config)
        try:
            if instance is not None:
                provider = IBMProvider(instance=instance)
            else:
                provider = IBMProvider()
        except IBMProviderError as err:
            logging.warn(
                (
                    "Provider was not specified enough, specify hub,"
                    "group and project correctly (check your IBMQ account)."
                )
            )
            raise err

        return provider

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    @classmethod
    def _get_backend_info(cls, backend: "_QiskIBMBackend") -> BackendInfo:
        config = backend.configuration()
        characterisation = process_characterisation(backend)
        averaged_errors = get_avg_characterisation(characterisation)
        characterisation_keys = [
            "t1times",
            "t2times",
            "Frequencies",
            "GateTimes",
        ]
        arch = characterisation["Architecture"]
        # filter entries to keep
        filtered_characterisation = {
            k: v for k, v in characterisation.items() if k in characterisation_keys
        }
        # see below for references for config definitions
        # quantum-computing.ibm.com/services/resources/docs/resources/manage/systems/:
        # midcircuit-measurement/
        # dynamic-circuits/feature-table
        supports_mid_measure = config.simulator or config.multi_meas_enabled
        supports_fast_feedforward = (
            hasattr(config, "supported_features")
            and "qasm3" in config.supported_features
        )

        # simulator i.e. "ibmq_qasm_simulator" does not have `supported_instructions`
        # attribute
        supports_reset = (
            hasattr(config, "supported_instructions")
            and "reset" in config.supported_instructions
        )
        gate_set = _tk_gate_set(backend)
        backend_info = BackendInfo(
            cls.__name__,
            backend.name,
            __extension_version__,
            arch,
            gate_set.union(
                {
                    OpType.RangePredicate,
                    OpType.Conditional,
                }
            )
            if supports_fast_feedforward
            else gate_set,
            supports_midcircuit_measurement=supports_mid_measure,
            supports_fast_feedforward=supports_fast_feedforward,
            supports_reset=supports_reset,
            all_node_gate_errors=characterisation["NodeErrors"],
            all_edge_gate_errors=characterisation["EdgeErrors"],
            all_readout_errors=characterisation["ReadoutErrors"],
            averaged_node_gate_errors=averaged_errors["node_errors"],
            averaged_edge_gate_errors=averaged_errors["edge_errors"],  # type: ignore
            averaged_readout_errors=averaged_errors["readout_errors"],
            misc={"characterisation": filtered_characterisation},
        )
        return backend_info

    @classmethod
    def available_devices(cls, **kwargs: Any) -> List[BackendInfo]:
        provider: Optional["IBMProvider"] = kwargs.get("provider")
        if provider is None:
            if kwargs.get("instance") is not None:
                provider = cls._get_provider(
                    instance=kwargs.get("instance"), qiskit_config=None
                )
            else:
                provider = IBMProvider()

        backend_info_list = [
            cls._get_backend_info(backend) for backend in provider.backends()
        ]
        return backend_info_list

    @property
    def required_predicates(self) -> List[Predicate]:
        predicates = [
            NoSymbolsPredicate(),
            MaxNQubitsPredicate(self._backend_info.n_nodes),
            GateSetPredicate(
                self._backend_info.gate_set.union(
                    {
                        OpType.Barrier,
                    }
                )
            ),
        ]
        mid_measure = self._backend_info.supports_midcircuit_measurement
        fast_feedforward = self._backend_info.supports_fast_feedforward
        if not mid_measure:
            predicates = [
                NoClassicalControlPredicate(),
                NoMidMeasurePredicate(),
            ] + predicates
        if not fast_feedforward:
            predicates = [
                NoFastFeedforwardPredicate(),
            ] + predicates
        return predicates

    def default_compilation_pass(
        self, optimisation_level: int = 2, placement_options: Optional[Dict] = None
    ) -> BasePass:
        """
        A suggested compilation pass that will will, if possible, produce an equivalent
        circuit suitable for running on this backend.

        At a minimum it will ensure that compatible gates are used and that all two-
        qubit interactions are compatible with the backend's qubit architecture. At
        higher optimisation levels, further optimisations may be applied.

        This is a an abstract method which is implemented in the backend itself, and so
        is tailored to the backend's requirements.

        The default compilation passes for the :py:class:`IBMQBackend`,
        :py:class:`IBMQEmulatorBackend` and the
        Aer simulators support an optional ``placement_options`` dictionary containing
        arguments to override the default settings in :py:class:`NoiseAwarePlacement`.

        :param optimisation_level: The level of optimisation to perform during
            compilation.

            - Level 0 does the minimum required to solves the device constraints,
              without any optimisation.
            - Level 1 additionally performs some light optimisations.
            - Level 2 (the default) adds more computationally intensive optimisations
              that should give the best results from execution.

        :type optimisation_level: int, optional

        :param placement_options: Optional argument allowing the user to override
          the default settings in :py:class:`NoiseAwarePlacement`.
        :type placement_options: Dict, optional
        :return: Compilation pass guaranteeing required predicates.
        :rtype: BasePass
        """
        assert optimisation_level in range(3)
        passlist = [DecomposeBoxes()]
        # If you make changes to the default_compilation_pass,
        # then please update this page accordingly
        # https://cqcl.github.io/pytket-qiskit/api/index.html#default-compilation
        # Edit this docs source file -> pytket-qiskit/docs/intro.txt
        if optimisation_level == 0:
            if self._supports_rz:
                # If the Rz gate is unsupported then the rebase should be skipped
                # This prevents an error when compiling to the stabilizer backend
                # where no TK1 replacement can be found for the rebase.
                passlist.append(self.rebase_pass())
        elif optimisation_level == 1:
            passlist.append(SynthesiseTket())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        mid_measure = self._backend_info.supports_midcircuit_measurement
        arch = self._backend_info.architecture
        if not isinstance(arch, FullyConnected):
            if placement_options is not None:
                noise_aware_placement = NoiseAwarePlacement(
                    arch,
                    self._backend_info.averaged_node_gate_errors,  # type: ignore
                    self._backend_info.averaged_edge_gate_errors,  # type: ignore
                    self._backend_info.averaged_readout_errors,  # type: ignore
                    **placement_options,
                )
            else:
                noise_aware_placement = NoiseAwarePlacement(
                    arch,
                    self._backend_info.averaged_node_gate_errors,  # type: ignore
                    self._backend_info.averaged_edge_gate_errors,  # type: ignore
                    self._backend_info.averaged_readout_errors,  # type: ignore
                )

            passlist.append(
                CXMappingPass(
                    arch,
                    noise_aware_placement,
                    directed_cx=False,
                    delay_measures=(not mid_measure),
                )
            )
            passlist.append(NaivePlacementPass(arch))
        if optimisation_level == 1:
            passlist.append(SynthesiseTket())
        if optimisation_level == 2:
            passlist.extend(
                [
                    KAKDecomposition(allow_swaps=False),
                    CliffordSimp(False),
                    SynthesiseTket(),
                ]
            )

        if self._supports_rz:
            passlist.extend([self.rebase_pass(), RemoveRedundancies()])
        return SequencePass(passlist)

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        # IBMQ job ID, index, number of measurements per shot, post-processing circuit
        return (str, int, int, str)

    def rebase_pass(self) -> BasePass:
        return auto_rebase_pass(self._primitive_gates)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> List[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported `kwargs`:
        - `postprocess`: apply end-of-circuit simplifications and classical
          postprocessing to improve fidelity of results (bool, default False)
        - `simplify_initial`: apply the pytket ``SimplifyInitial`` pass to improve
          fidelity of results assuming all qubits initialized to zero (bool, default
          False)
        """
        circuits = list(circuits)

        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        handle_list: List[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        postprocess = kwargs.get("postprocess", False)
        simplify_initial = kwargs.get("simplify_initial", False)

        batch_id = 0  # identify batches for debug purposes only
        for (n_shots, batch), indices in zip(circuit_batches, batch_order):
            for chunk in itertools.zip_longest(
                *([iter(zip(batch, indices))] * self._max_per_job)
            ):
                filtchunk = list(filter(lambda x: x is not None, chunk))
                batch_chunk, indices_chunk = zip(*filtchunk)

                if valid_check:
                    self._check_all_circuits(batch_chunk)

                qcs, ppcirc_strs = [], []
                for tkc in batch_chunk:
                    if postprocess:
                        c0, ppcirc = prepare_circuit(tkc, allow_classical=False)
                        ppcirc_rep = ppcirc.to_dict()
                    else:
                        c0, ppcirc_rep = tkc, None
                    if simplify_initial:
                        SimplifyInitial(
                            allow_classical=False, create_all_qubits=True
                        ).apply(c0)
                    qcs.append(tk_to_qiskit(c0))
                    ppcirc_strs.append(json.dumps(ppcirc_rep))
                if self._MACHINE_DEBUG:
                    for i, ind in enumerate(indices_chunk):
                        handle_list[ind] = ResultHandle(
                            _DEBUG_HANDLE_PREFIX + str((n_shots, batch_id)),
                            i,
                            batch_chunk[i].n_qubits,
                            ppcirc_strs[i],
                        )
                else:
                    options = Options()
                    options.optimization_level = 0
                    options.resilience_level = 0
                    options.transpilation.skip_transpilation = True
                    options.execution.shots = n_shots
                    sampler = Sampler(session=self._session, options=options)
                    job = sampler.run(
                        circuits=qcs,
                        dynamic=self.backend_info.supports_fast_feedforward,
                    )
                    job_id = job.job_id()
                    for i, ind in enumerate(indices_chunk):
                        handle_list[ind] = ResultHandle(
                            job_id, i, qcs[i].count_ops()["measure"], ppcirc_strs[i]
                        )
            batch_id += 1
        for handle in handle_list:
            assert handle is not None
            self._cache[handle] = dict()
        return cast(List[ResultHandle], handle_list)

    def _retrieve_job(self, jobid: str) -> RuntimeJob:
        return self._service.job(jobid)

    def cancel(self, handle: ResultHandle) -> None:
        if not self._MACHINE_DEBUG:
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
        jobid, index, n_meas, ppcirc_str = handle
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        cache_key = (jobid, index)
        if cache_key not in self._ibm_res_cache:
            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                shots: int
                shots, _ = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
                res = _gen_debug_results(n_meas, shots, index)
            else:
                try:
                    job = self._retrieve_job(jobid)
                except Exception as e:
                    warn(f"Unable to retrieve job {jobid}: {e}")
                    raise CircuitNotRunError(handle)
                # RuntimeJob has no queue_position attribute, which is referenced
                # via job_monitor see-> https://github.com/CQCL/pytket-qiskit/issues/48
                # therefore we can't use job_monitor until fixed
                if self._monitor and job:
                    #     job_monitor(job)
                    status = job.status()
                    while status.name not in ["DONE", "CANCELLED", "ERROR"]:
                        status = job.status()
                        print("Job status is", status.name)
                        sleep(10)

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
                width=n_meas,
                big_endian=False,
            )
            tket_counts[array] = sample_count
        # Convert to `BackendResult`:
        result = BackendResult(counts=tket_counts, ppcirc=ppcirc)

        self._cache[handle] = {"result": result}
        return result
