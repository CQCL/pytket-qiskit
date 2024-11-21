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

import itertools
import json
from ast import literal_eval
from collections import Counter, OrderedDict
from collections.abc import Sequence
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    cast,
)
from warnings import warn

import numpy as np
from qiskit_ibm_runtime import (  # type: ignore
    QiskitRuntimeService,
    RuntimeJob,
    SamplerOptions,
    SamplerV2,
    Session,
)
from qiskit_ibm_runtime.models.backend_configuration import (  # type: ignore
    PulseBackendConfiguration,
)
from qiskit_ibm_runtime.models.backend_properties import (  # type: ignore
    BackendProperties,
)

from pytket.architecture import Architecture, FullyConnected
from pytket.backends import Backend, CircuitNotRunError, CircuitStatus, ResultHandle
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Bit, Circuit, OpType
from pytket.passes import (
    AutoRebase,
    BasePass,
    CliffordSimp,
    CustomPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    GreedyPauliSimp,
    KAKDecomposition,
    RemoveBarriers,
    RemoveRedundancies,
    SequencePass,
    SimplifyInitial,
    SynthesiseTket,
)
from pytket.predicates import (
    DirectednessPredicate,
    GateSetPredicate,
    MaxNQubitsPredicate,
    NoClassicalControlPredicate,
    NoFastFeedforwardPredicate,
    NoMidMeasurePredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.utils.results import KwargTypes
from qiskit.primitives import (  # type: ignore
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubResult,
)

# RuntimeJob has no queue_position attribute, which is referenced
# via job_monitor see-> https://github.com/CQCL/pytket-qiskit/issues/48
# therefore we can't use job_monitor until fixed
# from qiskit.tools.monitor import job_monitor  # type: ignore
from .._metadata import __extension_version__
from ..qiskit_convert import (
    _tk_gate_set,
    get_avg_characterisation,
    process_characterisation_from_config,
    tk_to_qiskit,
)
from .config import QiskitConfig
from .ibm_utils import _STATUS_MAP, _batch_circuits, _gen_lightsabre_transformation

if TYPE_CHECKING:
    from qiskit_ibm_runtime.ibm_backend import IBMBackend  # type: ignore

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"


def _gen_debug_results(n_bits: int, shots: int) -> PrimitiveResult:
    n_u8s = (n_bits - 1) // 8 + 1
    arr = np.array([[0] * n_u8s for _ in range(shots)], dtype=np.uint8)
    return PrimitiveResult([SamplerPubResult(DataBin(c=BitArray(arr, n_bits)))])


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
    if token is not None and not QiskitRuntimeService.saved_accounts():
        QiskitRuntimeService.save_account(
            channel="ibm_quantum", token=token, overwrite=True
        )


def _get_primitive_gates(gateset: set[OpType]) -> set[OpType]:
    if gateset >= {OpType.X, OpType.SX, OpType.Rz, OpType.CX}:
        return {OpType.X, OpType.SX, OpType.Rz, OpType.CX}
    elif gateset >= {OpType.X, OpType.SX, OpType.Rz, OpType.ECR}:
        return {OpType.X, OpType.SX, OpType.Rz, OpType.ECR}
    else:
        return gateset


def _int_from_readout(readout: np.ndarray) -> int:
    # Weird mixture of big- and little-endian here.
    n_bytes = len(readout)
    return sum(int(x) << (8 * (n_bytes - 1 - i)) for i, x in enumerate(readout))


class IBMQBackend(Backend):
    """A backend for running circuits on remote IBMQ devices.

    The provider arguments of `hub`, `group` and `project` can
    be specified here as parameters or set in the config file
    using :py:meth:`pytket.extensions.qiskit.set_ibmq_config`.
    This function can also be used to set the IBMQ API token.

    :param backend_name: Name of the IBMQ device, e.g. `ibmq_16_melbourne`.
    :param instance: String containing information about the hub/group/project.
    :param monitor: Use the IBM job monitor. Defaults to True.
    :raises ValueError: If no IBMQ account is loaded and none exists on the disk.
    :param service: A QiskitRuntimeService
    :param token: Authentication token to use the `QiskitRuntimeService`.
    :param sampler_options: A customised `qiskit_ibm_runtime` `SamplerOptions` instance.
        See the Qiskit documentation at
        https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.options.SamplerOptions
        for details and default values.
    """

    _supports_shots = False
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(
        self,
        backend_name: str,
        instance: Optional[str] = None,
        monitor: bool = True,
        service: Optional[QiskitRuntimeService] = None,
        token: Optional[str] = None,
        sampler_options: SamplerOptions = None,
    ):
        super().__init__()
        self._pytket_config = QiskitConfig.from_default_config_file()
        self._service = (
            self._get_service(instance=instance, qiskit_config=self._pytket_config)
            if service is None
            else service
        )
        self._backend: IBMBackend = self._service.backend(backend_name)
        config: PulseBackendConfiguration = self._backend.configuration()
        self._max_per_job = getattr(config, "max_experiments", 1)

        gate_set = _tk_gate_set(config)
        props: Optional[BackendProperties] = self._backend.properties()
        self._backend_info = self._get_backend_info(config, props)

        self._service = QiskitRuntimeService(
            channel="ibm_quantum", token=token, instance=instance
        )
        self._session = Session(backend=self._backend)

        self._primitive_gates = _get_primitive_gates(gate_set)

        self._supports_rz = OpType.Rz in self._primitive_gates

        self._monitor = monitor

        # cache of results keyed by job id and circuit index
        self._ibm_res_cache: dict[
            tuple[str, int], tuple[Counter, Optional[list[Bit]]]
        ] = dict()

        if sampler_options is None:
            sampler_options = SamplerOptions()
        self._sampler_options = sampler_options

        self._MACHINE_DEBUG = False

    @staticmethod
    def _get_service(
        instance: Optional[str],
        qiskit_config: Optional[QiskitConfig],
    ) -> QiskitRuntimeService:
        _save_ibmq_auth(qiskit_config)
        if instance is not None:
            return QiskitRuntimeService(channel="ibm_quantum", instance=instance)
        else:
            return QiskitRuntimeService(channel="ibm_quantum")

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    @classmethod
    def _get_backend_info(
        cls,
        config: PulseBackendConfiguration,
        props: Optional[BackendProperties],
    ) -> BackendInfo:
        """Construct a BackendInfo from data returned by the IBMQ API.

        :param config: The configuration of this backend.
        :param props: The measured properties of this backend (not required).
        :return: Information about the backend.
        """
        characterisation = process_characterisation_from_config(config, props)
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
        gate_set = _tk_gate_set(config)
        backend_info = BackendInfo(
            cls.__name__,
            config.backend_name,
            __extension_version__,
            arch,
            (
                gate_set.union(
                    {
                        OpType.RangePredicate,
                        OpType.Conditional,
                    }
                )
                if supports_fast_feedforward
                else gate_set
            ),
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
    def available_devices(cls, **kwargs: Any) -> list[BackendInfo]:
        service: Optional[QiskitRuntimeService] = kwargs.get("service")
        if service is None:
            instance = kwargs.get("instance")
            if instance is not None:
                service = cls._get_service(instance=instance, qiskit_config=None)
            else:
                service = QiskitRuntimeService(channel="ibm_quantum")

        backend_info_list = []
        for backend in service.backends():
            config = backend.configuration()
            props = backend.properties()
            backend_info_list.append(cls._get_backend_info(config, props))

        return backend_info_list

    @property
    def required_predicates(self) -> list[Predicate]:
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
        if isinstance(self.backend_info.architecture, Architecture):
            predicates.append(DirectednessPredicate(self.backend_info.architecture))

        mid_measure = self._backend_info.supports_midcircuit_measurement
        fast_feedforward = self._backend_info.supports_fast_feedforward
        if not mid_measure:
            predicates.append(NoClassicalControlPredicate())
            predicates.append(NoMidMeasurePredicate())
        if not fast_feedforward:
            predicates.append(NoFastFeedforwardPredicate())
        return predicates

    def default_compilation_pass(
        self,
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        """
        A suggested compilation pass that will will, if possible, produce an equivalent
        circuit suitable for running on this backend.

        At a minimum it will ensure that compatible gates are used and that all two-
        qubit interactions are compatible with the backend's qubit architecture. At
        higher optimisation levels, further optimisations may be applied.

        This is a an abstract method which is implemented in the backend itself, and so
        is tailored to the backend's requirements.

        The default compilation passes for the :py:class:`IBMQBackend` and the

        :param optimisation_level: The level of optimisation to perform during
            compilation.
        :param timeout: Parameter for optimisation level 3, given in seconds.

            - Level 0 does the minimum required to solves the device constraints,
              without any optimisation.
            - Level 1 additionally performs some light optimisations.
            - Level 2 (the default) adds more computationally intensive optimisations
              that should give the best results from execution.
            - Level 3 re-synthesises the circuit using the computationally intensive
              `GreedyPauliSimp`. This will remove any barriers while optimising.


        :return: Compilation pass guaranteeing required predicates.
        """
        config: PulseBackendConfiguration = self._backend.configuration()
        props: Optional[BackendProperties] = self._backend.properties()
        return IBMQBackend.default_compilation_pass_offline(
            config, props, optimisation_level, timeout
        )

    @staticmethod
    def default_compilation_pass_offline(
        config: PulseBackendConfiguration,
        props: Optional[BackendProperties],
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        backend_info = IBMQBackend._get_backend_info(config, props)
        primitive_gates = _get_primitive_gates(_tk_gate_set(config))
        supports_rz = OpType.Rz in primitive_gates

        assert optimisation_level in range(4)
        passlist = [DecomposeBoxes()]
        # If you make changes to the default_compilation_pass,
        # then please update this page accordingly
        # https://docs.quantinuum.com/tket/extensions/pytket-qiskit/index.html#default-compilation
        # Edit this docs source file -> pytket-qiskit/docs/intro.txt
        if optimisation_level == 0:
            if supports_rz:
                # If the Rz gate is unsupported then the rebase should be skipped
                # This prevents an error when compiling to the stabilizer backend
                # where no TK1 replacement can be found for the rebase.
                passlist.append(IBMQBackend.rebase_pass_offline(primitive_gates))
        elif optimisation_level == 1:
            passlist.append(SynthesiseTket())
        elif optimisation_level == 2:
            passlist.append(FullPeepholeOptimise())
        elif optimisation_level == 3:
            passlist.append(RemoveBarriers())
            passlist.append(AutoRebase({OpType.CX, OpType.H, OpType.Rz}))
            passlist.append(
                GreedyPauliSimp(thread_timeout=timeout, only_reduce=True, trials=10)
            )
        arch = backend_info.architecture
        assert arch is not None
        if not isinstance(arch, FullyConnected):
            passlist.append(AutoRebase(primitive_gates))
            passlist.append(
                CustomPass(
                    _gen_lightsabre_transformation(arch, optimisation_level),
                    "lightsabre",
                )
            )
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
        if optimisation_level == 3:
            passlist.append(SynthesiseTket())
        passlist.extend(
            [IBMQBackend.rebase_pass_offline(primitive_gates), RemoveRedundancies()]
        )
        return SequencePass(passlist)

    def get_compiled_circuit(
        self, circuit: Circuit, optimisation_level: int = 2, timeout: int = 300
    ) -> Circuit:
        """
        Return a single circuit compiled with :py:meth:`default_compilation_pass`.

        :param optimisation_level: Allows values of 0, 1, 2 or 3, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :type optimisation_level: int, optional
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass `GreedyPauliSimp`. Increase for
            optimising larger circuits.
        :type timeout: int, optional

        :return: An optimised quantum circuit
        :rtype: Circuit
        """
        return_circuit = circuit.copy()
        if optimisation_level == 3 and circuit.n_gates_of_type(OpType.Barrier) > 0:
            warnings.warn(
                "Barrier operations in this circuit will be removed when using "
                "optimisation level 3."
            )
        self.default_compilation_pass(optimisation_level, timeout).apply(return_circuit)
        return return_circuit

    def get_compiled_circuits(
        self,
        circuits: Sequence[Circuit],
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> list[Circuit]:
        """Compile a sequence of circuits with :py:meth:`default_compilation_pass`
        and return the list of compiled circuits (does not act in place).

        As well as applying a degree of optimisation (controlled by the
        `optimisation_level` parameter), this method tries to ensure that the circuits
        can be run on the backend (i.e. successfully passed to
        :py:meth:`process_circuits`), for example by rebasing to the supported gate set,
        or routing to match the connectivity of the device. However, this is not always
        possible, for example if the circuit contains classical operations that are not
        supported by the backend. You may use :py:meth:`valid_circuit` to check whether
        the circuit meets the backend's requirements after compilation. This validity
        check is included in :py:meth:`process_circuits` by default, before any circuits
        are submitted to the backend.

        If the validity check fails, you can obtain more information about the failure
        by iterating through the predicates in the `required_predicates` property of the
        backend, and running the :py:meth:`verify` method on each in turn with your
        circuit.

        :param circuits: The circuits to compile.
        :type circuit: Sequence[Circuit]
        :param optimisation_level: The level of optimisation to perform during
            compilation. See :py:meth:`default_compilation_pass` for a description of
            the different levels (0, 1, 2 or 3). Defaults to 2.
        :type optimisation_level: int, optional
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass `GreedyPauliSimp`. Increase for
            optimising larger circuits.
        :type timeout: int, optional
        :return: Compiled circuits.
        :rtype: List[Circuit]
        """
        return [
            self.get_compiled_circuit(c, optimisation_level, timeout) for c in circuits
        ]

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        # IBMQ job ID, index, number of bits, post-processing circuit
        return (str, int, int, str)

    def rebase_pass(self) -> BasePass:
        return IBMQBackend.rebase_pass_offline(self._primitive_gates)

    @staticmethod
    def rebase_pass_offline(primitive_gates: set[OpType]) -> BasePass:
        return AutoRebase(primitive_gates)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[Optional[int]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        :Keyword Arguments:
            * `postprocess`:
                apply end-of-circuit simplifications and classical
                postprocessing to improve fidelity of results (bool, default False)
            * `simplify_initial`:
                apply the pytket ``SimplifyInitial`` pass to improve
                fidelity of results assuming all qubits initialized to zero
                (bool, default False)
            * `sampler_options`:
                A customised `qiskit_ibm_runtime` `SamplerOptions` instance. See
                the Qiskit documentation at
                https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.options.SamplerOptions
                for details and default values.
        """
        circuits = list(circuits)

        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        handle_list: list[Optional[ResultHandle]] = [None] * len(circuits)
        circuit_batches, batch_order = _batch_circuits(circuits, n_shots_list)

        postprocess = kwargs.get("postprocess", False)
        simplify_initial = kwargs.get("simplify_initial", False)

        sampler_options: SamplerOptions = kwargs.get("sampler_options")
        if sampler_options is None:
            sampler_options = self._sampler_options

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
                            batch_chunk[i].n_bits,
                            ppcirc_strs[i],
                        )
                else:
                    sampler = SamplerV2(mode=self._session, options=sampler_options)
                    job = sampler.run(qcs, shots=n_shots)
                    job_id = job.job_id()
                    for i, ind in enumerate(indices_chunk):
                        handle_list[ind] = ResultHandle(
                            job_id, i, qcs[i].num_clbits, ppcirc_strs[i]
                        )
            batch_id += 1  # noqa: SIM113
        for handle in handle_list:
            assert handle is not None
            self._cache[handle] = dict()
        return cast(list[ResultHandle], handle_list)

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
        return CircuitStatus(_STATUS_MAP[ibmstatus], ibmstatus)

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
        jobid, index, n_bits, ppcirc_str = handle
        ppcirc_rep = json.loads(ppcirc_str)
        ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None
        cache_key = (jobid, index)
        if cache_key not in self._ibm_res_cache:
            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                shots: int
                shots, _ = literal_eval(jobid[len(_DEBUG_HANDLE_PREFIX) :])
                res = _gen_debug_results(n_bits, shots)
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
                    while status not in ["DONE", "CANCELLED", "ERROR"]:
                        status = job.status()
                        sleep(10)

                res = job.result(timeout=kwargs.get("timeout"))
            assert isinstance(res, PrimitiveResult)
            for circ_index, pub_result in enumerate(res._pub_results):
                data = pub_result.data
                c_regs = OrderedDict(
                    (reg_name, data.__getattribute__(reg_name).num_bits)
                    for reg_name in sorted(data.keys())
                )
                readouts = BitArray.concatenate_bits(
                    [data.__getattribute__(reg_name) for reg_name in c_regs]
                ).array
                self._ibm_res_cache[(jobid, circ_index)] = (
                    Counter(_int_from_readout(readout) for readout in readouts),
                    list(
                        itertools.chain.from_iterable(
                            [Bit(reg_name, i) for i in range(reg_size)]
                            for reg_name, reg_size in c_regs.items()
                        )
                    ),
                )

        counts, c_bits = self._ibm_res_cache[cache_key]  # Counter[int], list[Bit]
        # Convert to `OutcomeArray`:
        tket_counts: Counter = Counter()
        for outcome_key, sample_count in counts.items():
            array = OutcomeArray.from_ints(
                ints=[outcome_key],
                width=n_bits,
                big_endian=False,
            )
            tket_counts[array] = sample_count
        # Convert to `BackendResult`:
        result = BackendResult(c_bits=c_bits, counts=tket_counts, ppcirc=ppcirc)

        self._cache[handle] = {"result": result}
        return result
