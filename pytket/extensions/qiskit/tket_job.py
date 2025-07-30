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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pytket.backends import ResultHandle, StatusEnum
from pytket.backends.backend import Backend, KwargTypes
from pytket.circuit import Bit, Qubit, UnitID
from pytket.extensions.qiskit.result_convert import (
    _get_header_info,
    backendresult_to_qiskit_resultdata,
)
from qiskit.providers import JobStatus, JobV1  # type: ignore
from qiskit.result import Result  # type: ignore

if TYPE_CHECKING:
    from pytket.extensions.qiskit.tket_backend import TketBackend


@dataclass
class JobInfo:
    circuit_name: str
    qbits: list[Qubit]
    cbits: list[Bit]
    n_shots: int | None


class TketJob(JobV1):
    """TketJob wraps a :py:class:`~pytket.backends.resulthandle.ResultHandle` list as a
    :py:class:`qiskit.providers.JobV1`"""

    def __init__(
        self,
        backend: "TketBackend",
        handles: list[ResultHandle],
        jobinfos: list[JobInfo],
        final_maps: list[None] | list[dict[UnitID, UnitID]],
    ):
        """Initializes the asynchronous job."""

        super().__init__(backend, str(handles[0]))
        self._handles = handles
        self._jobinfos = jobinfos
        self._result: Result | None = None
        self._final_maps = final_maps

    @property
    def _pytket_backend(self) -> Backend:
        return cast("TketBackend", self._backend)._backend  # noqa: SLF001

    def submit(self) -> None:
        # Circuits have already been submitted before obtaining the job
        pass

    def result(self, **kwargs: KwargTypes) -> Result:
        if self._result is not None:
            return self._result
        result_list = []
        for h, jobinfo, fm in zip(
            self._handles, self._jobinfos, self._final_maps, strict=False
        ):
            tk_result = self._pytket_backend.get_result(h)
            creg_sizes, clbit_labels = _get_header_info(jobinfo.cbits)  # type: ignore
            qreg_sizes, qubit_labels = _get_header_info(jobinfo.qbits)  # type: ignore
            memory_slots = sum(size for _, size in creg_sizes)
            result_list.append(
                {
                    "shots": jobinfo.n_shots,
                    "success": True,
                    "data": backendresult_to_qiskit_resultdata(
                        tk_result,
                        jobinfo.cbits,  # type: ignore
                        jobinfo.qbits,  # type: ignore
                        fm,
                    ),
                    "header": {
                        "name": jobinfo.circuit_name,
                        "creg_sizes": creg_sizes,
                        "memory_slots": memory_slots,
                        "clbit_labels": clbit_labels,
                        "qreg_sizes": qreg_sizes,
                        "qubit_labels": qubit_labels,
                    },
                }
            )
            self._pytket_backend.pop_result(h)
        self._result = Result.from_dict(
            {
                "results": result_list,
                "backend_name": self._backend.name,
                "backend_version": self._backend.backend_version,
                "job_id": self._job_id,
                "qobj_id": ", ".join(str(hand) for hand in self._handles),
                "success": True,
            }
        )
        return self._result

    def cancel(self) -> None:
        for h in self._handles:
            self._pytket_backend.cancel(h)

    def status(self) -> Any:
        status_list = [self._pytket_backend.circuit_status(h) for h in self._handles]
        if any(s.status == StatusEnum.RUNNING for s in status_list):
            return JobStatus.RUNNING
        if any(s.status == StatusEnum.ERROR for s in status_list):
            return JobStatus.ERROR
        if any(s.status == StatusEnum.CANCELLED for s in status_list):
            return JobStatus.CANCELLED
        if all(s.status == StatusEnum.COMPLETED for s in status_list):
            return JobStatus.DONE
        return JobStatus.INITIALIZING
