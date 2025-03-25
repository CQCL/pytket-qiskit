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


from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from typing import (
    Any,
    Optional,
)

import numpy as np

from pytket.backends.backendresult import BackendResult
from pytket.circuit import Bit, Qubit, UnitID
from pytket.utils.outcomearray import OutcomeArray
from qiskit.result import Result  # type: ignore
from qiskit.result.models import ExperimentResult  # type: ignore


def _get_registers_from_uids(uids: list[UnitID]) -> dict[str, set[UnitID]]:
    registers: dict[str, set[UnitID]] = defaultdict(set)
    for uid in uids:
        registers[uid.reg_name].add(uid)
    return registers


LabelsType = list[tuple[str, int]]


def _get_header_info(uids: list[UnitID]) -> tuple[LabelsType, LabelsType]:
    registers = _get_registers_from_uids(uids)
    reg_sizes = [(name, max(uids).index[0] + 1) for name, uids in registers.items()]
    reg_labels = [
        (name, uid.index[0]) for name, indices in registers.items() for uid in uids
    ]

    return reg_sizes, reg_labels


def _qiskit_ordered_uids(uids: list[UnitID]) -> list[UnitID]:
    registers = _get_registers_from_uids(uids)
    names = sorted(registers.keys())
    return [uid for name in names for uid in sorted(registers[name], reverse=True)]


def _hex_to_outar(hexes: Sequence[str], width: int) -> OutcomeArray:
    ints = [int(hexst, 16) for hexst in hexes]
    return OutcomeArray.from_ints(ints, width, big_endian=False)


# An empty ExperimentResult can be an empty dict, but it can also be a dict
# filled with empty values.
def _result_is_empty_shots(result: ExperimentResult) -> bool:
    if not result.shots > 0:
        # 0-shots results don't count as empty; they are simply ignored
        return False

    datadict = result.data.to_dict()
    return bool(
        len(datadict) == 0
        or "memory" in datadict
        and len(datadict["memory"]) == 0
        or "counts" in datadict
        and len(datadict["counts"]) == 0
    )


# In some cases, Qiskit returns a result with fields we don't expect -
# for example, a circuit with classical bits run on AerStateBackend will
# return counts (whether or not there were measurements). The include_foo
# arguments should be set based on what the backend supports.
def qiskit_experimentresult_to_backendresult(
    result: ExperimentResult,
    include_counts: bool = True,
    include_shots: bool = True,
    include_state: bool = True,
    include_unitary: bool = True,
    include_density_matrix: bool = True,
) -> BackendResult:
    if not result.success:
        raise RuntimeError(result.status)

    header = result.header
    width = header.memory_slots

    c_bits, q_bits = None, None
    if hasattr(header, "creg_sizes"):
        c_bits = []
        for name, size in header.creg_sizes:
            for index in range(size):
                c_bits.append(Bit(name, index))
    if hasattr(header, "qreg_sizes"):
        q_bits = []
        for name, size in header.qreg_sizes:
            for index in range(size):
                q_bits.append(Qubit(name, index))

    shots, counts, state, unitary, density_matrix = (None,) * 5
    datadict = result.data.to_dict()
    if _result_is_empty_shots(result) and include_shots:
        n_bits = len(c_bits) if c_bits else 0
        shots = OutcomeArray.from_readouts(
            np.zeros((result.shots, n_bits), dtype=np.uint8)
        )
    else:
        if "memory" in datadict and include_shots:
            memory = datadict["memory"]
            shots = _hex_to_outar(memory, width)
        elif "counts" in datadict and include_counts:
            qis_counts = datadict["counts"]
            counts = Counter(
                dict(
                    (_hex_to_outar([hexst], width), count)
                    for hexst, count in qis_counts.items()
                )
            )

        if "statevector" in datadict and include_state:
            state = datadict["statevector"].reverse_qargs().data

        if "unitary" in datadict and include_unitary:
            unitary = datadict["unitary"].reverse_qargs().data

        if "density_matrix" in datadict and include_density_matrix:
            density_matrix = datadict["density_matrix"].reverse_qargs().data

    return BackendResult(
        c_bits=c_bits,
        q_bits=q_bits,
        shots=shots,
        counts=counts,
        state=state,
        unitary=unitary,
        density_matrix=density_matrix,
        ppcirc=None,
    )


def qiskit_result_to_backendresult(
    res: Result,
    include_counts: bool = True,
    include_shots: bool = True,
    include_state: bool = True,
    include_unitary: bool = True,
    include_density_matrix: bool = True,
) -> Iterator[BackendResult]:
    for result in res.results:
        yield qiskit_experimentresult_to_backendresult(
            result,
            include_counts,
            include_shots,
            include_state,
            include_unitary,
            include_density_matrix,
        )


def backendresult_to_qiskit_resultdata(
    res: BackendResult,
    cbits: list[UnitID],
    qbits: list[UnitID],
    final_map: Optional[dict[UnitID, UnitID]],
) -> dict[str, Any]:
    data: dict[str, Any] = dict()
    if res.contains_state_results:
        qbits = _qiskit_ordered_uids(qbits)
        qbits.sort(reverse=True)
        if final_map:
            qbits = [final_map[q] for q in qbits]
        stored_res = res.get_result(qbits)
        if stored_res.state is not None:
            data["statevector"] = stored_res.state
        if stored_res.unitary is not None:
            data["unitary"] = stored_res.unitary
    if res.contains_measured_results:
        cbits = _qiskit_ordered_uids(cbits)
        if final_map:
            cbits = [final_map[c] for c in cbits]
        stored_res = res.get_result(cbits)
        if stored_res.shots is not None:
            data["memory"] = [hex(i) for i in stored_res.shots.to_intlist()]
            data["counts"] = dict(Counter(data["memory"]))
        elif stored_res.counts is not None:
            data["counts"] = {
                hex(i.to_intlist()[0]): f for i, f in stored_res.counts.items()
            }
    return data
