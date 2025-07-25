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

from qiskit_aer.backends import AerSimulator  # type: ignore

from pytket.extensions.qiskit import (
    AerBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQBackend,
)
from pytket.passes import BasePass
from qiskit.converters import circuit_to_dag, dag_to_circuit  # type: ignore
from qiskit.dagcircuit import DAGCircuit  # type: ignore
from qiskit.providers import BackendV2  # type: ignore
from qiskit.transpiler.basepasses import BasePass as qBasePass  # type: ignore
from qiskit.transpiler.basepasses import TransformationPass

from .qiskit_convert import qiskit_to_tk, tk_to_qiskit


class TketPass(TransformationPass):
    """The tket compiler to be plugged in to the Qiskit compilation sequence"""

    def __init__(self, tket_pass: BasePass):
        """Wraps a pytket compiler pass as a
        :py:class:`qiskit.transpiler.TransformationPass`. A
        :py:class:`qiskit.dagcircuit.DAGCircuit` is converted to a pytket
        :py:class:`~pytket._tket.circuit.Circuit`. `tket_pass` will be run and the result is converted back.

        :param tket_pass: The pytket compiler pass to run
        """
        qBasePass.__init__(self)
        self._pass = tket_pass

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run a preconfigured optimisation pass on the circuit and route for the given
        backend.

        :param dag: The circuit to optimise and route

        :return: The modified circuit
        """
        qc = dag_to_circuit(dag)
        old_parameters = qc.parameters
        circ = qiskit_to_tk(qc)
        self._pass.apply(circ)
        qc = tk_to_qiskit(circ)
        new_param_lookup = {p._symbol_expr: p for p in qc.parameters}  # noqa: SLF001
        subs_map = {new_param_lookup[p._symbol_expr]: p for p in old_parameters}  # noqa: SLF001
        qc.assign_parameters(subs_map, inplace=True)
        newdag = circuit_to_dag(qc)
        newdag.name = dag.name
        return newdag


class TketAutoPass(TketPass):
    """The tket compiler to be plugged in to the Qiskit compilation sequence"""

    _aer_backend_map = {  # noqa: RUF012
        "aer_simulator": AerBackend,
        "aer_simulator_statevector": AerStateBackend,
        "aer_simulator_unitary": AerUnitaryBackend,
    }

    def __init__(
        self,
        backend: BackendV2,
        optimisation_level: int = 2,
        instance: str | None = None,
        token: str | None = None,
    ):
        """Identifies a Qiskit backend and provides the corresponding default
        compilation pass from pytket as a
        :py:class:`qiskit.transpiler.TransformationPass`.

        :param backend: The Qiskit backend to target. Accepts Aer or IBMQ backends.
        :param optimisation_level: The level of optimisation to perform during
            compilation. Level 0 just solves the device constraints without
            optimising. Level 1 additionally performs some light optimisations.
            Level 2 adds more computationally intensive optimisations. Defaults to 2.
        :param instance: Instance for the :py:class:`~qiskit_ibm_runtime.QiskitRuntimeService`.
        :param token: Authentication token to use the :py:class:`~qiskit_ibm_runtime.QiskitRuntimeService`.
        """
        if isinstance(backend, AerSimulator):
            tk_backend = self._aer_backend_map[backend.name]()
        else:
            tk_backend = IBMQBackend(backend.name, instance=instance, token=token)
        super().__init__(tk_backend.default_compilation_pass(optimisation_level))
