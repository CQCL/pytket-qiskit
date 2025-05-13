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


"""Methods to allow conversion between Qiskit and pytket circuit classes"""

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    cast,
)
from uuid import UUID

import numpy as np
import sympy
from numpy.typing import NDArray
from qiskit_ibm_runtime.models.backend_configuration import (  # type: ignore
    QasmBackendConfiguration,
)
from qiskit_ibm_runtime.models.backend_properties import (  # type: ignore
    BackendProperties,
)
from symengine import sympify  # type: ignore
from symengine.lib import symengine_wrapper  # type: ignore

import qiskit.circuit.library.standard_gates as qiskit_gates  # type: ignore
from pytket.architecture import Architecture, FullyConnected
from pytket.circuit import (
    Bit,
    CircBox,
    Circuit,
    Conditional,
    Node,
    Op,
    OpType,
    QControlBox,
    Qubit,
    StatePreparationBox,
    Unitary1qBox,
    Unitary2qBox,
    Unitary3qBox,
    UnitType,
)
from pytket.circuit.logic_exp import reg_eq, reg_neq
from pytket.passes import AutoRebase
from pytket.pauli import Pauli, QubitPauliString
from pytket.unit_id import _TEMP_BIT_NAME
from pytket.utils import (
    QubitPauliOperator,
    gen_term_sequence_circuit,
    permute_rows_cols_in_unitary,
)
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit import (
    Barrier,
    Clbit,
    ControlledGate,
    Gate,
    IfElseOp,
    Instruction,
    InstructionSet,
    Measure,
    Parameter,
    ParameterExpression,
    Reset,
)
from qiskit.circuit import Qubit as QCQubit
from qiskit.circuit.library import (
    CRYGate,
    Initialize,
    PauliEvolutionGate,
    RYGate,
    StatePreparation,
    UnitaryGate,
)

if TYPE_CHECKING:
    from qiskit_ibm_runtime.ibm_backend import IBMBackend  # type: ignore
    from qiskit_ibm_runtime.models.backend_properties import Nduv

    from pytket.circuit import UnitID
    from pytket.unit_id import BitRegister
    from qiskit.circuit.quantumcircuitdata import QuantumCircuitData  # type: ignore

_qiskit_gates_1q = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.HGate: OpType.H,
    qiskit_gates.IGate: OpType.noop,
    qiskit_gates.PhaseGate: OpType.U1,
    qiskit_gates.RGate: OpType.PhasedX,
    qiskit_gates.RXGate: OpType.Rx,
    qiskit_gates.RYGate: OpType.Ry,
    qiskit_gates.RZGate: OpType.Rz,
    qiskit_gates.SdgGate: OpType.Sdg,
    qiskit_gates.SGate: OpType.S,
    qiskit_gates.SXdgGate: OpType.SXdg,
    qiskit_gates.SXGate: OpType.SX,
    qiskit_gates.TdgGate: OpType.Tdg,
    qiskit_gates.TGate: OpType.T,
    qiskit_gates.U1Gate: OpType.U1,
    qiskit_gates.U2Gate: OpType.U2,
    qiskit_gates.U3Gate: OpType.U3,
    qiskit_gates.UGate: OpType.U3,
    qiskit_gates.XGate: OpType.X,
    qiskit_gates.YGate: OpType.Y,
    qiskit_gates.ZGate: OpType.Z,
}

_qiskit_gates_2q = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.CHGate: OpType.CH,
    qiskit_gates.CPhaseGate: OpType.CU1,
    qiskit_gates.CRXGate: OpType.CRx,
    qiskit_gates.CRYGate: OpType.CRy,
    qiskit_gates.CRZGate: OpType.CRz,
    qiskit_gates.CUGate: OpType.CU3,
    qiskit_gates.CU1Gate: OpType.CU1,
    qiskit_gates.CU3Gate: OpType.CU3,
    qiskit_gates.CXGate: OpType.CX,
    qiskit_gates.CSXGate: OpType.CSX,
    qiskit_gates.CYGate: OpType.CY,
    qiskit_gates.CZGate: OpType.CZ,
    qiskit_gates.ECRGate: OpType.ECR,
    qiskit_gates.iSwapGate: OpType.ISWAPMax,
    qiskit_gates.RXXGate: OpType.XXPhase,
    qiskit_gates.RYYGate: OpType.YYPhase,
    qiskit_gates.RZZGate: OpType.ZZPhase,
    qiskit_gates.SwapGate: OpType.SWAP,
}

_qiskit_gates_other = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.C3XGate: OpType.CnX,
    qiskit_gates.C4XGate: OpType.CnX,
    qiskit_gates.CCXGate: OpType.CCX,
    qiskit_gates.CCZGate: OpType.CnZ,
    qiskit_gates.CSwapGate: OpType.CSWAP,
    # Multi-controlled gates (qiskit expects a list of controls followed by the target):
    qiskit_gates.MCXGate: OpType.CnX,
    qiskit_gates.MCXGrayCode: OpType.CnX,
    qiskit_gates.MCXRecursive: OpType.CnX,
    qiskit_gates.MCXVChain: OpType.CnX,
    # Special types:
    Barrier: OpType.Barrier,
    Instruction: OpType.CircBox,
    Gate: OpType.CircBox,
    Measure: OpType.Measure,
    Reset: OpType.Reset,
    Initialize: OpType.StatePreparationBox,
    StatePreparation: OpType.StatePreparationBox,
}

_known_qiskit_gate = {**_qiskit_gates_1q, **_qiskit_gates_2q, **_qiskit_gates_other}

# Some qiskit gates are aliases (e.g. UGate and U3Gate).
# In such cases this reversal will select one or the other.
_known_qiskit_gate_rev = {v: k for k, v in _known_qiskit_gate.items()}

# Ensure U3 maps to UGate. (U3Gate deprecated in Qiskit but equivalent.)
_known_qiskit_gate_rev[OpType.U3] = qiskit_gates.UGate

# There is a bijective mapping, but requires some special parameter conversions
# tk1(a, b, c) = U(b, a-1/2, c+1/2) + phase(-(a+c)/2)
_known_qiskit_gate_rev[OpType.TK1] = qiskit_gates.UGate

# some gates are only equal up to global phase, support their conversion
# from tket -> qiskit
_known_gate_rev_phase = {
    optype: (qgate, 0.0) for optype, qgate in _known_qiskit_gate_rev.items()
}

_known_gate_rev_phase[OpType.V] = (qiskit_gates.SXGate, -0.25)
_known_gate_rev_phase[OpType.Vdg] = (qiskit_gates.SXdgGate, 0.25)

# use minor signature hacks to figure out the string names of qiskit Gate objects
_gate_str_2_optype: dict[str, OpType] = dict()  # noqa: C408
for gate, optype in _known_qiskit_gate.items():
    if gate in (
        UnitaryGate,
        Instruction,
        Gate,
        qiskit_gates.MCXGate,  # all of these have special (c*n)x names
        qiskit_gates.MCXGrayCode,
        qiskit_gates.MCXRecursive,
        qiskit_gates.MCXVChain,
    ):
        continue
    sig = signature(gate.__init__)
    # name is only a property of the instance, not the class
    # so initialize with the correct number of dummy variables
    n_params = len([p for p in sig.parameters.values() if p.default is p.empty]) - 1
    name = gate(*([1] * n_params)).name
    _gate_str_2_optype[name] = optype

_gate_str_2_optype_rev = {v: k for k, v in _gate_str_2_optype.items()}
# the aliasing of the name is ok in the reverse map
_gate_str_2_optype_rev[OpType.Unitary1qBox] = "unitary"


def _tk_gate_set(config: QasmBackendConfiguration) -> set[OpType]:
    """Set of tket gate types supported by the qiskit backend"""
    if config.simulator:
        gate_set = {
            _gate_str_2_optype[gate_str]
            for gate_str in config.basis_gates
            if gate_str in _gate_str_2_optype
        }.union({OpType.Measure, OpType.Reset, OpType.Barrier})
        return gate_set  # noqa: RET504

    return {
        _gate_str_2_optype[gate_str]
        for gate_str in config.supported_instructions
        if gate_str in _gate_str_2_optype
    }


def _qpo_from_peg(peg: PauliEvolutionGate, qubits: list[Qubit]) -> QubitPauliOperator:
    op = peg.operator
    t = peg.params[0]
    qpodict = {}
    for p, c in zip(op.paulis, op.coeffs, strict=False):
        if np.iscomplex(c):
            raise ValueError(f"Coefficient for Pauli {p} is non-real.")
        coeff = param_to_tk(t) * c
        qpslist = []
        pstr = p.to_label()
        for a in pstr:
            if a == "X":
                qpslist.append(Pauli.X)
            elif a == "Y":
                qpslist.append(Pauli.Y)
            elif a == "Z":
                qpslist.append(Pauli.Z)
            else:
                assert a == "I"
                qpslist.append(Pauli.I)
        qpodict[QubitPauliString(qubits, qpslist)] = coeff
    return QubitPauliOperator(qpodict)


def _string_to_circuit(
    circuit_string: str,
    n_qubits: int,
    qiskit_prep: Initialize | StatePreparation,
) -> Circuit:
    """Helper function to generate circuits for Initialize
    and StatePreparation objects built with strings"""

    circ = Circuit(n_qubits)
    # Check if Instruction is Initialize or Statepreparation
    # If Initialize, add resets
    if isinstance(qiskit_prep, Initialize):
        for qubit in circ.qubits:
            circ.Reset(qubit)

    # We iterate through the string in reverse to add the
    # gates in the correct order (endian-ness).
    for qubit_index, character in enumerate(reversed(circuit_string)):
        match character:
            case "0":
                pass
            case "1":
                circ.X(qubit_index)
            case "+":
                circ.H(qubit_index)
            case "-":
                circ.X(qubit_index)
                circ.H(qubit_index)
            case "r":
                circ.H(qubit_index)
                circ.S(qubit_index)
            case "l":
                circ.H(qubit_index)
                circ.Sdg(qubit_index)
            case _:
                raise ValueError(
                    f"Cannot parse string for character {character}. "  # noqa: ISC003
                    + "The supported characters are {'0', '1', '+', '-', 'r', 'l'}."
                )

    return circ


def _get_pytket_ctrl_state(bitstring: str, n_bits: int) -> tuple[bool, ...]:
    "Converts a little endian string '001'=1 (LE) to (1, 0, 0)."
    assert set(bitstring).issubset({"0", "1"})
    padded_bitstring = bitstring.zfill(n_bits)
    pytket_ctrl_state = reversed([bool(int(b)) for b in padded_bitstring])
    return tuple(pytket_ctrl_state)


def _all_bits_set(integer: int, n_bits: int) -> bool:
    return integer.bit_count() == n_bits


def _get_controlled_tket_optype(c_gate: ControlledGate) -> OpType:
    """Get a pytket controlled OpType from a qiskit ControlledGate."""

    # If the control state is not "all |1>", use QControlBox
    if not _all_bits_set(c_gate.ctrl_state, c_gate.num_ctrl_qubits):
        return OpType.QControlBox

    if c_gate.base_class in _known_qiskit_gate:
        # First we check if the gate is in _known_qiskit_gate
        # this avoids CZ being converted to CnZ
        return _known_qiskit_gate[c_gate.base_class]

    match c_gate.base_gate.base_class:
        case qiskit_gates.RYGate:
            return OpType.CnRy
        case qiskit_gates.YGate:
            return OpType.CnY
        case qiskit_gates.ZGate:
            return OpType.CnZ
        case _:
            if (
                c_gate.base_gate.base_class in _known_qiskit_gate
                or c_gate.base_gate.base_class is UnitaryGate
            ):
                return OpType.QControlBox
            raise NotImplementedError(
                "Conversion of qiskit ControlledGate with base gate "  # noqa: ISC003
                + f"base gate {c_gate.base_gate}"
                + "not implemented."
            )


def _optype_from_qiskit_instruction(instruction: Instruction) -> OpType:
    """Get a pytket OpType from a qiskit Instruction."""
    if isinstance(instruction, ControlledGate):
        return _get_controlled_tket_optype(instruction)
    try:
        optype = _known_qiskit_gate[instruction.base_class]
        return optype  # noqa: RET504
    except KeyError:
        raise NotImplementedError(  # noqa: B904
            f"Conversion of qiskit's {instruction.name} instruction is "  # noqa: ISC003
            + "currently unsupported by qiskit_to_tk. Consider "
            + "using QuantumCircuit.decompose() before attempting "
            + "conversion."
        )


UnitaryBox = Unitary1qBox | Unitary2qBox | Unitary3qBox


def _get_unitary_box(unitary: NDArray[np.complex128], num_qubits: int) -> UnitaryBox:
    match num_qubits:
        case 1:
            assert unitary.shape == (2, 2)
            return Unitary1qBox(unitary)
        case 2:
            assert unitary.shape == (4, 4)
            return Unitary2qBox(unitary)
        case 3:
            assert unitary.shape == (8, 8)
            return Unitary3qBox(unitary)
        case _:
            raise NotImplementedError(
                f"Conversion of {num_qubits}-qubit unitary gates not supported."
            )


def _get_qcontrol_box(c_gate: ControlledGate, params: list[float]) -> QControlBox:
    qiskit_ctrl_state: str = bin(c_gate.ctrl_state)[2:]
    pytket_ctrl_state: tuple[bool, ...] = _get_pytket_ctrl_state(
        bitstring=qiskit_ctrl_state, n_bits=c_gate.num_ctrl_qubits
    )
    if isinstance(c_gate.base_gate, UnitaryGate):
        unitary = c_gate.base_gate.params[0]
        # Here we reverse the order of the columns to correct for endianness.
        new_unitary: NDArray[np.complex128] = permute_rows_cols_in_unitary(
            matrix=unitary,
            permutation=tuple(reversed(range(c_gate.base_gate.num_qubits))),
        )
        base_op: Op = _get_unitary_box(new_unitary, c_gate.base_gate.num_qubits)
    else:
        base_tket_gate: OpType = _known_qiskit_gate[c_gate.base_gate.base_class]

        base_op: Op = Op.create(base_tket_gate, params)  # type: ignore

    return QControlBox(
        base_op, n_controls=c_gate.num_ctrl_qubits, control_state=pytket_ctrl_state
    )


def _add_state_preparation(
    tkc: Circuit, qubits: list[Qubit], prep: Initialize | StatePreparation
) -> None:
    """Handles different cases of Initialize and StatePreparation
    and appends the appropriate state preparation to a Circuit instance."""

    # Check how Initialize or StatePrep is constructed
    # With a string, an int or an array of amplitudes
    if len(prep.params) != 1:
        if isinstance(prep.params[0], str):
            # Parse string to get the right single qubit gates
            circuit_string: str = "".join(prep.params)
            circuit = _string_to_circuit(
                circuit_string, prep.num_qubits, qiskit_prep=prep
            )
            tkc.add_circuit(circuit, qubits)
        else:
            amplitude_array: NDArray[np.complex128] = np.array(prep.params)
            pytket_state_prep_box = StatePreparationBox(
                amplitude_array, with_initial_reset=(type(prep) is Initialize)
            )

            # Need to reverse qubits here (endian-ness)
            reversed_qubits = list(reversed(qubits))
            tkc.add_gate(pytket_state_prep_box, reversed_qubits)
    elif isinstance(prep.params[0], complex):
        # convert int to a binary string and apply X for |1>
        integer_parameter = int(prep.params[0].real)
        bit_string = bin(integer_parameter)[2:]
        circuit = _string_to_circuit(bit_string, prep.num_qubits, qiskit_prep=prep)
        tkc.add_circuit(circuit, qubits)
    else:
        raise TypeError(
            "Unrecognised type of Instruction.params "  # noqa: ISC003
            + "when trying to convert Initialize or StatePreparation instruction."
        )


def _build_circbox(instr: Instruction, circuit: QuantumCircuit) -> CircBox:
    qregs = [QuantumRegister(instr.num_qubits, "q")] if instr.num_qubits > 0 else []
    cregs = [ClassicalRegister(instr.num_clbits, "c")] if instr.num_clbits > 0 else []
    builder = CircuitBuilder(qregs, cregs)
    builder.add_qiskit_data(circuit, instr.definition)
    subc = builder.circuit()
    subc.name = instr.name
    return CircBox(subc)


def _build_rename_map(
    qcirc: QuantumCircuit,
    if_else_builder: "CircuitBuilder",
    outer_builder: "CircuitBuilder",
    qargs: list[QCQubit],
    cargs: list[Clbit],
) -> dict[Qubit | Bit, Qubit | Bit]:
    rename_map: dict[Qubit | Bit, Qubit | Bit] = {}
    for i, inner_q in enumerate(qcirc.qubits):
        rename_map[if_else_builder.qbmap[inner_q]] = outer_builder.qbmap[qargs[i]]
    for i, inner_c in enumerate(qcirc.clbits):
        rename_map[if_else_builder.cbmap[inner_c]] = outer_builder.cbmap[cargs[i]]
    return rename_map


# Used for handling of IfElseOp
# docs -> https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.IfElseOp
# Examples -> https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow
# pytket-qiskit issue -> https://github.com/CQCL/pytket-qiskit/issues/415
def _pytket_circuits_from_ifelseop(
    if_else_op: IfElseOp,
    outer_builder: "CircuitBuilder",
    qargs: list[QCQubit],
    cargs: list[Clbit],
) -> tuple[Circuit, Circuit | None]:
    # Extract the QuantumCircuit implementing true_body
    if_qc: QuantumCircuit = if_else_op.blocks[0]
    # if_qc can have empty qregs, so build from bits
    if_builder = CircuitBuilder.from_qiskit_units(
        if_qc.qubits, if_qc.clbits, if_qc.name, param_to_tk(if_qc.global_phase)
    )
    if_builder.add_qiskit_data(if_qc)
    if_circuit = if_builder.circuit()
    # if_circuit might have a different set of registers, which might
    # cause problems when appending to the outer circuit.
    # We rename the units to make sure the registers in the inner circuit
    # is a subset of the registers in the ourter circuit.
    if_rename_map = _build_rename_map(
        qcirc=if_qc,
        if_else_builder=if_builder,
        outer_builder=outer_builder,
        qargs=qargs,
        cargs=cargs,
    )
    if_circuit.rename_units(if_rename_map)  # type: ignore
    if_circuit.name = "If"
    if_circuit.remove_blank_wires(
        keep_blank_classical_wires=False,
        remove_classical_only_at_end_of_register=False,
    )

    # The false_body arg is optional
    if len(if_else_op.blocks) == 2:  # noqa: PLR2004
        else_qc: QuantumCircuit = if_else_op.blocks[1]
        else_builder = CircuitBuilder.from_qiskit_units(
            else_qc.qubits,
            else_qc.clbits,
            else_qc.name,
            param_to_tk(else_qc.global_phase),
        )
        else_builder.add_qiskit_data(else_qc)
        else_circuit = else_builder.circuit()
        # else_circuit might have a different set of registers
        else_rename_map = _build_rename_map(
            else_qc,
            if_else_builder=else_builder,
            outer_builder=outer_builder,
            qargs=qargs,
            cargs=cargs,
        )
        else_circuit.rename_units(else_rename_map)  # type: ignore
        else_circuit.name = "Else"
        else_circuit.remove_blank_wires(
            keep_blank_classical_wires=False,
            remove_classical_only_at_end_of_register=False,
        )
        return if_circuit, else_circuit

    # If no false_body is specified IfElseOp.blocks is of length 1.
    # In this case we return a Circuit implementing true_body and None.
    return if_circuit, None


def _append_if_else_circuit(
    if_else_op: IfElseOp,
    outer_builder: "CircuitBuilder",
    bits: list[Bit],
    qargs: list[QCQubit],
    cargs: list[Clbit],
) -> None:
    # Get two pytket circuits which implement the true_body and false_body.
    if_circ, else_circ = _pytket_circuits_from_ifelseop(
        if_else_op, outer_builder, qargs, cargs
    )
    # else_circ can be None if no false_body is specified.
    if isinstance(if_else_op.condition[0], Clbit):
        if len(bits) != 1:
            raise NotImplementedError("Conditions on multiple bits not supported")
        outer_builder.tkc.add_circbox(
            circbox=CircBox(if_circ),
            args=if_circ.qubits + if_circ.bits,  # type: ignore
            condition_bits=bits,
            condition_value=if_else_op.condition[1],
        )
        # If we have an else_circ defined, add it to the circuit
        if else_circ is not None:
            outer_builder.tkc.add_circbox(
                circbox=CircBox(else_circ),
                args=else_circ.qubits + else_circ.bits,  # type: ignore
                condition_bits=bits,
                condition_value=1 ^ if_else_op.condition[1],
            )

    elif isinstance(if_else_op.condition[0], ClassicalRegister):
        pytket_bit_reg: BitRegister = outer_builder.tkc.get_c_register(
            if_else_op.condition[0].name
        )
        outer_builder.tkc.add_circbox(
            circbox=CircBox(if_circ),
            args=if_circ.qubits + if_circ.bits,  # type: ignore
            condition=reg_eq(pytket_bit_reg, if_else_op.condition[1]),
        )
        if else_circ is not None:
            outer_builder.tkc.add_circbox(
                circbox=CircBox(else_circ),
                args=else_circ.qubits + else_circ.bits,  # type: ignore
                condition=reg_neq(pytket_bit_reg, if_else_op.condition[1]),
            )
    else:
        raise TypeError(
            "Unrecognized type used to construct IfElseOp. Expected "  # noqa: ISC003
            + f"ClBit or ClassicalRegister, got {type(if_else_op.condition[0])}"
        )


class CircuitBuilder:
    def __init__(
        self,
        qregs: list[QuantumRegister],
        cregs: list[ClassicalRegister] | None = None,
        name: str | None = None,
        phase: sympy.Expr | None = None,
    ):
        self.qbmap = {}
        self.cbmap = {}
        if name is not None:
            self.tkc = Circuit(name=name)
        else:
            self.tkc = Circuit()
        if phase is not None:
            self.tkc.add_phase(phase)
        for reg in qregs:
            self.tkc.add_q_register(reg.name, len(reg))
            for i, qb in enumerate(reg):
                self.qbmap[qb] = Qubit(reg.name, i)
        self.cregmap = {}
        if cregs is not None:
            for reg in cregs:
                tk_reg = self.tkc.add_c_register(reg.name, len(reg))
                self.cregmap.update({reg: tk_reg})
                for i, cb in enumerate(reg):
                    self.cbmap[cb] = Bit(reg.name, i)

    @classmethod
    def from_qiskit_units(
        cls,
        qubits: list[QCQubit],
        bits: list[Clbit],
        name: str | None = None,
        phase: sympy.Expr | None = None,
    ) -> "CircuitBuilder":
        """Construct a circuit builder from Qiskit's qubits and clbits"""
        builder = cls([], None, name, phase)
        for qb in qubits:
            tk_qb = Qubit(qb._register.name, qb._index)  # noqa: SLF001
            builder.tkc.add_qubit(tk_qb)
            builder.qbmap[qb] = tk_qb
        for cb in bits:
            tk_cb = Bit(cb._register.name, cb._index)  # noqa: SLF001
            builder.tkc.add_bit(tk_cb)
            builder.cbmap[cb] = tk_cb
        return builder

    def circuit(self) -> Circuit:
        return self.tkc

    def add_qiskit_data(  # noqa: PLR0912
        self,
        circuit: QuantumCircuit,
        data: Optional["QuantumCircuitData"] = None,
    ) -> None:
        data = data or circuit.data
        for datum in data:
            instr, qargs, cargs = datum.operation, datum.qubits, datum.clbits
            qubits: list[Qubit] = [self.qbmap[qbit] for qbit in qargs]
            bits: list[Bit] = [self.cbmap[bit] for bit in cargs]

            optype = None
            if type(instr) not in (PauliEvolutionGate, UnitaryGate, IfElseOp):
                # Handling of PauliEvolutionGate, UnitaryGate and IfElseOp below
                optype = _optype_from_qiskit_instruction(instruction=instr)

            if optype == OpType.QControlBox:
                params = [param_to_tk(p) for p in instr.base_gate.params]
                q_ctrl_box = _get_qcontrol_box(c_gate=instr, params=params)
                self.tkc.add_qcontrolbox(q_ctrl_box, qubits)

            elif optype == OpType.StatePreparationBox:
                # Append OpType found by stateprep helpers
                _add_state_preparation(self.tkc, qubits, instr)

            # Note: These IfElseOp/if_test type conditions are only handled
            # for single bit conditions and conditions on entire registers.
            elif type(instr) is IfElseOp:
                _append_if_else_circuit(
                    if_else_op=instr,
                    outer_builder=self,
                    bits=bits,
                    qargs=qargs,
                    cargs=cargs,
                )

            elif type(instr) is PauliEvolutionGate:
                qpo = _qpo_from_peg(instr, qubits)
                empty_circ = Circuit(len(qargs))
                circ = gen_term_sequence_circuit(qpo, empty_circ)
                ccbox = CircBox(circ)
                self.tkc.add_circbox(ccbox, qubits)

            elif type(instr) is UnitaryGate:
                unitary = cast("NDArray[np.complex128]", instr.params[0])
                if len(qubits) == 0:
                    # If the UnitaryGate acts on no qubits, we add a phase.
                    self.tkc.add_phase(np.angle(unitary[0][0]) / np.pi)
                else:
                    unitary_box = _get_unitary_box(
                        unitary=unitary, num_qubits=instr.num_qubits
                    )
                    self.tkc.add_gate(
                        unitary_box,
                        list(reversed(qubits)),
                    )

            elif optype == OpType.Barrier:
                self.tkc.add_barrier(qubits)

            elif optype == OpType.CircBox:
                circbox = _build_circbox(instr, circuit)
                self.tkc.add_circbox(circbox, qubits + bits)  # type: ignore

            elif optype == OpType.CU3 and type(instr) is qiskit_gates.CUGate:
                if instr.params[-1] == 0:
                    self.tkc.add_gate(
                        optype,
                        [param_to_tk(p) for p in instr.params[:-1]],
                        qubits,
                    )
                else:
                    raise NotImplementedError("CUGate with nonzero phase")
            else:
                params = [param_to_tk(p) for p in instr.params]
                self.tkc.add_gate(
                    optype,  # type: ignore
                    params,
                    qubits + bits,  # type: ignore
                )


def qiskit_to_tk(qcirc: QuantumCircuit, preserve_param_uuid: bool = False) -> Circuit:
    """
    Converts a qiskit :py:class:`qiskit.QuantumCircuit` to a pytket :py:class:`Circuit`.

    :param qcirc: A circuit to be converted
    :param preserve_param_uuid: Whether to preserve symbolic Parameter uuids
        by appending them to the tket Circuit symbol names as "_UUID:<uuid>".
        This can be useful if you want to reassign Parameters after conversion
        to tket and back, as it is necessary for Parameter object equality
        to be preserved.
    :return: The converted circuit
    """
    circ_name = qcirc.name
    # Parameter uses a hidden _uuid for equality check
    # we optionally preserve this in parameter name for later use
    if preserve_param_uuid:
        updates = {p: Parameter(f"{p.name}_UUID:{p._uuid}") for p in qcirc.parameters}  # noqa: SLF001
        qcirc = cast("QuantumCircuit", qcirc.assign_parameters(updates))

    builder = CircuitBuilder(
        qregs=qcirc.qregs,
        cregs=qcirc.cregs,
        name=circ_name,
        phase=param_to_tk(qcirc.global_phase),
    )
    builder.add_qiskit_data(qcirc)
    return builder.circuit()


def _get_qiskit_control_state(bool_list: list[bool]) -> str:
    return "".join(str(int(b)) for b in bool_list)[::-1]


def param_to_tk(p: float | ParameterExpression) -> sympy.Expr:
    if isinstance(p, ParameterExpression):
        symexpr = p._symbol_expr  # noqa: SLF001
        try:
            return symexpr._sympy_() / sympy.pi
        except AttributeError:
            return symexpr / sympy.pi
    else:
        return p / sympy.pi


def param_to_qiskit(
    p: sympy.Expr, symb_map: dict[Parameter, sympy.Symbol]
) -> float | ParameterExpression:
    ppi = p * sympy.pi
    if len(ppi.free_symbols) == 0:
        return float(ppi.evalf())
    return ParameterExpression(symb_map, sympify(ppi))


def _get_params(
    op: Op, symb_map: dict[Parameter, sympy.Symbol]
) -> list[float | ParameterExpression]:
    return [param_to_qiskit(p, symb_map) for p in op.params]


def _apply_qiskit_instruction(
    qcirc: QuantumCircuit,
    instruc: Instruction,
    qargs: Iterable[UnitType.qubit],  # type: ignore
    cargs: Iterable[Clbit] = None,  # type: ignore  # noqa: RUF013
    condition: tuple[ClassicalRegister | Clbit, int] | None = None,
) -> None:
    if condition is None:
        qcirc.append(instruc, qargs, cargs)
    else:
        with qcirc.if_test(condition):
            qcirc.append(instruc, qargs, cargs)


def _has_if_else(qc: QuantumCircuit) -> bool:
    """Check if a QuantumCircuit contains an IfElseOp."""
    return "if_else" in qc.count_ops()


def append_tk_command_to_qiskit(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915
    op: "Op",
    args: list["UnitID"],
    qcirc: QuantumCircuit,
    qregmap: dict[str, QuantumRegister],
    cregmap: dict[str, ClassicalRegister],
    symb_map: dict[Parameter, sympy.Symbol],
    range_preds: dict[Bit, tuple[list["UnitID"], int]],
    condition: tuple[ClassicalRegister | Clbit, int] | None = None,
) -> InstructionSet:
    optype = op.type
    if optype == OpType.Measure:
        qubit = args[0]
        bit = args[1]
        qb = qregmap[qubit.reg_name][qubit.index[0]]
        b = cregmap[bit.reg_name][bit.index[0]]
        # If the bit is storing a range predicate it should be invalidated:
        range_preds.pop(bit, None)  # type: ignore
        _apply_qiskit_instruction(qcirc, Measure(), [qb], [b], condition)
        return qcirc

    if optype == OpType.Reset:
        qb = qregmap[args[0].reg_name][args[0].index[0]]
        _apply_qiskit_instruction(qcirc, Reset(), qargs=[qb], condition=condition)
        return qcirc

    if optype in (OpType.CircBox, OpType.ExpBox, OpType.PauliExpBox, OpType.CustomGate):
        subcircuit = op.get_circuit()  # type: ignore
        subqc = tk_to_qiskit(subcircuit)
        qargs = []
        cargs = []
        for a in args:
            if a.type == UnitType.qubit:
                qargs.append(qregmap[a.reg_name][a.index[0]])
            else:
                cargs.append(cregmap[a.reg_name][a.index[0]])
        if optype == OpType.CustomGate:
            instruc = subqc.to_gate()
            instruc.name = op.get_name()
            _apply_qiskit_instruction(
                qcirc=qcirc, instruc=instruc, qargs=qargs, condition=condition
            )
        elif _has_if_else(subqc):
            # Detect control flow in CircBoxes and raise an error.
            raise NotImplementedError(
                "Conversion of CircBox(es) containing conditional"  # noqa: ISC003
                + " gates not currently supported by tk_to_qiskit"
            )
        else:
            instruc = subqc.to_instruction()
            _apply_qiskit_instruction(
                qcirc=qcirc, instruc=instruc, qargs=qargs, condition=condition
            )
        return qcirc

    if optype in (OpType.Unitary1qBox, OpType.Unitary2qBox, OpType.Unitary3qBox):
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        u = op.get_matrix()  # type: ignore
        unitary_gate = UnitaryGate(u, label="unitary")
        # Note reversal of qubits, to account for endianness (pytket unitaries are
        # ILO-BE == DLO-LE; qiskit unitaries are ILO-LE == DLO-BE).
        _apply_qiskit_instruction(
            qcirc, unitary_gate, qargs=list(reversed(qargs)), condition=condition
        )
        return qcirc

    if optype == OpType.StatePreparationBox:
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        statevector_array = op.get_statevector()  # type: ignore
        # check if the StatePreparationBox contains resets
        if op.with_initial_reset():  # type: ignore
            initializer = Initialize(statevector_array)
            _apply_qiskit_instruction(
                qcirc=qcirc,
                instruc=initializer,
                qargs=list(reversed(qargs)),
                condition=condition,
            )
            return qcirc
        qiskit_state_prep_box = StatePreparation(statevector_array)
        _apply_qiskit_instruction(
            qcirc=qcirc,
            instruc=qiskit_state_prep_box,
            qargs=list(reversed(qargs)),
            condition=condition,
        )
        return qcirc

    if optype == OpType.QControlBox:
        assert isinstance(op, QControlBox)
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        pytket_control_state: list[bool] = op.get_control_state_bits()
        qiskit_control_state: str = _get_qiskit_control_state(pytket_control_state)
        try:
            gatetype, phase = _known_gate_rev_phase[op.get_op().type]
        except KeyError:
            raise NotImplementedError(  # noqa: B904
                "Conversion of QControlBox with base gate"  # noqa: ISC003
                + f"{op.get_op()} not supported by tk_to_qiskit."
            )
        params = _get_params(op.get_op(), symb_map)
        operation = gatetype(*params)
        _apply_qiskit_instruction(
            qcirc=qcirc,
            instruc=operation.control(
                num_ctrl_qubits=op.get_n_controls(), ctrl_state=qiskit_control_state
            ),
            qargs=qargs,
            condition=condition,
        )
        return qcirc

    if optype == OpType.Barrier:
        if any(q.type == UnitType.bit for q in args):
            raise NotImplementedError(
                "Qiskit Barriers are not defined for classical bits."
            )
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        barr = Barrier(len(args))
        _apply_qiskit_instruction(qcirc, instruc=barr, qargs=qargs, condition=condition)
        return qcirc

    if optype == OpType.RangePredicate:
        if op.lower != op.upper:  # type: ignore
            raise NotImplementedError
        range_preds[args[-1]] = (args[:-1], op.lower)  # type: ignore
        # attach predicate to bit,
        # subsequent conditional will handle it
        return Instruction("", 0, 0, [])
    if optype == OpType.Conditional:
        assert isinstance(op, Conditional)
        if op.op.type == OpType.Conditional:
            # See https://github.com/CQCL/pytket-qiskit/issues/442
            raise NotImplementedError("Nested conditional not supported")
        if op.op.type == OpType.Phase:
            # conditional phase not supported
            return InstructionSet()
        if args[0] in range_preds:
            assert op.value == 1
            condition_bits, value = range_preds[args[0]]  # type: ignore
            args = condition_bits + args[1:]
            width = len(condition_bits)
        else:
            width = op.width
            value = op.value
        regname = args[0].reg_name
        for i, a in enumerate(args[:width]):  # noqa: B007
            if a.reg_name != regname:
                raise NotImplementedError("Conditions can only use a single register")
        if len(cregmap[regname]) == width:
            for i, a in enumerate(args[:width]):
                if a.index != [i]:
                    raise NotImplementedError(
                        """Conditions must be an entire register in\
 order or only one bit of one register"""
                    )
            condition = (cregmap[regname], value)
        elif width == 1:
            condition = (cregmap[regname][args[0].index[0]], value)
        else:
            raise NotImplementedError(
                """Conditions must be an entire register in\
order or only one bit of one register"""
            )
        instruction = append_tk_command_to_qiskit(
            op.op,
            args[width:],
            qcirc,
            qregmap,
            cregmap,
            symb_map,
            range_preds,
            condition=condition,
        )

        return instruction  # noqa: RET504
    # normal gates
    qargs = [qregmap[q.reg_name][q.index[0]] for q in args]

    if optype == OpType.CnX:
        _apply_qiskit_instruction(
            qcirc,
            qiskit_gates.MCXGate(num_ctrl_qubits=len(qargs) - 1),
            qargs=qargs,
            condition=condition,
        )
        return qcirc

    if optype == OpType.CnY:
        _apply_qiskit_instruction(
            qcirc=qcirc,
            instruc=qiskit_gates.YGate().control(len(qargs) - 1),
            qargs=qargs,
            condition=condition,
        )
        return qcirc
    if optype == OpType.CnZ:
        if len(qargs) == 2:  # noqa: PLR2004
            z_gate = qiskit_gates.CZGate()
        else:
            z_gate = qiskit_gates.ZGate().control(len(qargs) - 1)
            z_gate.name = "mcz"
        _apply_qiskit_instruction(
            qcirc=qcirc, instruc=z_gate, qargs=qargs, condition=condition
        )
        return qcirc

    if optype == OpType.CnRy:
        # might as well do a bit more checking
        assert len(op.params) == 1
        alpha = param_to_qiskit(op.params[0], symb_map)
        assert len(qargs) >= 2  # noqa: PLR2004
        if len(qargs) == 2:  # noqa: PLR2004
            # presumably more efficient; single control only
            new_gate = CRYGate(alpha)
        else:
            new_gate = RYGate(alpha).control(len(qargs) - 1)
        _apply_qiskit_instruction(qcirc, new_gate, qargs=qargs, condition=condition)
        return qcirc

    if optype == OpType.CU3:
        params = _get_params(op, symb_map) + [0]  # noqa: RUF005
        _apply_qiskit_instruction(
            qcirc, qiskit_gates.CUGate(*params), qargs=qargs, condition=condition
        )
        return qcirc

    if optype == OpType.TK1:
        params = _get_params(op, symb_map)
        half = ParameterExpression(symb_map, sympify(sympy.pi / 2))
        qcirc.global_phase += -params[0] / 2 - params[2] / 2
        _apply_qiskit_instruction(
            qcirc,
            qiskit_gates.UGate(params[1], params[0] - half, params[2] + half),
            qargs=qargs,
            condition=condition,
        )
        return qcirc

    if optype == OpType.Phase:
        params = _get_params(op, symb_map)
        assert len(params) == 1
        # TODO is there a way to make this conditional?
        qcirc.global_phase += params[0]
        return InstructionSet()

    # others are direct translations
    try:
        gatetype, phase = _known_gate_rev_phase[optype]
    except KeyError as error:
        raise NotImplementedError(
            "Cannot convert tket Op to Qiskit gate: " + op.get_name()
        ) from error
    params = _get_params(op, symb_map)
    g = gatetype(*params)
    if type(phase) is float:
        qcirc.global_phase += phase * np.pi
    else:
        qcirc.global_phase += sympify(phase * sympy.pi)
    _apply_qiskit_instruction(
        qcirc=qcirc,
        instruc=g,
        qargs=qargs,
        condition=condition,
    )
    return qcirc


# The set of tket gates that can be converted directly to qiskit gates
_supported_tket_gates = set(_known_gate_rev_phase.keys())

_additional_multi_controlled_gates = {OpType.CnY, OpType.CnZ, OpType.CnRy}

# tket gates which are protected from being decomposed in the rebase
_protected_tket_gates = (
    _supported_tket_gates
    | _additional_multi_controlled_gates
    | {
        OpType.Unitary1qBox,
        OpType.Unitary2qBox,
        OpType.Unitary3qBox,
        OpType.QControlBox,
    }
    | {OpType.CustomGate}
)

# This is a rebase to the set of tket gates which have an exact substitution in qiskit
supported_gate_rebase = AutoRebase(_protected_tket_gates)


def tk_to_qiskit(
    tkcirc: Circuit,
    replace_implicit_swaps: bool = False,
    perm_warning: bool = True,
) -> QuantumCircuit:
    """
    Converts a pytket :py:class:`Circuit` to a qiskit :py:class:`qiskit.QuantumCircuit`.

    In many cases there will be a qiskit gate to exactly replace each tket gate.
    If no exact replacement can be found for a part of the circuit then an equivalent
    circuit will be returned using the tket gates which are supported in qiskit.

    Note that implicit swaps in a pytket Circuit are not handled by default.
    Consider using the replace_implicit_swaps flag to replace these implicit swaps with
    SWAP gates.

    :param tkcirc: A :py:class:`Circuit` to be converted
    :param replace_implicit_swaps: Implement implicit permutation by adding SWAPs
        to the end of the circuit.
    :param perm_warning: Warn if an input circuit has implicit qubit permutations,
        and `replace_implicit_swaps` is `False`. True by default.
    :return: The converted circuit
    """
    tkc = tkcirc.copy()  # Make a local copy of tkcirc
    if replace_implicit_swaps:
        tkc.replace_implicit_wire_swaps()

    if tkcirc.has_implicit_wireswaps and perm_warning and not replace_implicit_swaps:
        warnings.warn(  # noqa: B028
            "The pytket Circuit contains implicit qubit permutations"  # noqa: ISC003
            + " which aren't handled by default."
            + " Consider using the replace_implicit_swaps flag in tk_to_qiskit or"
            + " replacing them using Circuit.replace_implicit_swaps()."
        )

    qcirc = QuantumCircuit(name=tkc.name)
    qreg_sizes: dict[str, int] = {}
    for qb in tkc.qubits:
        if len(qb.index) != 1:
            raise NotImplementedError("Qiskit registers must use a single index")
        if (qb.reg_name not in qreg_sizes) or (qb.index[0] >= qreg_sizes[qb.reg_name]):
            qreg_sizes.update({qb.reg_name: qb.index[0] + 1})
    c_regs = tkcirc.c_registers
    if set(bit for reg in c_regs for bit in reg) != set(tkcirc.bits):  # noqa: C401
        raise NotImplementedError("Bit registers must be singly indexed from zero")
    qregmap = {}
    for reg_name, size in qreg_sizes.items():
        qis_reg = QuantumRegister(size, reg_name)
        qregmap.update({reg_name: qis_reg})
        qcirc.add_register(qis_reg)
    cregmap = {}
    for c_reg in c_regs:
        if c_reg.name != _TEMP_BIT_NAME:
            qis_reg = ClassicalRegister(c_reg.size, c_reg.name)
            cregmap.update({c_reg.name: qis_reg})
            qcirc.add_register(qis_reg)
    symb_map = {Parameter(str(s)): s for s in tkc.free_symbols()}
    range_preds: dict[Bit, tuple[list[UnitID], int]] = dict()  # noqa: C408

    # Apply a rebase to the set of pytket gates which have replacements in qiskit
    supported_gate_rebase.apply(tkc)

    for command in tkc:
        append_tk_command_to_qiskit(
            command.op, command.args, qcirc, qregmap, cregmap, symb_map, range_preds
        )
    qcirc.global_phase += param_to_qiskit(tkc.phase, symb_map)

    # if UUID stored in name, set parameter uuids accordingly (see qiskit_to_tk)
    updates = dict()  # noqa: C408
    for p in qcirc.parameters:
        name_spl = p.name.split("_UUID:", 2)
        if len(name_spl) == 2:  # noqa: PLR2004
            p_name, uuid_str = name_spl
            uuid = UUID(uuid_str)
            # See Parameter.__init__() in qiskit/circuit/parameter.py.
            new_p = Parameter(p_name)
            new_p._uuid = uuid  # noqa: SLF001
            new_p._parameter_keys = frozenset(  # noqa: SLF001
                ((symengine_wrapper.Symbol(p_name), uuid),)
            )
            new_p._hash = hash((new_p._parameter_keys, new_p._symbol_expr))  # noqa: SLF001
            updates[p] = new_p
    qcirc.assign_parameters(updates, inplace=True)

    return qcirc


def process_characterisation(backend: "IBMBackend") -> dict[str, Any]:
    """Convert a :py:class:`qiskit_ibm_runtime.ibm_backend.IBMBackend` to a
    dictionary containing device Characteristics

    :param backend: A backend to be converted
    :return: A dictionary containing device characteristics
    """
    config = backend.configuration()
    props = backend.properties()
    return process_characterisation_from_config(config, props)


def process_characterisation_from_config(  # noqa: PLR0915
    config: QasmBackendConfiguration, properties: BackendProperties | None
) -> dict[str, Any]:
    """Obtain a dictionary containing device Characteristics given config and props.

    :param config: A IBMQ configuration object
    :param properties: An optional IBMQ properties object
    :return: A dictionary containing device characteristics
    """

    # TODO explicitly check for and separate 1 and 2 qubit gates
    def return_value_if_found(iterator: Iterable["Nduv"], name: str) -> Any | None:
        try:
            first_found = next(filter(lambda item: item.name == name, iterator))
        except StopIteration:
            return None
        if hasattr(first_found, "value"):
            return first_found.value
        return None

    coupling_map = config.coupling_map
    n_qubits = config.n_qubits
    if coupling_map is None:
        # Assume full connectivity
        arc: FullyConnected | Architecture = FullyConnected(n_qubits)
    else:
        arc = Architecture(coupling_map)

    link_errors: dict = defaultdict(dict)
    node_errors: dict = defaultdict(dict)
    readout_errors: dict = {}

    t1_times = []
    t2_times = []
    frequencies = []
    gate_times = []

    if properties is not None:
        for index, qubit_info in enumerate(properties.qubits):
            t1_times.append([index, return_value_if_found(qubit_info, "T1")])
            t2_times.append([index, return_value_if_found(qubit_info, "T2")])
            frequencies.append([index, return_value_if_found(qubit_info, "frequency")])
            # readout error as a symmetric 2x2 matrix
            offdiag = return_value_if_found(qubit_info, "readout_error")
            if offdiag:
                diag = 1.0 - offdiag
                readout_errors[index] = [[diag, offdiag], [offdiag, diag]]
            else:
                readout_errors[index] = None

        for gate in properties.gates:
            name = gate.gate
            if name in _gate_str_2_optype:
                optype = _gate_str_2_optype[name]
                qubits = gate.qubits
                gate_error = return_value_if_found(gate.parameters, "gate_error")
                gate_error = gate_error if gate_error else 0.0
                gate_length = return_value_if_found(gate.parameters, "gate_length")
                gate_length = gate_length if gate_length else 0.0
                gate_times.append([name, qubits, gate_length])
                # add gate fidelities to their relevant lists
                if len(qubits) == 1:
                    node_errors[qubits[0]].update({optype: gate_error})
                elif len(qubits) == 2:  # noqa: PLR2004
                    link_errors[tuple(qubits)].update({optype: gate_error})
                    opposite_link = tuple(qubits[::-1])
                    if opposite_link not in coupling_map:
                        # to simulate a worse reverse direction square the fidelity
                        link_errors[opposite_link].update({optype: 2 * gate_error})

    # map type (k1 -> k2) -> v[k1] -> v[k2]
    K1 = TypeVar("K1")
    K2 = TypeVar("K2")
    V = TypeVar("V")
    convert_keys_t = Callable[[Callable[[K1], K2], dict[K1, V]], dict[K2, V]]

    # convert qubits to architecture Nodes
    convert_keys: convert_keys_t = lambda f, d: {f(k): v for k, v in d.items()}

    node_errors = convert_keys(lambda q: Node(q), node_errors)
    link_errors = convert_keys(lambda p: (Node(p[0]), Node(p[1])), link_errors)
    readout_errors = convert_keys(lambda q: Node(q), readout_errors)

    characterisation: dict[str, Any] = dict()  # noqa: C408
    characterisation["NodeErrors"] = node_errors
    characterisation["EdgeErrors"] = link_errors
    characterisation["ReadoutErrors"] = readout_errors
    characterisation["Architecture"] = arc
    characterisation["t1times"] = t1_times
    characterisation["t2times"] = t2_times
    characterisation["Frequencies"] = frequencies
    characterisation["GateTimes"] = gate_times

    return characterisation


def get_avg_characterisation(
    characterisation: dict[str, Any],
) -> dict[str, dict[Node, float]]:
    """
    Convert gate-specific characterisation into readout, one- and two-qubit errors

    Used to convert a typical output from `process_characterisation` into an input
    noise characterisation for NoiseAwarePlacement
    """

    K = TypeVar("K")
    V1 = TypeVar("V1")
    V2 = TypeVar("V2")
    map_values_t = Callable[[Callable[[V1], V2], dict[K, V1]], dict[K, V2]]
    map_values: map_values_t = lambda f, d: {k: f(v) for k, v in d.items()}

    node_errors = cast(
        "dict[Node, dict[OpType, float]]", characterisation["NodeErrors"]
    )
    link_errors = cast(
        "dict[tuple[Node, Node], dict[OpType, float]]", characterisation["EdgeErrors"]
    )
    readout_errors = cast(
        "dict[Node, list[list[float]]]", characterisation["ReadoutErrors"]
    )

    avg: Callable[[dict[Any, float]], float] = lambda xs: sum(xs.values()) / len(xs)
    avg_mat: Callable[[list[list[float]]], float] = (
        lambda xs: (xs[0][1] + xs[1][0]) / 2.0
    )

    avg_readout_errors = map_values(avg_mat, readout_errors)
    avg_node_errors = map_values(avg, node_errors)
    avg_link_errors = map_values(avg, link_errors)

    return {
        "node_errors": avg_node_errors,
        "edge_errors": avg_link_errors,
        "readout_errors": avg_readout_errors,
    }
