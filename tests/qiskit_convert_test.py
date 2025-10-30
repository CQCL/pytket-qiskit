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
import os
import warnings
from collections import Counter
from math import pi

import numpy as np
import pytest
import qiskit.circuit.library.standard_gates as qiskit_gates  # type: ignore
from pytket.circuit import (
    Bit,
    CircBox,
    Circuit,
    Conditional,
    CustomGateDef,
    Op,
    OpType,
    QControlBox,
    Qubit,
    StatePreparationBox,
    Unitary1qBox,
    Unitary2qBox,
    Unitary3qBox,
    reg_eq,
    reg_neq,
)
from pytket.passes import (
    CliffordSimp,
    DecomposeBoxes,
    FullPeepholeOptimise,
    RebaseTket,
    SequencePass,
)
from pytket.unit_id import _TEMP_BIT_NAME
from pytket.utils.results import (
    compare_statevectors,
    compare_unitaries,
    permute_rows_cols_in_unitary,
)
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit.circuit import IfElseOp, Parameter
from qiskit.circuit.equivalence_library import (  # type: ignore
    StandardEquivalenceLibrary,
)
from qiskit.circuit.library import (
    MCMTGate,
    PauliEvolutionGate,
    RYGate,
    UnitaryGate,
    XXPlusYYGate,
    n_local,
    real_amplitudes,
)
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector  # type: ignore
from qiskit.synthesis import SuzukiTrotter  # type: ignore
from qiskit.transpiler import (  # type: ignore
    CouplingMap,
    PassManager,
    PassManagerConfig,
)
from qiskit.transpiler.passes import BasisTranslator  # type: ignore
from qiskit.transpiler.preset_passmanagers.level2 import (  # type: ignore
    level_2_pass_manager,
)
from qiskit_aer import Aer  # type: ignore
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2  # type: ignore
from sympy import Symbol

from pytket.extensions.qiskit import IBMQBackend, qiskit_to_tk, tk_to_qiskit
from pytket.extensions.qiskit.backends import (
    AerBackend,
    qiskit_aer_backend,
)
from pytket.extensions.qiskit.qiskit_convert import _gate_str_2_optype
from pytket.extensions.qiskit.result_convert import qiskit_result_to_backendresult
from pytket.extensions.qiskit.tket_pass import TketAutoPass, TketPass

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None

REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires IBM configuration)"


# helper function for testing
def _get_qiskit_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Given a QuantumCircuit, use aer_simulator_statevector to compute its
    statevector, return the vector with its endianness adjusted"""
    back = Aer.get_backend("aer_simulator_statevector")
    qc.save_state()
    job = back.run(qc)
    return np.array(job.result().data()["statevector"].reverse_qargs().data)


def test_parameterised_circuit_global_phase() -> None:
    pass_1 = BasisTranslator(
        StandardEquivalenceLibrary,
        target_basis=FakeGuadalupeV2().configuration().basis_gates,
    )
    pass_2 = CliffordSimp()

    qc = QuantumCircuit(2)
    # qc.ryy(Parameter("MyParam"), 0, 1)

    pm = PassManager(pass_1)
    qc = pm.run(qc)

    tket_qc = qiskit_to_tk(qc)

    pass_2.apply(tket_qc)

    qc_2 = tk_to_qiskit(tket_qc)  # noqa: F841

    # assert type(qc_2.global_phase) is ParameterExpression


def test_classical_barrier_error() -> None:
    c = Circuit(1, 1)
    c.add_barrier([0], [0])
    with pytest.raises(NotImplementedError):
        tk_to_qiskit(c)


def test_convert_circuit_with_complex_params() -> None:
    with pytest.raises(ValueError):
        qiskit_op = SparsePauliOp(["Z"], coeffs=[1.0j])
        evolved_op = PauliEvolutionGate(
            qiskit_op, time=1, synthesis=SuzukiTrotter(reps=1)
        )
        evolution_circ = QuantumCircuit(1)
        evolution_circ.append(evolved_op, [0])
        tk_circ = qiskit_to_tk(evolution_circ)
        DecomposeBoxes().apply(tk_circ)


def get_test_circuit(measure: bool, reset: bool = True) -> QuantumCircuit:
    qr = QuantumRegister(4)
    cr = ClassicalRegister(4)
    qc = QuantumCircuit(qr, cr, name="test_circuit")
    qc.h(qr[0])
    qc.cx(qr[1], qr[0])
    qc.h(qr[0])
    qc.cx(qr[0], qr[3])
    qc.barrier(qr[3])
    if reset:
        qc.reset(qr[3])
    qc.rx(pi / 2, qr[3])
    qc.ry(0, qr[1])
    qc.z(qr[2])
    qc.ccx(qr[0], qr[1], qr[2])
    qc.ch(qr[0], qr[1])
    qc.cp(pi / 4, qr[0], qr[1])
    qc.cry(pi / 4, qr[0], qr[1])
    qc.crz(pi / 4, qr[1], qr[2])
    qc.cswap(qr[1], qr[2], qr[3])
    qc.cp(pi / 5, qr[2], qr[3])
    qc.cu(pi / 4, pi / 5, pi / 6, 0, qr[3], qr[0])
    qc.cy(qr[0], qr[1])
    qc.cz(qr[1], qr[2])
    qc.ecr(qr[0], qr[1])
    qc.id(qr[2])
    qc.iswap(qr[3], qr[0])
    qc.mcx([qr[0], qr[1], qr[2]], qr[3])
    qc.mcx([qr[1], qr[2], qr[3]], qr[0])
    qc.p(pi / 4, qr[1])
    qc.r(pi / 5, pi / 6, qr[2])
    qc.rxx(pi / 3, qr[2], qr[3])
    qc.ryy(pi / 3, qr[3], qr[2])
    qc.rz(pi / 4, qr[0])
    qc.rzz(pi / 5, qr[1], qr[2])
    qc.s(qr[3])
    qc.sdg(qr[0])
    qc.swap(qr[1], qr[2])
    qc.t(qr[3])
    qc.tdg(qr[0])
    qc.u(pi / 3, pi / 4, pi / 5, qr[0])
    qc.p(pi / 2, qr[1])
    qc.u(pi / 2, pi / 2, pi / 3, qr[2])
    qc.u(pi / 2, pi / 3, pi / 4, qr[3])
    qc.x(qr[0])
    qc.y(qr[1])

    if measure:
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        qc.measure(qr[3], cr[3])
    return qc


def test_convert() -> None:
    qc = get_test_circuit(False)
    tkc = qiskit_to_tk(qc)
    assert qc.name == tkc.name
    qc1 = tk_to_qiskit(tkc)
    assert qc1.name == tkc.name

    backend = Aer.get_backend("aer_simulator_statevector")

    qc.save_state()
    qc = transpile(qc, backend)
    job = backend.run([qc])
    state0 = job.result().get_statevector(qc)
    qc1.save_state()
    qc1 = transpile(qc1, backend)
    job1 = backend.run([qc1])
    state1 = job1.result().get_statevector(qc1)
    assert np.allclose(state0, state1, atol=1e-10)


# def test_symbolic() -> None:
#     pi2 = Symbol("pi2")
#     pi3 = Symbol("pi3")
#     pi0 = Symbol("pi0")
#     tkc = Circuit(3, 3, name="test").Ry(pi2, 1).Rx(pi3, 1).CX(1, 0)
#     tkc.add_phase(Symbol("pi0") * 2)
#     RebaseTket().apply(tkc)

#     qc = tk_to_qiskit(tkc)
#     tkc2 = qiskit_to_tk(qc)

#     assert tkc2.free_symbols() == {pi2, pi3, pi0}
#     tkc2.symbol_substitution({pi2: pi / 2, pi3: pi / 3, pi0: 0.1})

#     backend = Aer.get_backend("aer_simulator_statevector")
#     qc = tk_to_qiskit(tkc2)
#     assert qc.name == tkc.name
#     qc.save_state()
#     job = backend.run([qc])
#     state1 = job.result().get_statevector(qc)
#     state0 = np.array(
#         [
#             0.41273953 - 0.46964269j,
#             0.0 + 0.0j,
#             -0.0 + 0.0j,
#             -0.49533184 + 0.60309882j,
#             0.0 + 0.0j,
#             0.0 + 0.0j,
#             -0.0 + 0.0j,
#             -0.0 + 0.0j,
#         ]
#     )
#     assert np.allclose(state0, state1, atol=1e-10)


# def test_symbolic_2() -> None:
#     pi2 = Symbol("pi2")
#     pi3 = Symbol("pi3")
#     pi0 = Symbol("pi0")
#     tkc = Circuit(2, name="test").Ry(pi2, 1).Rx(pi3, 1).CX(1, 0)
#     tkc.add_phase(pi0 * 2)
#     RebaseTket().apply(tkc)

#     qc = tk_to_qiskit(tkc)

#     tkc1 = qiskit_to_tk(qc)
#     RebaseTket().apply(tkc1)
#     assert tkc == tkc1


def test_measures() -> None:
    qc = get_test_circuit(True)
    backend = qiskit_aer_backend("aer_simulator")
    qc = transpile(qc, backend)
    job = backend.run([qc], seed_simulator=7)
    counts0 = job.result().get_counts(qc)
    tkc = qiskit_to_tk(qc)
    qc = tk_to_qiskit(tkc)
    qc = transpile(qc, backend)
    job = backend.run([qc], seed_simulator=7)
    counts1 = job.result().get_counts(qc)
    for result, count in counts1.items():
        result_str = result.replace(" ", "")
        if counts0[result_str] != count:
            assert False  # noqa: B011


def test_boxes() -> None:
    c = Circuit(2)
    c.S(0)
    c.H(1)
    c.CX(0, 1)
    cbox = CircBox(c)
    d = Circuit(3, name="d")
    d.add_circbox(cbox, [0, 1])
    d.add_circbox(cbox, [1, 2])
    u = np.asarray([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    ubox = Unitary2qBox(u)
    d.add_unitary2qbox(ubox, 0, 1)
    qsc = tk_to_qiskit(d)
    d1 = qiskit_to_tk(qsc)
    assert len(d1.get_commands()) == 3
    DecomposeBoxes().apply(d)
    DecomposeBoxes().apply(d1)
    assert d == d1


def test_Unitary1qBox() -> None:
    c = Circuit(1)
    u = np.asarray([[0, 1], [1, 0]])
    ubox = Unitary1qBox(u)
    c.add_unitary1qbox(ubox, 0)
    # Convert to qiskit
    qc = tk_to_qiskit(c)
    # Verify that unitary from simulator is correct
    back = Aer.get_backend("aer_simulator_unitary")
    qc.save_unitary()
    job = back.run(qc).result()
    a = job.get_unitary(qc)
    u1 = np.asarray(a)
    assert np.allclose(u1, u)


def test_Unitary2qBox() -> None:
    c = Circuit(2)
    u = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    ubox = Unitary2qBox(u)
    c.add_unitary2qbox(ubox, 0, 1)
    # Convert to qiskit
    qc = tk_to_qiskit(c)
    # Verify that unitary from simulator is correct
    back = Aer.get_backend("aer_simulator_unitary")
    qc.save_unitary()
    job = back.run(qc).result()
    a = job.get_unitary(qc)
    u1 = permute_rows_cols_in_unitary(np.asarray(a), (1, 0))  # correct for endianness
    assert np.allclose(u1, u)


def test_Unitary3qBox() -> None:
    c = Circuit(3)
    u = np.asarray(
        [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    ubox = Unitary3qBox(u)
    c.add_unitary3qbox(ubox, 0, 1, 2)
    # Convert to qiskit
    qc = tk_to_qiskit(c)
    # Verify that unitary from simulator is correct
    back = Aer.get_backend("aer_simulator_unitary")
    qc.save_unitary()
    job = back.run(qc).result()
    a = job.get_unitary(qc)
    u1 = permute_rows_cols_in_unitary(
        np.asarray(a), (2, 1, 0)
    )  # correct for endianness
    assert np.allclose(u1, u)


def test_gates_phase() -> None:
    c = Circuit(4).SX(0).V(1).V(2).Vdg(3).Phase(0.5)
    qc = tk_to_qiskit(c)

    qr = QuantumRegister(4, "q")
    qc_correct = QuantumCircuit(qr)
    qc_correct.sx(qr[0])
    qc_correct.sx(qr[1])
    qc_correct.sx(qr[2])
    qc_correct.sxdg(qr[3])
    qc_correct.global_phase = pi / 4

    assert qc == qc_correct


def test_tketpass() -> None:
    qc = get_test_circuit(False, False)
    tkpass = FullPeepholeOptimise()
    back = Aer.get_backend("aer_simulator_unitary")
    for _ in range(12):
        tkc = qiskit_to_tk(qc)
        tkpass.apply(tkc)
    qc1 = tk_to_qiskit(tkc)
    qc1.save_unitary()
    res = back.run(qc1).result()
    u1 = res.get_unitary(qc1)
    qispass = TketPass(tkpass)
    pm = PassManager(qispass)
    qc2 = pm.run(qc)
    qc2.save_unitary()
    res = back.run(qc2).result()
    u2 = res.get_unitary(qc2)
    assert np.allclose(u1, u2)


@pytest.mark.timeout(None)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
def test_tketautopass(brussels_backend: IBMQBackend) -> None:
    backends = [
        Aer.get_backend("aer_simulator_statevector"),
        qiskit_aer_backend("aer_simulator"),
        Aer.get_backend("aer_simulator_unitary"),
    ]
    backends.append(brussels_backend._backend)  # noqa: SLF001
    for back in backends:
        for o_level in range(3):
            tkpass = TketAutoPass(
                back,
                o_level,
                instance=os.getenv("PYTKET_REMOTE_IBM_CLOUD_INSTANCE"),
                token=os.getenv("PYTKET_REMOTE_IBM_CLOUD_TOKEN"),
            )
            qc = get_test_circuit(True)
            pm = PassManager(passes=tkpass)
            pm.run(qc)


def test_instruction() -> None:
    # TKET-446
    op = SparsePauliOp(["XXI", "YYI", "ZZZ"], [0.3, 0.5, -0.4])
    evo_instr = PauliEvolutionGate(op, time=1.2, synthesis=SuzukiTrotter(reps=1))
    evolution_circ = QuantumCircuit(3)
    evolution_circ.append(evo_instr, [0, 1, 2])
    tk_circ = qiskit_to_tk(evolution_circ)
    cmds = tk_circ.get_commands()
    assert len(cmds) == 1
    assert cmds[0].op.type == OpType.CircBox


def test_conditions() -> None:
    box_c = Circuit(2, 2)
    box_c.Z(0)
    box_c.Y(1, condition_bits=[0, 1], condition_value=1)
    box_c.Measure(0, 0, condition_bits=[0, 1], condition_value=0)
    box = CircBox(box_c)

    u = np.asarray([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    ubox = Unitary2qBox(u)

    c = Circuit(2, 2, name="c")
    b = c.add_c_register("b", 1)
    c.add_circbox(
        box,
        [Qubit(0), Qubit(1), Bit(0), Bit(1)],
        condition_bits=[b[0]],
        condition_value=1,
    )
    c.add_unitary2qbox(
        ubox, Qubit(0), Qubit(1), condition_bits=[b[0]], condition_value=0
    )
    # Converting a CircBox containing conditional gates gives an error
    # TODO consider removing this restriction
    with pytest.raises(NotImplementedError):
        _ = tk_to_qiskit(c)


def test_condition_errors() -> None:
    with pytest.raises(Exception) as errorinfo:
        c = Circuit(2, 2)
        b = c.add_c_register("b", 2)
        c.X(Qubit(0), condition_bits=[b[0], Bit(0)], condition_value=1)
        tk_to_qiskit(c)
    assert "Conditions can only use a single register" in str(errorinfo.value)
    with pytest.raises(Exception) as errorinfo:
        c = Circuit(2, 2)
        c.X(0, condition_bits=[1, 0], condition_value=1)
        tk_to_qiskit(c)
    assert "Conditions must be an entire register in order" in str(errorinfo.value)


def test_correction() -> None:
    checked_x = Circuit(2, 1)
    checked_x.CX(0, 1)
    checked_x.X(0)
    checked_x.CX(0, 1)
    checked_x.Measure(1, 0)
    x_box = CircBox(checked_x)
    c = Circuit()
    target = Qubit("t", 0)
    ancilla = Qubit("a", 0)
    success = Bit("s", 0)
    c.add_qubit(target)
    c.add_qubit(ancilla)
    c.add_bit(success)
    c.add_circbox(x_box, args=[target, ancilla, success])
    c.add_circbox(
        x_box,
        args=[target, ancilla, success],
        condition_bits=[success],
        condition_value=0,
    )
    comp_pass = SequencePass([DecomposeBoxes(), RebaseTket()])
    comp_pass.apply(c)
    tk_to_qiskit(c)


def test_cnx() -> None:
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr, name="cnx_circuit")
    qc.mcx([0, 1, 2, 3], 4)
    c = qiskit_to_tk(qc)
    cmds = c.get_commands()
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.op.type == OpType.CnX
    assert len(cmd.qubits) == 5
    qregname = qc.qregs[0].name
    assert cmd.qubits[4] == Qubit(qregname, 4)


def test_convert_cnz_to_qiskit() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/460
    circ = Circuit(1).add_gate(OpType.CnZ, [0])
    qc = tk_to_qiskit(circ)
    assert qc[0].name == "z"
    circ = Circuit(2).add_gate(OpType.CnZ, [0, 1])
    qc = tk_to_qiskit(circ)
    assert qc[0].name == "cz"
    circ = Circuit(3).add_gate(OpType.CnZ, [0, 1, 2])
    qc = tk_to_qiskit(circ)
    assert qc[0].name == "mcz"


def test_gate_str_2_optype() -> None:
    samples = {
        "barrier": OpType.Barrier,
        "cx": OpType.CX,
        "mcx": OpType.CnX,
        "x": OpType.X,
    }
    assert all(_gate_str_2_optype[key] == val for key, val in samples.items())


def test_customgate() -> None:
    a = Symbol("a")
    def_circ = Circuit(2)
    def_circ.CZ(0, 1)
    def_circ.Rx(a, 1)
    gate_def = CustomGateDef.define("MyCRx", def_circ, [a])
    circ = Circuit(3)
    circ.Rx(0.1, 0)
    circ.Rx(0.4, 2)
    circ.add_custom_gate(gate_def, [0.2], [0, 1])

    qc1 = tk_to_qiskit(circ)
    newcirc = qiskit_to_tk(qc1)

    qc2 = tk_to_qiskit(newcirc)
    correct_circ = Circuit(3).Rx(0.1, 0).Rx(0.4, 2).CZ(0, 1).Rx(0.2, 1)
    correct_qc = tk_to_qiskit(correct_circ)

    backend = Aer.get_backend("aer_simulator_statevector")
    states = []
    for qc in (qc1, qc2, correct_qc):
        qc.save_state()
        qc = transpile(qc, backend)  # noqa: PLW2901
        job = backend.run([qc])
        states.append(job.result().get_statevector(qc))

    assert compare_statevectors(states[0], states[1])
    assert compare_statevectors(states[1], states[2])


def test_convert_result() -> None:
    # testing fix to register order bug TKET-752
    qr1 = QuantumRegister(1, name="q1")
    qr2 = QuantumRegister(2, name="q2")
    cr = ClassicalRegister(5, name="z")
    cr2 = ClassicalRegister(2, name="b")
    qc = QuantumCircuit(qr1, qr2, cr, cr2)
    qc.x(qr1[0])
    qc.x(qr2[1])

    # check statevector
    simulator = Aer.get_backend("aer_simulator_statevector")
    qc1 = qc.copy()
    qc1.save_state()
    qisk_result = simulator.run(qc1, shots=10).result()

    # exclude counts from result (we don't expect them
    # for the statevector sim after all)
    tk_res = next(qiskit_result_to_backendresult(qisk_result, include_counts=False))

    state = tk_res.get_state([Qubit("q2", 1), Qubit("q1", 0), Qubit("q2", 0)])
    correct_state = np.zeros(1 << 3, dtype=complex)
    correct_state[6] = 1 + 0j
    assert compare_statevectors(state, correct_state)
    # also check that we don't return counts in tket result
    # even if the qiskit result includes them
    assert tk_res._counts is None  # noqa: SLF001

    # check measured
    qc.measure(qr1[0], cr[0])
    qc.measure(qr2[1], cr2[0])

    simulator = qiskit_aer_backend("aer_simulator")
    qisk_result = simulator.run(qc, shots=10).result()

    tk_res = next(qiskit_result_to_backendresult(qisk_result))
    one_bits = [Bit("z", 0), Bit("b", 0)]
    zero_bits = [Bit("z", i) for i in range(1, 5)] + [Bit("b", 1)]

    assert tk_res.get_counts(one_bits) == Counter({(1, 1): 10})
    assert tk_res.get_counts(zero_bits) == Counter({(0, 0, 0, 0, 0): 10})


def add_x(
    qbit: int, qr: QuantumRegister, circuits: list[Circuit | QuantumCircuit]
) -> None:
    """Add an x gate to each circuit in a list,
    each one being either a tket or qiskit circuit."""
    for circ in circuits:
        if isinstance(circ, Circuit):
            circ.add_gate(OpType.X, [qbit])
        else:
            circ.x(qr[qbit])


def add_cnry(
    param: float,
    qbits: list[int],
    qr: QuantumRegister,
    circuits: list[Circuit | QuantumCircuit],
) -> None:
    """Add a CnRy gate to each circuit in a list,
    each one being either a tket or qiskit circuit."""
    assert len(qbits) >= 2
    for circ in circuits:
        if isinstance(circ, Circuit):
            circ.add_gate(OpType.CnRy, param, qbits)
        else:
            # param was "raw", so needs an extra PI.
            new_ry_gate = RYGate(param * pi)
            new_gate = MCMTGate(
                gate=new_ry_gate, num_ctrl_qubits=len(qbits) - 1, num_target_qubits=1
            )
            circ.append(new_gate, [qr[nn] for nn in qbits])


def assert_tket_circuits_identical(circuits: list[Circuit]) -> None:
    """Apart from the circuit names and qubit labels, assert that
    all circuits in the list are identical (i.e., identical gates), not just equivalent
    (having the same unitary matrix)."""
    if len(circuits) <= 1:
        return
    circ_copies = []

    for nn in range(len(circuits)):
        assert type(circuits[nn]) is Circuit
        circ = circuits[nn].copy()
        circ.name = "tk_circ_must_be_same_name"
        qbs = circ.qubits
        qubit_map = {qbs[mm]: Qubit("node", mm) for mm in range(len(qbs))}
        circ.rename_units(qubit_map)  # type: ignore
        circ_copies.append(circ)
    for nn in range(1, len(circ_copies)):
        assert circ_copies[0] == circ_copies[nn]


def assert_equivalence(
    circuits: list[Circuit | QuantumCircuit],
    require_qk_conversions_equality: bool = True,
    require_tk_equality: bool = True,
) -> None:
    """Given a list of circuits (either tket or qiskit), simulate them to calculate
    unitary matrices, and fail if they are not all almost equal.
    Also, (unless require_tk_equality is false), assert that
    all tket circuits are equal.
    If require_qk_conversions_equality is true,
    treat qk->tk conversions as if they were originally tk circuits and test
    for equality (rather than just equivalence), if require_tk_equality is true.
    """
    assert len(circuits) >= 2
    tk_circuits = []

    # We want unique circuit names, otherwise it confuses the Qiskit backend.
    names: set[str] = set()
    for nn in range(len(circuits)):
        if isinstance(circuits[nn], Circuit):
            if require_tk_equality:
                tk_circuits.append(circuits[nn])
            # Of course, use the tket simulator directly once available.
            # But not yet, so need to convert to qiskit circuits.
            circuits[nn] = tk_to_qiskit(circuits[nn])
        elif require_qk_conversions_equality and require_tk_equality:
            tk_circuits.append(qiskit_to_tk(circuits[nn]))
        names.add(circuits[nn].name)  # type: ignore
    assert len(names) == len(circuits)
    assert_tket_circuits_identical(tk_circuits)

    backend = Aer.get_backend("aer_simulator_unitary")
    unitaries = []
    for circ in circuits:
        assert isinstance(circ, QuantumCircuit)
        circ1 = circ.copy()
        circ1.save_unitary()
        circ1 = transpile(circ1, backend)
        job = backend.run(circ1)
        unitaries.append(job.result().get_unitary(circ1))
    for nn in range(1, len(circuits)):
        # Default np.allclose is very lax here, so use strict tolerances
        assert np.allclose(unitaries[0], unitaries[nn], atol=1e-14, rtol=0.0)


def qcirc_to_tcirc(qcirc: QuantumCircuit) -> Circuit:
    """Changes the name also, to avoid backend result clashes."""
    tcirc = qiskit_to_tk(qcirc)
    tcirc.name = "new tket circ from " + qcirc.name
    return tcirc


def test_cnry_conversion() -> None:
    """This is for TKET-991.
    Maintain parallel circuits, check equivalence at each stage.
    It would be good to subsume this as part of general
    randomised tests, where we add random gates in sequence."""
    tcirc = Circuit(3, name="parallel tcirc")
    qr = QuantumRegister(3, "q")
    qcirc = QuantumCircuit(qr, name="parallel qcirc")
    add_x(0, qr, [tcirc, qcirc])
    add_x(1, qr, [tcirc, qcirc])

    # It seems like we can test tket circuits for equality,
    # but not equivalence (since a direct tket simulator, with a
    # circuit->unitary function, is not yet available in pytket.
    # When it is available, we should add it here).
    #
    # Amusingly enough, it seems like we can test Qiskit circuits
    # for equivalence, but not for equality!
    #
    # Note that loops tk->qk->tk and qk->tk->qk should preserve
    # equivalence, but need not preserve equality because of different
    # gate sets.
    assert_equivalence([tcirc, qcirc])

    add_x(2, qr, [tcirc, qcirc])
    assert_equivalence([tcirc, qcirc])

    new_tcirc = qcirc_to_tcirc(qcirc)
    assert_equivalence([tcirc, qcirc, new_tcirc])

    add_x(0, qr, [tcirc, qcirc, new_tcirc])
    assert_equivalence([tcirc, qcirc, new_tcirc])

    add_cnry(0.1, [0, 1], qr, [tcirc, qcirc, new_tcirc])
    add_x(2, qr, [tcirc, qcirc, new_tcirc])

    # Because adding the CnRy gate to Qiskit circuits involves
    #       circ.append(new_gate, ...),
    # converting back to tket produces a CircBox rather than a CnRy gate.
    # So we cannot get tket equality, even though we have equivalence
    assert_equivalence([tcirc, qcirc, new_tcirc], require_qk_conversions_equality=False)

    add_x(0, qr, [qcirc, tcirc, new_tcirc])
    assert_equivalence([tcirc, qcirc, new_tcirc], require_qk_conversions_equality=False)

    add_cnry(0.2, [1, 0, 2], qr, [tcirc, qcirc, new_tcirc])
    assert_equivalence([tcirc, qcirc, new_tcirc], require_qk_conversions_equality=False)

    add_x(2, qr, [tcirc, qcirc, new_tcirc])
    add_x(1, qr, [tcirc, qcirc, new_tcirc])
    add_x(0, qr, [tcirc, qcirc, new_tcirc])
    assert_equivalence([tcirc, qcirc, new_tcirc], require_qk_conversions_equality=False)

    new_tcirc = qcirc_to_tcirc(qcirc)
    assert_equivalence(
        [tcirc, qcirc, new_tcirc],
        require_qk_conversions_equality=False,
        # We've done qk->tk conversion to get new_tcirc, so
        # we do not expect equality between new_tcirc and tcirc.
        require_tk_equality=False,
    )


# # pytket-extensions issue #72
# def test_parameter_equality() -> None:
#     param_a = Parameter("a")
#     param_b = Parameter("b")

#     circ = QuantumCircuit(2)
#     circ.rx(param_a, 0)
#     circ.ry(param_b, 1)
#     circ.cx(0, 1)
#     # fails with preserve_param_uuid=False
#     # as Parameter uuid attribute is not preserved
#     # and so fails equality check at assign_parameters
#     pytket_circ = qiskit_to_tk(circ, preserve_param_uuid=True)
#     final_circ = tk_to_qiskit(pytket_circ)

#     assert final_circ.parameters == circ.parameters

#     param_dict = dict(zip([param_a, param_b], [1, 2], strict=False))
#     final_circ.assign_parameters(param_dict, inplace=True)

#     assert len(final_circ.parameters) == 0


# https://github.com/CQCL/pytket-extensions/issues/275
def test_convert_multi_c_reg() -> None:
    c = Circuit()
    q0, q1 = c.add_q_register("q", 2)
    c.add_c_register("c", 2)
    [m0] = c.add_c_register("m", 1)
    c.add_gate(OpType.X, [], [q1], condition_bits=[m0], condition_value=1)
    c.CX(q0, q1)
    c.add_gate(OpType.TK1, [0.5, 0.5, 0.5], [q0])
    qcirc = tk_to_qiskit(c)
    circ = qiskit_to_tk(qcirc)
    assert circ.get_commands()[1].args == [Bit("tk_SCRATCH_BIT", 0), q1]


# test that tk_to_qiskit works after adding OpType.CRx and OpType.CRy
def test_crx_and_cry() -> None:
    tk_circ = Circuit(2)
    tk_circ.CRx(0.5, 0, 1)
    tk_circ.CRy(0.2, 1, 0)
    qiskit_circ = tk_to_qiskit(tk_circ)
    ops_dict = qiskit_circ.count_ops()
    assert ops_dict["crx"] == 1 and ops_dict["cry"] == 1


# test that tk_to_qiskit works for gates which don't have
# an exact substitution in qiskit e.g. ZZMax
# See issue "Add support for ZZMax gate in converters" #486
def test_rebased_conversion() -> None:
    tket_circzz = Circuit(3)
    tket_circzz.V(0).H(1).Vdg(2)
    tket_circzz.CV(0, 2)
    tket_circzz.add_gate(OpType.ZZMax, [0, 1])
    tket_circzz.add_gate(OpType.TK1, [0.1, 0.2, 0.3], [2])
    tket_circzz.add_gate(OpType.PhasedISWAP, [0.25, -0.5], [0, 2])
    qiskit_circzz = tk_to_qiskit(tket_circzz)
    tket_circzz2 = qiskit_to_tk(qiskit_circzz)
    u1 = tket_circzz.get_unitary()
    u2 = tket_circzz2.get_unitary()
    assert compare_unitaries(u1, u2)


# def test_convert_symbolic_circ() -> None:
#     a = fresh_symbol("alpha")

#     circ = Circuit(2)

#     circ.ZZPhase(a, 0, 1)

#     qc = tk_to_qiskit(circ)
#     _ = qiskit_to_tk(qc)


# def test_convert_symbolic_circ_2() -> None:
#     a = fresh_symbol("beta")

#     circ = Circuit(2)

#     circ.ZZPhase(a, 0, 1)

#     qc = tk_to_qiskit(circ)

#     _ = qiskit_to_tk(qc)


@pytest.mark.xfail(reason="PauliEvolutionGate with symbolic parameter not supported")
def test_parametrized_evolution() -> None:
    operator = SparsePauliOp(["XXZ", "YXY"], coeffs=[1.0, 0.5]) * Parameter("x")
    evolved_circ_op = PauliEvolutionGate(
        operator, time=1, synthesis=SuzukiTrotter(reps=2, order=4)
    )
    qc = QuantumCircuit(3)
    qc.append(evolved_circ_op, [0, 1, 2])
    tk_qc: Circuit = qiskit_to_tk(qc)
    assert len(tk_qc.free_symbols()) == 1


def test_multicontrolled_gate_conversion() -> None:
    my_qc = QuantumCircuit(4)
    my_qc.append(qiskit_gates.YGate().control(3), [0, 1, 2, 3])
    my_qc.append(qiskit_gates.RYGate(0.25).control(3), [0, 1, 2, 3])
    my_qc.append(qiskit_gates.ZGate().control(3), [0, 1, 2, 3])
    my_tkc = qiskit_to_tk(my_qc)
    my_tkc.add_gate(OpType.CnRy, [0.95], [0, 1, 2, 3])
    my_tkc.add_gate(OpType.CnZ, [1, 2, 3, 0])
    my_tkc.add_gate(OpType.CnY, [0, 1, 3, 2])
    unitary_before = my_tkc.get_unitary()
    assert my_tkc.n_gates_of_type(OpType.CnY) == 2
    assert my_tkc.n_gates_of_type(OpType.CnZ) == 2
    assert my_tkc.n_gates_of_type(OpType.CnRy) == 2
    my_new_qc = tk_to_qiskit(my_tkc)
    qiskit_ops = my_new_qc.count_ops()
    assert qiskit_ops["c3y"] and qiskit_ops["mcz"] and qiskit_ops["c3ry"] == 2
    tcirc = qiskit_to_tk(my_new_qc)
    unitary_after = tcirc.get_unitary()
    assert compare_unitaries(unitary_before, unitary_after)


def test_qcontrolbox_conversion() -> None:
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr)
    c2h_gate = qiskit_gates.HGate().control(2)
    qc.append(c2h_gate, qr)
    c = qiskit_to_tk(qc)
    assert c.n_gates == 1
    assert c.n_gates_of_type(OpType.QControlBox) == 1
    c3rx_gate = qiskit_gates.RXGate(0.7).control(3)
    c3rz_gate = qiskit_gates.RZGate(pi / 4).control(3)
    c2rzz_gate = qiskit_gates.RZZGate(pi / 3).control(2)
    qc2 = QuantumCircuit(4)
    qc2.append(c3rz_gate, [0, 1, 3, 2])
    qc2.append(c3rx_gate, [0, 1, 2, 3])
    qc2.append(c2rzz_gate, [0, 1, 2, 3])
    tkc2 = qiskit_to_tk(qc2)
    assert tkc2.n_gates == 3
    assert tkc2.n_gates_of_type(OpType.QControlBox) == 3


def test_controlled_unitary_conversion() -> None:
    u = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    qc = QuantumCircuit(4)
    cu_gate = UnitaryGate(u).control(num_ctrl_qubits=2, ctrl_state="01")
    qc.append(cu_gate, [0, 1, 2, 3])
    u_qc = permute_rows_cols_in_unitary(Operator(qc).data, (3, 2, 1, 0))
    tkc = qiskit_to_tk(qc)
    u_tkc = tkc.get_unitary()
    assert np.allclose(u_qc, u_tkc)


def test_qcontrol_box_conversion_to_qiskit() -> None:
    ccch_001 = QControlBox(
        Op.create(OpType.H), n_controls=3, control_state=(False, False, True)
    )
    cccs_110 = QControlBox(
        Op.create(OpType.S), n_controls=3, control_state=(True, True, False)
    )
    cccRy_100 = QControlBox(
        Op.create(OpType.Ry, 0.73), n_controls=3, control_state=(True, False, False)
    )
    ccU3_10 = QControlBox(
        Op.create(OpType.U3, [0.1, 0.2, 0.3]), n_controls=2, control_state=(True, False)
    )
    circ1 = Circuit(4, name="test_circ")
    circ1.add_gate(ccch_001, [0, 1, 2, 3])
    circ1.add_gate(cccs_110, [0, 1, 2, 3])
    circ1.add_gate(cccRy_100, [3, 2, 1, 0])
    circ1.add_gate(ccU3_10, [1, 0, 2])
    qc = tk_to_qiskit(circ1)
    qiskit_unitary = permute_rows_cols_in_unitary(Operator(qc).data, (3, 2, 1, 0))
    assert compare_unitaries(qiskit_unitary, circ1.get_unitary())
    circ2 = qiskit_to_tk(qc)
    DecomposeBoxes().apply(circ1)
    DecomposeBoxes().apply(circ2)
    assert circ1 == circ2


# Ensures that the tk_to_qiskit converter does not cancel redundant gates
def test_tk_to_qiskit_redundancies() -> None:
    h_circ = Circuit(1).H(0).H(0)
    qc_h = tk_to_qiskit(h_circ)
    assert qc_h.count_ops()["h"] == 2


def test_ccx_conversion() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/117
    c00 = QuantumCircuit(3)
    c00.ccx(0, 1, 2, 0)  # 0 = "00" (little-endian)
    assert compare_unitaries(
        qiskit_to_tk(c00).get_unitary(),
        Circuit(3).X(0).X(1).CCX(0, 1, 2).X(0).X(1).get_unitary(),
    )
    c10 = QuantumCircuit(3)
    c10.ccx(0, 1, 2, 1)  # 1 = "10" (little-endian)
    assert compare_unitaries(
        qiskit_to_tk(c10).get_unitary(),
        Circuit(3).X(1).CCX(0, 1, 2).X(1).get_unitary(),
    )
    c01 = QuantumCircuit(3)
    c01.ccx(0, 1, 2, 2)  # 2 = "01" (little-endian)
    assert compare_unitaries(
        qiskit_to_tk(c01).get_unitary(),
        Circuit(3).X(0).CCX(0, 1, 2).X(0).get_unitary(),
    )
    c11 = QuantumCircuit(3)
    c11.ccx(0, 1, 2, 3)  # 3 = "11" (little-endian)
    assert compare_unitaries(
        qiskit_to_tk(c11).get_unitary(),
        Circuit(3).CCX(0, 1, 2).get_unitary(),
    )


def test_conditional_conversion() -> None:
    c = Circuit(1, 2, "conditional_circ")
    c.X(0, condition_bits=[0], condition_value=1)

    c_qiskit = tk_to_qiskit(c)
    c_tket = qiskit_to_tk(c_qiskit)

    expected_circ = Circuit(1, 2, "conditional_circ")
    if_box = CircBox(Circuit(1, name="If").X(0))
    expected_circ.add_circbox(
        if_box, [Qubit(0)], condition_bits=[Bit(0)], condition_value=1
    )

    assert c_tket == expected_circ


def test_conditional_conversion_2() -> None:
    c = Circuit(1, 2, "conditional_circ_2")
    c.X(0, condition_bits=[1], condition_value=1)
    c_qiskit = tk_to_qiskit(c)
    c_tket = qiskit_to_tk(c_qiskit)

    expected_circ = Circuit(1, 2, "conditional_circ_2")
    if_box = CircBox(Circuit(1, name="If").X(0))
    expected_circ.add_circbox(
        if_box, [Qubit(0)], condition_bits=[Bit(1)], condition_value=1
    )

    assert c_tket == expected_circ


# https://github.com/CQCL/pytket-qiskit/issues/100
def test_state_prep_conversion_array_or_list() -> None:
    # State prep with list of real amplitudes
    ghz_state_permuted = np.array([0, 0, 1 / np.sqrt(2), 0, 0, 0, 0, 1 / np.sqrt(2)])
    qc_sp = QuantumCircuit(3)
    qc_sp.prepare_state(ghz_state_permuted)
    tk_sp = qiskit_to_tk(qc_sp)
    assert tk_sp.n_gates_of_type(OpType.StatePreparationBox) == 1
    assert tk_sp.n_gates == 1
    assert compare_statevectors(tk_sp.get_statevector(), ghz_state_permuted)
    # State prep with ndarray of complex amplitudes
    qc_sp2 = QuantumCircuit(2)
    complex_statvector = np.array([1 / np.sqrt(2), 0, -1.0j / np.sqrt(2), 0])
    qc_sp2.initialize(complex_statvector, qc_sp2.qubits)
    tk_sp2 = qiskit_to_tk(qc_sp2)
    assert tk_sp2.n_gates_of_type(OpType.StatePreparationBox) == 1
    assert tk_sp2.n_gates == 1
    # test tket -> qiskit conversion
    converted_qiskit_qc = tk_to_qiskit(tk_sp2)
    assert converted_qiskit_qc.count_ops()["initialize"] == 1
    tk_sp3 = qiskit_to_tk(converted_qiskit_qc)
    # check circuit decomposes as expected
    DecomposeBoxes().apply(tk_sp3)
    assert tk_sp3.n_gates_of_type(OpType.Reset) == 2
    state_arr = 1 / np.sqrt(2) * np.array([1, 1, 0, 0])
    sv = Statevector(state_arr)
    qc_2 = QuantumCircuit(2)
    qc_2.prepare_state(sv, [0, 1])
    tkc_2 = qiskit_to_tk(qc_2)
    assert tkc_2.n_gates_of_type(OpType.StatePreparationBox) == 1


def test_state_prep_conversion_with_int() -> None:
    qc = QuantumCircuit(4)
    qc.prepare_state(7, qc.qubits)
    tkc7 = qiskit_to_tk(qc)
    assert tkc7.n_gates_of_type(OpType.X) == 3
    qc_sv = _get_qiskit_statevector(qc.decompose())
    assert compare_statevectors(tkc7.get_statevector(), qc_sv)
    int_statevector = Statevector.from_int(5, 8)
    qc_s = QuantumCircuit(3)
    qc_s.prepare_state(int_statevector)
    # unfortunately Aer doesn't support state_preparation
    # instructions so we decompose first
    d_qc_s = qc_s.decompose(reps=5)
    sv_int = _get_qiskit_statevector(d_qc_s)
    tkc_int = qiskit_to_tk(qc_s)
    tkc_int_sv = tkc_int.get_statevector()
    assert compare_statevectors(tkc_int_sv, sv_int)


def test_state_prep_conversion_with_str() -> None:
    qc = QuantumCircuit(5)
    qc.initialize("rl+-1")
    tk_circ = qiskit_to_tk(qc)
    assert tk_circ.n_gates_of_type(OpType.Reset) == 5
    assert tk_circ.n_gates_of_type(OpType.H) == 4
    assert tk_circ.n_gates_of_type(OpType.X) == 2
    qc_string_sp = QuantumCircuit(3)
    qc_string_sp.prepare_state("r-l")
    decomposed_qc = qc_string_sp.decompose(reps=4)
    qiskit_sv = _get_qiskit_statevector(decomposed_qc)
    tk_string_sp = qiskit_to_tk(qc_string_sp)
    assert tk_string_sp.n_gates_of_type(OpType.H) == 3
    assert tk_string_sp.n_gates_of_type(OpType.Sdg) == 1
    assert compare_statevectors(qiskit_sv, tk_string_sp.get_statevector())
    sv_str = Statevector.from_label("rr+-")
    sv_qc = QuantumCircuit(4)
    sv_qc.prepare_state(sv_str)
    decomposed_sv_qc = sv_qc.decompose(reps=6)
    sv_array = _get_qiskit_statevector(decomposed_sv_qc)
    tkc_sv = qiskit_to_tk(sv_qc)
    assert compare_statevectors(sv_array, tkc_sv.get_statevector())


def test_conversion_to_tket_with_and_without_resets() -> None:
    test_state = 1 / np.sqrt(3) * np.array([1, 1, 0, 0, 0, 0, 1, 0])
    tket_sp_reset = StatePreparationBox(test_state, with_initial_reset=True)
    tk_circ_reset = Circuit(3).add_gate(tket_sp_reset, [0, 1, 2])
    qiskit_qc_init = tk_to_qiskit(tk_circ_reset)
    assert qiskit_qc_init.count_ops()["initialize"] == 1
    tket_sp_no_reset = StatePreparationBox(test_state, with_initial_reset=False)
    tket_circ_no_reset = Circuit(3).add_gate(tket_sp_no_reset, [0, 1, 2])
    tkc_sv = tket_circ_no_reset.get_statevector()
    qiskit_qc_sp = tk_to_qiskit(tket_circ_no_reset)
    assert qiskit_qc_sp.count_ops()["state_preparation"] == 1
    decomp_qc = qiskit_qc_sp.decompose(reps=5)
    qiskit_state = _get_qiskit_statevector(decomp_qc)
    assert compare_statevectors(tkc_sv, qiskit_state)


def test_unitary_gate() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/122
    qkc = QuantumCircuit(3)
    for n in range(4):
        u = np.eye(1 << n, dtype=complex)
        gate = UnitaryGate(u)
        qkc.append(gate, list(range(n)))
    tkc = qiskit_to_tk(qkc)
    cmds = tkc.get_commands()
    assert len(cmds) == 3
    assert cmds[0].op.type == OpType.Unitary1qBox
    assert cmds[1].op.type == OpType.Unitary2qBox
    assert cmds[2].op.type == OpType.Unitary3qBox


def test_ccz_conversion() -> None:
    qc_ccz = QuantumCircuit(4)
    qc_ccz.append(qiskit_gates.CCZGate(), [0, 1, 2])
    qc_ccz.append(qiskit_gates.CCZGate(), [3, 1, 0])
    tkc_ccz = qiskit_to_tk(qc_ccz)
    assert tkc_ccz.n_gates_of_type(OpType.CnZ) == tkc_ccz.n_gates == 2
    # bidirectional CnZ conversion already supported
    qc_ccz2 = tk_to_qiskit(tkc_ccz)
    assert qc_ccz2.count_ops()["mcz"] == 2
    tkc_ccz2 = qiskit_to_tk(qc_ccz2)
    assert compare_unitaries(tkc_ccz.get_unitary(), tkc_ccz2.get_unitary())


def test_csx_conversion() -> None:
    qc_csx = QuantumCircuit(2)
    qc_csx.append(qiskit_gates.CSXGate(), [0, 1])
    qc_csx.append(qiskit_gates.CSXGate(), [1, 0])
    converted_tkc = qiskit_to_tk(qc_csx)
    assert converted_tkc.n_gates == 2
    assert converted_tkc.n_gates_of_type(OpType.CSX) == 2
    u1 = converted_tkc.get_unitary()
    new_tkc_csx = Circuit(2)
    new_tkc_csx.add_gate(OpType.CSX, [0, 1]).add_gate(OpType.CSX, [1, 0])
    u2 = new_tkc_csx.get_unitary()
    assert compare_unitaries(u1, u2)
    converted_qc = tk_to_qiskit(new_tkc_csx)
    assert converted_qc.count_ops()["csx"] == 2
    qc_c3sx = QuantumCircuit(4)
    qc_c3sx.append(qiskit_gates.C3SXGate(), [0, 1, 2, 3])
    tkc_c3sx = qiskit_to_tk(qc_c3sx)
    assert tkc_c3sx.n_gates == tkc_c3sx.n_gates_of_type(OpType.QControlBox) == 1


def test_CS_and_CSdg() -> None:
    qiskit_qc = QuantumCircuit(2)
    qiskit_qc.append(qiskit_gates.CSGate(), [0, 1])
    qiskit_qc.append(qiskit_gates.CSdgGate(), [0, 1])
    qiskit_qc.append(qiskit_gates.CSGate(), [1, 0])
    qiskit_qc.append(qiskit_gates.CSdgGate(), [1, 0])
    tkc = qiskit_to_tk(qiskit_qc)
    assert tkc.n_gates_of_type(OpType.QControlBox) == 4


def test_failed_conversion_error() -> None:
    qc = QuantumCircuit(2)
    qc.append(XXPlusYYGate(0.1), [0, 1])  # add unsupported gate
    with pytest.raises(
        NotImplementedError, match=r"Conversion of qiskit's xx_plus_yy instruction"
    ):
        qiskit_to_tk(qc)


# https://github.com/CQCL/pytket-qiskit/issues/200
def test_real_amplitudes_numeric_params() -> None:
    qc = QuantumCircuit(3)
    params = [np.pi / 2] * 9
    real_amps1 = real_amplitudes(3, reps=2)
    real_amps2 = real_amps1.assign_parameters(params)
    qc.compose(real_amps2, qubits=[0, 1, 2], inplace=True)
    # Unitary operator of the qiskit circuit. Order reversed from little -> big endian.
    # The reversal means we can check it for equivalence with a tket unitary
    qiskit_unitary = Operator(qc.reverse_bits()).data
    converted_tkc = qiskit_to_tk(qc)
    assert converted_tkc.n_gates == 13
    assert converted_tkc.n_gates_of_type(OpType.CX) == 4
    assert converted_tkc.n_gates_of_type(OpType.Ry) == 9
    unitary1 = converted_tkc.get_unitary()
    qc2 = tk_to_qiskit(converted_tkc)
    tkc2 = qiskit_to_tk(qc2)
    unitary2 = tkc2.get_unitary()
    assert compare_unitaries(qiskit_unitary, unitary1)
    assert compare_unitaries(unitary1, unitary2)


# https://github.com/CQCL/pytket-qiskit/issues/256
@pytest.mark.xfail(reason="Limited support for symbolic conversions")
def test_symbolic_param_conv() -> None:
    qc = n_local(2, "ry", "cz", reps=1, entanglement="linear")
    qc_transpiled = transpile(
        qc, basis_gates=["sx", "rz", "cx", "x"], optimization_level=3
    )

    tket_qc = qiskit_to_tk(qc_transpiled)
    CliffordSimp().apply(tket_qc)
    transformed_qc = tk_to_qiskit(tket_qc)

    qc_transpiled_again = transpile(transformed_qc, basis_gates=["sx", "rz", "cx", "x"])

    qc_transpiled_again = qc_transpiled_again.assign_parameters(
        {
            qc_transpiled_again.parameters[i]: 0
            for i in range(len(qc_transpiled_again.parameters))
        }
    )


def test_implicit_swap_warning() -> None:
    c = Circuit(2).H(0).SWAP(0, 1)
    c.replace_SWAPs()
    c.measure_all()
    with pytest.warns(UserWarning, match="The pytket Circuit contains implicit qubit"):
        tk_to_qiskit(c)

    shots_backend = AerBackend()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shots_backend.run_circuit(c)


# https://github.com/CQCL/pytket-qiskit/issues/337
def test_nonregister_bits() -> None:
    c = Circuit(1).X(0).measure_all()
    c.rename_units({Bit(0): Bit(1)})
    with pytest.raises(NotImplementedError):
        tk_to_qiskit(c)


# https://github.com/CQCL/pytket-qiskit/issues/415
def test_ifelseop_two_branches() -> None:
    qreg = QuantumRegister(1, "r")
    creg = ClassicalRegister(1, "s")
    circuit = QuantumCircuit(qreg, creg)

    circuit.h(qreg[0])
    circuit.measure(qreg[0], creg[0])

    with circuit.if_test((creg[0], 1)) as else_:
        circuit.h(qreg[0])
    with else_:
        circuit.x(qreg[0])
    circuit.measure(qreg[0], creg[0])

    tkc = qiskit_to_tk(circuit)
    tkc.name = "test_circ"

    # Manually build the expected pytket Circuit.
    # Validate against tkc.
    expected_circ = Circuit(name="test_circ")
    r_reg = expected_circ.add_q_register("r", 1)
    s_reg = expected_circ.add_c_register("s", 1)
    expected_circ.H(r_reg[0])
    expected_circ.Measure(r_reg[0], s_reg[0])

    h_circ = Circuit()
    h_reg = h_circ.add_q_register("r", 1)
    h_circ.name = "If"
    h_circ.H(h_reg[0])

    x_circ = Circuit()
    x_reg = x_circ.add_q_register("r", 1)
    x_circ.name = "Else"
    x_circ.X(x_reg[0])

    expected_circ.add_circbox(
        CircBox(h_circ), [r_reg[0]], condition_bits=[s_reg[0]], condition_value=1
    )
    expected_circ.add_circbox(
        CircBox(x_circ), [r_reg[0]], condition_bits=[s_reg[0]], condition_value=0
    )

    expected_circ.Measure(r_reg[0], s_reg[0])

    assert expected_circ == tkc


# https://github.com/CQCL/pytket-qiskit/issues/415
def test_ifelseop_one_branch() -> None:
    qubits = QuantumRegister(1, "q1")
    clbits = ClassicalRegister(1, "c1")
    circuit = QuantumCircuit(qubits, clbits)
    (q0,) = qubits
    (c0,) = clbits

    circuit.h(q0)
    circuit.measure(q0, c0)
    with circuit.if_test((c0, 1)):
        circuit.x(q0)
    circuit.measure(q0, c0)

    tket_circ_if_else = qiskit_to_tk(circuit)
    tket_circ_if_else.name = "test_circ"

    # Manually build the expected pytket Circuit.
    # Validate against tket_circ_if_else.
    expected_circ = Circuit()
    expected_circ.name = "test_circ"
    q1_tk = expected_circ.add_q_register("q1", 1)
    c1_tk = expected_circ.add_c_register("c1", 1)
    expected_circ.H(q1_tk[0])
    expected_circ.Measure(q1_tk[0], c1_tk[0])
    x_circ = Circuit()
    x_circ.name = "If"
    xq1 = x_circ.add_q_register("q1", 1)
    x_circ.X(xq1[0])
    expected_circ.add_circbox(
        CircBox(x_circ), [q1_tk[0]], condition_bits=[c1_tk[0]], condition_value=1
    )

    expected_circ.Measure(q1_tk[0], c1_tk[0])

    assert tket_circ_if_else == expected_circ


# https://github.com/CQCL/pytket-qiskit/issues/452
def test_ifelseop_reg_cond_if() -> None:
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "c")
    circuit = QuantumCircuit(creg, qreg)
    (q0, q1, q2) = qreg
    (c0, c1, c2) = creg
    circuit.h(q0)
    circuit.h(q1)
    circuit.h(q2)
    circuit.measure(q0, c0)
    circuit.measure(q1, c1)
    circuit.measure(q2, c2)
    # Condition is on a register not a bit
    with circuit.if_test((creg, 2)):
        circuit.x(q0)
        circuit.y(q1)
        circuit.z(q2)
    circuit.measure(q0, c0)
    circuit.measure(q1, c1)
    circuit.measure(q2, c2)

    tkc: Circuit = qiskit_to_tk(circuit)
    tkc.name = "test_circ"

    expected_circ = Circuit()
    expected_circ.name = "test_circ"
    qreg_tk = expected_circ.add_q_register("q", 3)
    creg_tk = expected_circ.add_c_register("c", 3)
    expected_circ.H(qreg_tk[0])
    expected_circ.H(qreg_tk[1])
    expected_circ.H(qreg_tk[2])
    expected_circ.Measure(qreg_tk[0], creg_tk[0])
    expected_circ.Measure(qreg_tk[1], creg_tk[1])
    expected_circ.Measure(qreg_tk[2], creg_tk[2])

    pauli_circ = Circuit()
    pauli_circ.name = "If"
    pauli_qreg = pauli_circ.add_q_register("q", 3)
    pauli_circ.X(pauli_qreg[0]).Y(pauli_qreg[1]).Z(pauli_qreg[2])
    expected_circ.add_circbox(
        CircBox(pauli_circ),
        [qreg_tk[0], qreg_tk[1], qreg_tk[2]],
        condition=reg_eq(creg_tk, 2),
    )

    expected_circ.Measure(qreg_tk[0], creg_tk[0])
    expected_circ.Measure(qreg_tk[1], creg_tk[1])
    expected_circ.Measure(qreg_tk[2], creg_tk[2])

    assert expected_circ == tkc


# https://github.com/CQCL/pytket-qiskit/issues/452
def test_ifelseop_reg_cond_if_else() -> None:
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    circuit = QuantumCircuit(creg, qreg)
    (q0, q1) = qreg
    (c0, c1) = creg

    circuit.h(q0)
    circuit.h(q1)
    circuit.measure(q0, c0)
    circuit.measure(q1, c1)
    # Condition is on a register not a bit
    with circuit.if_test((creg, 2)) as else_:
        circuit.x(q0)
        circuit.x(q1)
    with else_:
        circuit.y(q0)
        circuit.y(q1)
    circuit.measure(q0, c0)
    circuit.measure(q1, c1)
    tkc: Circuit = qiskit_to_tk(circuit)
    tkc.name = "test_circ"

    expected_circ = Circuit()
    expected_circ.name = "test_circ"
    qreg_tk = expected_circ.add_q_register("q", 2)
    creg_tk = expected_circ.add_c_register("c", 2)
    expected_circ.H(qreg_tk[0])
    expected_circ.H(qreg_tk[1])
    expected_circ.Measure(qreg_tk[0], creg_tk[0])
    expected_circ.Measure(qreg_tk[1], creg_tk[1])

    x_circ2 = Circuit()
    x_circ2.name = "If"
    x_qreg = x_circ2.add_q_register("q", 2)
    x_circ2.X(x_qreg[0]).X(x_qreg[1])
    expected_circ.add_circbox(
        CircBox(x_circ2), [qreg_tk[0], qreg_tk[1]], condition=reg_eq(creg_tk, 2)
    )

    y_circ2 = Circuit()
    y_circ2.name = "Else"
    y_qreg = y_circ2.add_q_register("q", 2)
    y_circ2.Y(y_qreg[0]).Y(y_qreg[1])
    expected_circ.add_circbox(
        CircBox(y_circ2), [qreg_tk[0], qreg_tk[1]], condition=reg_neq(creg_tk, 2)
    )

    expected_circ.Measure(qreg_tk[0], creg_tk[0])
    expected_circ.Measure(qreg_tk[1], creg_tk[1])

    assert expected_circ == tkc


def test_range_preds_with_conditionals() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/375
    c = Circuit(1, 1)
    treg = c.add_c_register(_TEMP_BIT_NAME, 1)
    c.add_c_range_predicate(1, 1, [Bit(0)], treg[0])
    c.add_gate(OpType.X, [Qubit(0)], condition_bits=[treg[0]], condition_value=1)
    c.add_gate(OpType.Y, [Qubit(0)], condition_bits=[treg[0]], condition_value=1)
    qkc = tk_to_qiskit(c)
    assert len(qkc) == 2
    assert len(qkc.qubits) == 1
    assert len(qkc.clbits) == 1


def test_nested_conditionals() -> None:
    # https://github.com/CQCL/pytket-qiskit/issues/442
    c0 = Circuit(1, 1).X(0, condition_bits=[0], condition_value=1)
    cbox = CircBox(c0)
    c = Circuit(1, 2)
    c.add_circbox(cbox, [Qubit(0), Bit(1)], condition_bits=[Bit(0)], condition_value=1)
    DecomposeBoxes().apply(c)
    with pytest.raises(NotImplementedError):
        # For now we do not support conversion of nested conditionals.
        _qkc = tk_to_qiskit(c)


def _fetch_if_elses(qc: QuantumCircuit) -> list[IfElseOp]:
    """Get a list of all IfElseOp instructions in a QuantumCircuit."""
    if_else_list = []
    for datum in qc.data:
        instr, _, _ = datum.operation, datum.qubits, datum.clbits
        if type(instr) is IfElseOp:
            if_else_list.append(instr)
    return if_else_list


def test_qiskitv2_conversions() -> None:
    circ = Circuit(4, 2)
    circ.H(0)
    circ.Measure(0, 0)
    circ.Measure(1, 1)
    prep = StatePreparationBox(1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0]))
    circ.add_gate(
        prep,
        args=[Qubit(0), Qubit(1), Qubit(2)],
        condition_bits=[Bit(0)],
        condition_value=1,
    )
    circ.add_gate(
        OpType.CnZ,
        [Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
        condition_bits=[Bit(0)],
        condition_value=0,
    )
    circ.TK1(
        0.7,
        0.8,
        0.9,
        qubit=Qubit(0),
        condition_bits=[Bit(0), Bit(1)],
        condition_value=2,
    )
    qc = tk_to_qiskit(circ)
    if_list = _fetch_if_elses(qc)
    assert qc.count_ops()["if_else"] == 3 == len(if_list)
    if_prep, if_cnz, if_tk1 = tuple(if_list)
    # Check condition values of the converted QuantumCircuit
    assert if_prep.condition[1] == 1
    assert if_cnz.condition[1] == 0
    assert if_tk1.condition == (ClassicalRegister(2, "c"), 2)


# https://github.com/CQCL/pytket-qiskit/issues/514
def test_bit_ref_circuit() -> None:
    qreg = QuantumRegister(1)
    qreg_setter = QuantumRegister(2)
    creg_A = ClassicalRegister(1)
    creg_B = ClassicalRegister(1)

    qc = QuantumCircuit(qreg, qreg_setter, creg_A, creg_B)
    qc.x(qreg_setter[1])

    with qc.if_test((creg_A[0], 0)) as _else:
        qc.measure(qreg_setter[1], creg_B[0])
    with _else:
        qc.measure(qreg_setter[0], creg_B[0])
    tkc = qiskit_to_tk(qc)
    cregs = tkc.c_registers
    assert len(cregs) == 2
    a_creg = cregs[0]
    b_creg = cregs[1]
    assert a_creg.size == 1
    assert b_creg.size == 1


def test_round_trip_with_qiskit_transpilation() -> None:
    circ = Circuit(4, 1)
    circ.H(0).Measure(0, 0)
    circ.U1(1 / 2, Qubit(1), condition_bits=[Bit(0)], condition_value=1)
    circ.U1(1 / 4, Qubit(2), condition_bits=[Bit(0)], condition_value=1)
    circ.U1(1 / 8, Qubit(3), condition_bits=[Bit(0)], condition_value=1)

    qc = tk_to_qiskit(circ)

    coupling = CouplingMap(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
    )
    config = PassManagerConfig(
        coupling_map=coupling,
        basis_gates=["cx", "sx", "x", "rz", "if_else"],
        seed_transpiler=0,
    )
    pass_manager = level_2_pass_manager(config)
    compiled_qc = pass_manager.run(qc)
    tk_circ = qiskit_to_tk(compiled_qc)
    assert tk_circ.n_gates_of_type(OpType.Conditional) == 3
    conditional_cmds = tk_circ.commands_of_type(OpType.Conditional)
    for cmd in conditional_cmds:
        assert isinstance(cmd.op, Conditional)
        assert isinstance(cmd.op.op, CircBox)
        if_circ = cmd.op.op.get_circuit()
        # Assert that each "If" block has only one Z-axis rotation
        assert if_circ.name == "If"
        assert if_circ.n_gates == 1
        assert if_circ.n_gates_of_type(OpType.Rz) == 1
