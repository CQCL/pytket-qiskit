from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import IfElseOp

qubits = QuantumRegister(2)
clbits = ClassicalRegister(2)
circuit = QuantumCircuit(qubits, clbits)
(q0, q1) = qubits
(c0, c1) = clbits
 
circuit.h(q0)
circuit.measure(q0, c0)

with circuit.if_test((c0, 1)) as else_:
    circuit.h(q1)
with else_:
    circuit.x(q1)
circuit.measure(q1, c1)

#print(circuit)


from pytket.extensions.qiskit import qiskit_to_tk
for datum in (circuit.data):
    instr, qargs, cargs = datum.operation, datum.qubits, datum.clbits
    if type(instr) is IfElseOp:
        if_qc: QuantumCircuit = instr.params[0]
        #print(if_qc.qregs)
        #print(if_qc)
        #tkc = qiskit_to_tk(if_qc)
        #print(tkc.get_commands())

tkc = qiskit_to_tk(circuit)
