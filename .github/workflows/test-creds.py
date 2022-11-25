from pytket.extensions.qiskit import IBMQBackend
from qiskit_ibm_runtime import QiskitRuntimeService

b = IBMQBackend("ibmq_lima", hub="ibm-q", group="open", project="main")

print("Made IBMQBackend")

service = QiskitRuntimeService(channel="ibm_quantum")

print("Made QiskitRuntimeService")
