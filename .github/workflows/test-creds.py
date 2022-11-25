import os
from pytket.extensions.qiskit import IBMQBackend
from qiskit import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService

if not IBMQ.stored_account():
    token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
    if token:
        print("Enabling account")
        IBMQ.enable_account(token)

b = IBMQBackend("ibmq_lima", hub="ibm-q", group="open", project="main")

print("Made IBMQBackend")

service = QiskitRuntimeService(channel="ibm_quantum")

print("Made QiskitRuntimeService")
