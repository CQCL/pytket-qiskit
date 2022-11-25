import os

from qiskit import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService

if not IBMQ.stored_account():
    print("No stored account")
    token = os.getenv("PYTKET_REMOTE_QISKIT_TOKEN")
    if token:
        print("Enabling account")
        IBMQ.enable_account(token)

provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")
b = provider.get_backend("ibmq_lima")

print("Made IBMQBackend")

service = QiskitRuntimeService(channel="ibm_quantum")

print("Made QiskitRuntimeService")
