API documentation
~~~~~~~~~~~~~~~~~

.. currentmodule:: pytket.extensions.qiskit


Available IBM backends
----------------------

.. autosummary::
    :nosignatures:

    IBMQBackend
    IBMQEmulatorBackend
    AerBackend
    AerStateBackend
    AerUnitaryBackend

Example usage of :py:class:`IBMQBackend`

::

    from pytket.extensions.qiskit import IBMQBackend
    from pytket import Circuit

    circ = Circuit(3).H(0).CCX(0, 1, 2).H(0).measure_all() # Define Circuit
    
    backend = IBMQBackend("ibm_hanoi") # Initalise Backend

    compiled_circ = backend.get_compiled_circuit(circ) # Compile
    handle = backend.process_circuit(compiled_circ, n_shots=500) # Execute circuit
    result = backend.get_result(handle) # Retrieve result


Converting circuits between pytket and qiskit
---------------------------------------------

.. autosummary::
    :nosignatures:

    qiskit_to_tk
    tk_to_qiskit

.. jupyter-execute::

    from qiskit import QuantumCircuit
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.circuit.display import render_circuit_jupyter

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    tkc = qiskit_to_tk(qc)
    render_circuit_jupyter(tkc)


.. jupyter-execute::

    from pytket.extensions.qiskit import tk_to_qiskit

    qc2 = tk_to_qiskit(tkc)
    print(qc2)


Using TKET directly on qiskit circuits
--------------------------------------
.. currentmodule:: pytket.extensions.qiskit.tket_backend

.. autosummary::
    :nosignatures:

    TketBackend

.. currentmodule:: pytket.extensions.qiskit.tket_pass

.. autosummary::
    :nosignatures:

    TketPass

.. currentmodule:: pytket.extensions.qiskit.tket_job

.. autosummary::
    :nosignatures:

    TketJob




.. autoclass:: pytket.extensions.qiskit.IBMQBackend
    :special-members: __init__
    :show-inheritance:
    :members:

.. autoclass:: pytket.extensions.qiskit.IBMQEmulatorBackend
    :special-members: __init__
    :show-inheritance:
    :members:

.. autoclass:: pytket.extensions.qiskit.AerBackend
    :special-members: __init__
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: pytket.extensions.qiskit.AerStateBackend
    :special-members: __init__
    :inherited-members:
    :show-inheritance:
    :members:

.. autoclass:: pytket.extensions.qiskit.AerUnitaryBackend
    :special-members: __init__
    :show-inheritance:
    :inherited-members:
    :members:

.. automodule:: pytket.extensions.qiskit
    :members: qiskit_to_tk, tk_to_qiskit, process_characterisation

.. automodule:: pytket.extensions.qiskit.tket_backend
    :show-inheritance:
    :members:
    :special-members: __init__

.. automodule:: pytket.extensions.qiskit.backends.crosstalk_model
    :members: CrosstalkParams


.. automodule:: pytket.extensions.qiskit.tket_pass
    :special-members: __init__
    :members:

.. automodule:: pytket.extensions.qiskit.tket_job
    :special-members: __init__
    :members:

.. automodule:: pytket.extensions.qiskit.backends.config
    :members:
