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

Users may wish to port quantum circuits between pytket and qiskit. This allows the features of both libraries to be used.
For instance those familiar with qiskit may wish to convert their circuits to pytket and use the available compilation passes to optimise circuits.

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

The circuit converters can also handle some higher level operations. Below we show an example of using the qiskit ``Initialize`` instruction. This is handled as a :py:class:`StatePreparationBox` with reset operations.

.. jupyter-execute::

    import numpy as np

    werner_state =  1 / np.sqrt(3) * np.array([0, 1, 1, 0, 1, 0, 0, 0])

    qc_state_circ = QuantumCircuit(3)
    qc_state_circ.initialize(werner_state, [0, 1, 2])

    tkc_state_circ = qiskit_to_tk(qc_state_circ)
    render_circuit_jupyter(tkc_state_circ)


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

Noise Modelling
---------------

.. currentmodule:: pytket.extensions.qiskit.backends.crosstalk_model

.. autosummary::
    :nosignatures:

    CrosstalkParams

.. currentmodule:: pytket.extensions.qiskit.backends.config

IBM Credential Configuration
----------------------------

See also the docs on `Access and Credentials <file:///Users/callum/work_projects/pytket-qiskit/docs/build/index.html#access-and-credentials>`_.

.. currentmodule:: pytket.extensions.qiskit.backends.config
    
.. autosummary::
    :nosignatures:

    QiskitConfig
    set_ibmq_config



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
