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

Converting circuits between pytket and qiskit
---------------------------------------------

Users may wish to port quantum circuits between pytket and qiskit. This allows the features of both libraries to be used.
For instance those familiar with qiskit may wish to convert their circuits to pytket and use the available compilation passes to optimise circuits.

.. autosummary::
    :nosignatures:

    qiskit_to_tk
    tk_to_qiskit



Using TKET directly on qiskit circuits
--------------------------------------

For usage of :py:class:`TketBackend` see the `qiskit integration notebook example <https://tket.quantinuum.com/examples/qiskit_integration.html>`_.

.. currentmodule:: pytket.extensions.qiskit.tket_backend

.. autosummary::
    :nosignatures:

    TketBackend

.. currentmodule:: pytket.extensions.qiskit.tket_pass

.. autosummary::
    :nosignatures:

    TketPass
    TketAutoPass

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

See also the docs on `Access and Credentials <https://tket.quantinuum.com/extensions/pytket-qiskit/#access-and-credentials>`_.

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
    :inherited-members:
    :members:

.. autoclass:: pytket.extensions.qiskit.AerStateBackend
    :special-members: __init__
    :inherited-members:
    :members:

.. autoclass:: pytket.extensions.qiskit.AerUnitaryBackend
    :special-members: __init__
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
