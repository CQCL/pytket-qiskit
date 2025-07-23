# API documentation

```{eval-rst}
.. automodule:: pytket.extensions.qiskit
.. automodule:: pytket.extensions.qiskit._metadata
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.backends
.. automodule:: pytket.extensions.qiskit.backends.ibm_utils
.. automodule:: pytket.extensions.qiskit.backends.ibm
.. autoclass:: pytket.extensions.qiskit.backends.ibm.IBMQBackend
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: available_devices
    .. automethod:: cancel
    .. automethod:: circuit_status
    .. automethod:: default_compilation_pass
    .. automethod:: default_compilation_pass_offline
    .. automethod:: get_compiled_circuit
    .. automethod:: get_compiled_circuits
    .. automethod:: get_result
    .. automethod:: process_circuits
    .. automethod:: rebase_pass
    .. automethod:: rebase_pass_offline
    .. automethod:: squash_pass_offline
    .. autoproperty:: backend_info
    .. autoproperty:: required_predicates

.. autoexception:: pytket.extensions.qiskit.backends.ibm.NoIBMQCredentialsError
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.backends.ibmq_emulator
.. autoclass:: pytket.extensions.qiskit.backends.ibmq_emulator.IBMQEmulatorBackend
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: cancel
    .. automethod:: circuit_status
    .. automethod:: default_compilation_pass
    .. automethod:: get_result
    .. automethod:: process_circuits
    .. automethod:: rebase_pass
    .. autoproperty:: backend_info
    .. autoproperty:: required_predicates
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.backends.aer
.. autoclass:: pytket.extensions.qiskit.backends.aer.AerBackend
    :special-members: __init__
    :inherited-members:
    :members:
```

```{eval-rst}
.. autoclass:: pytket.extensions.qiskit.backends.aer.AerStateBackend
    :special-members: __init__
    :inherited-members:
    :members:
```

```{eval-rst}
.. autoclass:: pytket.extensions.qiskit.backends.aer.AerUnitaryBackend
    :special-members: __init__
    :inherited-members:
    :members:
```

```{eval-rst}
.. autoclass:: pytket.extensions.qiskit.backends.aer.AerDensityMatrixBackend
    :special-members: __init__
    :inherited-members:
    :members:
```

```{eval-rst}
.. autofunction:: pytket.extensions.qiskit.backends.aer.qiskit_aer_backend
.. autoclass:: pytket.extensions.qiskit.backends.aer.NoiseModelCharacterisation
    :members:
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.qiskit_convert

    .. autofunction:: qiskit_to_tk
    .. autofunction:: tk_to_qiskit
    .. autofunction:: process_characterisation
    .. autofunction:: get_avg_characterisation
    .. autofunction:: process_characterisation_from_config
    .. autofunction:: append_tk_command_to_qiskit
    .. autofunction:: param_to_qiskit
    .. autofunction:: param_to_tk
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.tket_backend
    :show-inheritance:
    :members:
    :special-members: __init__
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.backends.crosstalk_model
    :members: CrosstalkParams, NoiseGate, NoisyCircuitBuilder, FractionalUnitary, get_gate_times_from_backendinfo

```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.tket_pass
    :special-members: __init__
    :members:
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.tket_job
    :special-members: __init__
    :members:
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.result_convert

    .. autofunction:: backendresult_to_qiskit_resultdata
    .. autofunction:: qiskit_result_to_backendresult
    .. autofunction:: qiskit_experimentresult_to_backendresult
```

```{eval-rst}
.. automodule:: pytket.extensions.qiskit.backends.config
    :members:
```
