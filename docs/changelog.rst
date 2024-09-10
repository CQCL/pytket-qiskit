Changelog
~~~~~~~~~

.. currentmodule:: pytket.extensions.qiskit


0.56.0 (September 2024)
-----------------------

* Added :py:class:`AerDensityMatrixBackend` simulator. This simulator has the option to support a :py:class:`NoiseModel`.
* Fix conversion of symbols into qiskit.
* Require qiskit >= 1.2.0.
* Add conversion of controlled unitary gates from qiskit to tket.
* Initialize `TketAutoPass` with a `BackendV2`.
* Update `TketBackend` to derive from `BackendV2`.
* Fix to allow `AerBackend` to work with multi-controlled Z gates.

0.55.0 (July 2024)
------------------

* Updated pytket version requirement to 1.30.

0.54.1 (June 2024)
------------------

* Relax version requirements on dependencies by removing upper bounds.
* Fix status reporting when running on hardware.

0.54.0 (June 2024)
------------------

* User can pass a `SamplerOptions` instance (from `qiskit-ibm-runtime`)
  via a keyword argument to both an `IBMQBackend` constructor and 
  an instance method `IBMQBackend.process_circuits`.
* Remove dependency on deprecated qiskit-ibm-provider.
* Remove support for deprecated "ibmq_qasm_simulator" backend.
* Forbid circuits with incomplete classical registers in ``tk_to_qiskit()``.
* Updated pytket version requirement to 1.29.
* Update qiskit-ibm-runtime version requirement to 0.24.1.
* Update qiskit version requirement to 1.1.
* Update qiskit-aer version requirement to 0.14.2.
* When constructing an Aer backend with a name for which more than one is
  available, emit a warning and pick the first in the list.

0.53.0 (April 2024)
-------------------

* Updated pytket version requirement to 1.27.
* Update qiskit-ibm-runtime version requirement to 0.23.

0.52.0 (April 2024)
-------------------

* Update pytket version requirement to 1.26.
* Update qiskit-aer version requirement to 0.14.
* Update conversion to qiskit to use symengine for symbolic circuits
* Add `IBMQBackend.default_compilation_pass_offline` for offline compilation given config and props objects.
* Add `DirectednessPredicate` to IBMQBackend
* Default compilation pass of IBMQBackend will keep ECR gates in the direction required by the backend.

0.51.0 (March 2024)
-------------------

* Update qiskit-ibm-runtime version requirement to 0.22.
* remove all remote simulators
* rename ``IBMQLocalEmulatorBackend`` to ``IBMQEmulatorBackend``
* ``IBMQEmulatorBackend`` will now run locally
* add support for contextual optimisation on local emulator

0.50.0 (March 2024)
-------------------

* Update qiskit-algorithms version requirement to 0.3.

0.49.0 (March 2024)
-------------------

* Update pytket version requirement to 1.25.
* Update qiskit version requirement to 1.0.
* Update qiskit-ibm-provider version requirement to 0.10.
* Update qiskit-ibm-runtime version requirement to 0.21.
* Add ``IBMQLocalEmulatorBackend`` for running local emulation of
  ``IBMQBackend`` using ``AerBackend`` with a noise model.

0.48.1rc1
---------

* Fix: use correct ``instance`` with ``IBMQEmulatorBackend``.

0.48.0 (January 2024)
---------------------

* Updated pytket version requirement to 1.24.
* Python 3.12 support added, 3.9 dropped.
* fix conditional bit in pytket to qiskit conversion
* fix symbolic conversion of parameter in conversion

0.47.0 (January 2024)
---------------------

* Update qiskit-ibm-runtime version to 0.17.0.
* Update qiskit-ibm-provider version to 0.8.0.
* Updated pytket version requirement to 1.23.

0.46.0 (November 2023)
----------------------

* Updated pytket version requirement to 1.22.
* Add support for circuits with barriers in the Aer simulators.
* Update qiskit version to 0.45.0.
* Update qiskit-ibm-runtime version to 0.15.1.
* Update qiskit-aer version to 0.13.0.
* Update qiskit-ibm-provider version to 0.7.2.
* Introduce dependency on qiskit-algorithms.
* Seed given to ``process_circuits()`` will be automatically incremented
  for the different circuit batches submitted.
* Fix :py:class:`RuntimeError` caused by the use of custom gates in ``qiskit_to_tk`` `#200 <https://github.com/CQCL/pytket-qiskit/issues/200>`_.

0.45.0 (October 2023)
---------------------

* Updated pytket version requirement to 1.21.
* Implement crosstalk noise model for AerBackend.
* Don't include ``SimplifyInitial`` in default passes; instead make it an option
  to ``process_circuits()``.

0.44.0 (September 2023)
-----------------------

* Fix to add include Measure, Reset and Conditional operations to the supported operations of :py:class:`AerStateBackend`.
* Update qiskit-ibm-runtime version to 0.12.0.
* Update qiskit-ibm-provider version to 0.7.0.
* Update pytket version requirement to 1.19.

0.43.0 (August 2023)
--------------------

* Update qiskit version to 0.44.
* Update qiskit-aer version to 0.12.2.
* Update qiskit-ibm-runtime version to 0.11.3.
* Update qiskit-ibm-provider version to 0.6.3.
* Add option to specify the maximum number of qubits supported by Aer backends
  (defaults to 40).

0.42.0 (August 2023)
--------------------

* Update pytket version requirement to 1.18.

0.41.0 (July 2023)
------------------

* Update pytket version requirement to 1.17.
* Fix conversion of qiskit `UnitaryGate` to and from pytket (up to 3 qubits).
* Fix handling of qiskit controlled gates in the :py:meth:`qiskit_to_tk` converter.
* Handle CCZ and CSX gates in circuit converters.

0.40.0 (June 2023)
------------------

* IBM devices are now accessed using the `qiskit-ibm-provider <https://github.com/Qiskit/qiskit-ibm-provider>`_ instead of the deprecated :py:class:`IBMQ`. This allows the newest IBM devices and simulators to be accessed through ``pytket-qiskit``. See the updated documentation on `credentials <https://tket.quantinuum.com/extensions/pytket-qiskit/index.html#access-and-credentials>`_.
* The parameters ``hub``, ``group`` and ``project`` are no longer handled as separate arguments in :py:class:`IBMQBackend` and :py:meth:`IBMQBackend.available_devices`. Use ``"instance=f"{hub}/{group}/{project}"`` instead.
* Added support for the {X, SX, Rz, ECR} in the default compilation pass for :py:class:`IBMQBackend` and :py:class:`IBMQEmulatorBackend`. This is the set of gates used by some of the new IBM devices.
* Fix to the :py:meth:`tk_to_qiskit` converter to prevent cancellation of redundant gates when converting to qiskit.
* Handle qiskit circuits with :py:class:`Initialize` and :py:class:`StatePreparation` instructions in the :py:meth:`qiskit_to_tk` converter. The :py:meth:`tk_to_qiskit` converter now handles :py:class:`StatePreparationBox`.
* Fix handling of control state in :py:meth:`qiskit_to_tk`.
* Update qiskit version to 0.43.1
* Update qiskit-ibm-runtime version to 0.11.1
* Update qiskit-ibm-provider version to 0.6.1
* Update pytket version to 1.16

0.39.0 (May 2023)
-----------------

* Updated pytket version requirement to 1.15.
* The :py:meth:`IBMQBackend.get_compiled_circuit` method now allows for optional arguments to override the default settings in the :py:class:`NoiseAwarePlacement`.

0.38.0 (April 2023)
-------------------

* Fix to ensure that the :py:class:`IBMBackend` and :py:class:`IBMQEmulatorBackend` both properly enforce :py:class:`MaxNQubitsPredicate`.
* Update qiskit version to 0.42.
* Updated pytket version requirement to 1.14.

0.37.1 (March 2023)
-------------------

* Fix backend settings for AerStateBackend and AerUnitaryBackend

0.37.0 (March 2023)
-------------------

* Fix faulty information in ``AerBackend().backend_info``
* Updated pytket version requirement to 1.13.

0.36.0 (February 2023)
----------------------

* Update qiskit version to 0.41.
* Fix order of Pauli terms when converting from ``QubitPauliOperator``.

0.35.0 (February 2023)
----------------------

* Automatically use IBMQ token if saved in pytket config and not saved in qiskit
  config.
* Update qiskit version to 0.40.
* Update code to remove some deprecation warnings.
* Work around https://github.com/Qiskit/qiskit-terra/issues/7865.

0.34.0 (January 2023)
---------------------

* Handle more multi-controlled gates in ``tk_to_qiskit`` and ``qiskit_to_tk`` converters (including CnY and CnZ).
* Drop support for Python 3.8; add support for 3.11.
* Fix ordering of registers in statevector simulation results.
* Remove ``reverse_index`` argument in ``tk_to_qiskit()``.
* Updated pytket version requirement to 1.11.

0.33.0 (December 2022)
----------------------

* Fix handling of parameter when converting ``PauliEvolutionGate`` to
  ``QubitPauliOperator``.
* Updated pytket version requirement to 1.10.

0.32.0 (December 2022)
----------------------

* Use ``qiskit_ibm_runtime`` services for sampling on ``IBMQBackend`` and
  ``IBMQEmulatorBackend``. Note that shots tables (ordered lists of results) are
  no longer available from these backends. (``BackendResult.get_shots()`` will
  fail; use ``get_counts()`` instead.)

* Fix incorrect circuit permutation handling for ``AerUnitaryBackend`` and ``AerStateBackend``.

0.31.0 (November 2022)
----------------------

* Update ``TketBackend`` to support ``FullyConnected`` architecture.
* Fix the issue that some qiskit methods can't retrieve results from ``TketJob``.
* Updated pytket version requirement to 1.9.
* Handle ``OpType.Phase`` when converting to qiskit.
* Change default optimization level in ``default_compilation_pass()`` to 2.

0.30.0 (November 2022)
----------------------

* Update qiskit version to 0.39.
* ``tk_to_qiskit`` now performs a rebase pass prior to conversion. Previously an error was returned if a ``Circuit`` contained gates such as ``OpType.ZZMax`` which have no exact replacement in qiskit. Now the unsupported gate will be implemented in terms of gates supported in qiskit rather than returning an error.
* Updated pytket version requirement to 1.8.

0.29.0 (October 2022)
---------------------

* Add post-routing ``KAKDecomposition`` to default pass with ``optimisation_level`` = 2.
* Add support for ``ECRGate`` in ``tk_to_qiskit`` conversion.
* Update qiskit version to 0.38.
* Updated pytket version requirement to 1.7.


0.28.0 (August 2022)
--------------------

* Improve result retrieval speed of ``AerUnitaryBackend`` and ``AerStateBackend``.
* Update qiskit version to 0.37.
* Updated pytket version requirement to 1.5.

0.27.0 (July 2022)
------------------

* Updated pytket version requirement to 1.4.

0.26.0 (June 2022)
------------------

* Updated pytket version requirement to 1.3.

0.25.0 (May 2022)
-----------------

* Updated pytket version requirement to 1.2.

0.24.0 (April 2022)
-------------------

* Fix two-qubit unitary conversions.
* Update qiskit version to 0.36.
* Updated pytket version requirement to 1.1.

0.23.0 (March 2022)
-------------------

* Removed ``characterisation`` property of backends. (Use `backend_info`
  instead.)
* Updated pytket version requirement to 1.0.

0.22.2 (February 2022)
----------------------

* Fixed :py:meth:`IBMQEmulatorBackend.rebase_pass`.

0.22.1 (February 2022)
----------------------

* Added :py:meth:`IBMQEmulatorBackend.rebase_pass`.

0.22.0 (February 2022)
----------------------

* Qiskit version updated to 0.34.
* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.

0.21.0 (January 2022)
---------------------

* Qiskit version updated to 0.33.
* Updated pytket version requirement to 0.18.

0.20.0 (November 2021)
----------------------

* Qiskit version updated to 0.32.
* Updated pytket version requirement to 0.17.

0.19.0 (October 2021)
---------------------

* Qiskit version updated to 0.31.
* Removed deprecated :py:meth:`AerUnitaryBackend.get_unitary`. Use
  :py:meth:`AerUnitaryBackend.run_circuit` and
  :py:meth:`pytket.backends.backendresult.BackendResult.get_unitary` instead.
* Updated pytket version requirement to 0.16.

0.18.0 (September 2021)
-----------------------

* Qiskit version updated to 0.30.
* Updated pytket version requirement to 0.15.

0.17.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.16.1 (July 2021)
------------------

* Fix slow/high memory use :py:meth:`AerBackend.get_operator_expectation_value`

0.16.0 (July 2021)
------------------

* Qiskit version updated to 0.28.
* Use provider API client to check job status without retrieving job in IBMQBackend.
* Updated pytket version requirement to 0.13.

0.15.1 (July 2021)
------------------

* Fixed bug in backends when n_shots argument was passed as list.

0.15.0 (June 2021)
------------------

* Updated pytket version requirement to 0.12.

0.14.0 (unreleased)
-------------------

* Qiskit version updated to 0.27.

0.13.0 (May 2021)
-----------------

* Updated pytket version requirement to 0.11.

0.12.0 (unreleased)
-------------------

* Qiskit version updated to 0.26.
* Code rewrites to avoid use of deprecated qiskit methods.
* Restriction to hermitian operators for expectation values in `AerBackend`.

0.11.0 (May 2021)
-----------------

* Contextual optimisation added to default compilation passes (except at optimisation level 0).
* Support for symbolic parameters in rebase pass.
* Correct phase when rebasing.
* Ability to preserve UUIDs of qiskit symbolic parameters when converting.
* Correction to error message.

0.10.0 (April 2021)
-------------------

* Support for symbolic phase in converters.
