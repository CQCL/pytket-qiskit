# Pytket Extensions

This repository contains the pytket-qiskit extension, using Quantinuum's
[pytket](https://cqcl.github.io/tket/pytket/api/index.html) quantum SDK.

# pytket-qiskit

[Pytket](https://cqcl.github.io/tket/pytket/api/index.html) is a python module for interfacing
with tket, a quantum computing toolkit and optimisation compiler developed by Quantinuum.

`pytket-qiskit` is an extension to `pytket` that allows `pytket` circuits to be
run on IBM backends and simulators, as well as conversion to and from Qiskit
representations.

## Getting started

`pytket-qiskit` is available for Python 3.8, 3.9 and 3.10, on Linux, MacOS
and Windows. To install, run:

::

    pip install pytket-qiskit


Backends Available Through pytket-qiskit
========================================

The ``pytket-qiskit`` extension has several types of available ``Backend``. These are the ``IBMQBackend``
and several types of simulator.

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Backend
     - Type
   * - `IBMQBackend <https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.IBMQBackend>`_
     - Interface to an IBM quantum computer.
   * - `IBMQEmulatorBackend <https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.IBMQEmulatorBackend>`_
     - Emulator for a chosen ``IBMBackend`` (Device specific).
   * - `AerBackend <https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerBackend>`_
     - A noiseless, shots-based simulator for quantum circuits [1]
   * - `AerStateBackend <https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerStateBackend>`_
     - Statevector simulator.
   * - `AerUnitaryBackend <https://cqcl.github.io/pytket-extensions/api/qiskit/api.html#pytket.extensions.qiskit.AerUnitaryBackend>`_ 
     - Unitary simulator

* [1] ``AerBackend`` is noiseless by default and has no architecture. However it can accept a user defined ``NoiseModel`` and ``Architecture``.

::

    from pytket.extensions.qiskit import IBMQBackend

    backend = IBMQBackend('ibmq_guadalupe')

Access and Credentials
======================

Default Compilation
===================

Every ``Backend`` in pytket has its own ``default_compilation_pass`` method. This method applies a sequence of optimisations to a circuit depending on the value of an ``optimisation_level`` parameter. This default compilation will ensure that the circuit meets all the constraints required to run on the Backend. The passes applied by different levels of optimisation are specified in the table below.

.. list-table:: **Default compilation pass for the IBMQBackend and IBMQEmulatorBackend**
   :widths: 25 25 25
   :header-rows: 1

   * - optimisation_level = 0
     - optimisation_level = 1 [1]
     - optimisation_level = 2
   * - `DecomposeBoxes <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.DecomposeBoxes>`_
     - `DecomposeBoxes <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.DecomposeBoxes>`_
     - `DecomposeBoxes <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.DecomposeBoxes>`_
   * - `CXMappingPass <https://cqcl.github.io/tket/pytket/api passes.html#pytket.passes.CXMappingPass>`_ [2]
     - `SynthesiseTket <https://cqcl.github.io/tket/pytket/api passes.html#pytket.passes.SynthesiseTket>`_
     - `FullPeepholeOptimise <https://cqcl.github.io/tket/pytket/api passes.html#pytket.passes.FullPeepholeOptimise>`_
   * - `NoiseAwarePlacement <https://cqcl.github.io/tket/pytket/api/placement.html#pytket.placement.NoiseAwarePlacement>`_
     - `CXMappingPass <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.CXMappingPass>`_ [2]
     - `CXMappingPass <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.CXMappingPass>`_ [2]
   * - `NaivePlacementPass <https://cqcl.github.io/tket/pytket/api/placement.html#pytket.passes.NaivePlacementPass>`_
     - `NaivePlacementPass <https://cqcl.github.io/tket/pytket/api/placement.html#pytket.passes.NaivePlacementPass>`_
     - `NaivePlacementPass <https://cqcl.github.io/tket/pytket/api/placement.html#pytket.passes.NaivePlacementPass>`_
   * - self.rebase_pass [3]
     - `SynthesiseTket <https://cqcl.github.io/tket/pytket/api passes.html#pytket.passes.SynthesiseTket>`_
     - `CliffordSimp(allow_swaps=False) <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.CliffordSimp>`_
   * - `RemoveRedundancies <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.RemoveRedundancies>`_
     - `SimplifyInitial <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.SimplifyInitial>`_
     - `SynthesiseTket <https://cqcl.github.io/tket/pytket/api passes.html#pytket.passes.SynthesiseTket>`_
   * -
     - self.rebase_pass [3]
     - `RemoveRedundancies <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.RemoveRedundancies>`_
   * - 
     -
     - `SimplifyInitial <https://cqcl.github.io/tket/pytket/api/passes.html#pytket.passes.SimplifyInitial>`_ [5].

* [1] If no value is specified then ``optimisation_level`` defaults to a value of 1.
* [2] Here ``CXMappingPass`` maps program qubits to the architecture using a `NoiseAwarePlacement <https://cqcl.github.io/tket/pytket/api/placement.html#pytket.placement.NoiseAwarePlacement>`_
* [3] self.rebase_pass is a rebase to the gateset supported by the backend, For IBM quantum devices that is {X, SX, Rz, CX}.
* [4] The ``default_compilation_pass`` for ``AerBackend`` is the same as above except it doesn't use ``SimplifyInitial``.
* [5] ``SimplifyInitial`` has arguments ``allow_classical=False`` and ``create_all_qubits=True``.

Backend Predicates
==================

Circuits must satisfy certain conditions before they can be processed on a device or simulator. In pytket these conditions are called predicates.

All pytket-qiskit backends have the following two predicates.

* `GateSetPredicate <https://cqcl.github.io/tket/pytket/api/predicates.html#pytket.predicates.GateSetPredicate>`_ - The circuit must contain only operations supported by the ``Backend``. To view supported Ops run ``BACKENDNAME.backend_info.gate_set``.
* `NoSymbolsPredicate <https://cqcl.github.io/tket/pytket/api/predicates.html#pytket.predicates.NoSymbolsPredicate>`_ - Parameterised gates must have numerical values when the circuit is executed.

The ``IBMQBackend`` and ``IBMQEmulatorBackend`` may also have the following predicates depending on the capabilities of the specified device.

* `NoClassicalControlPredicate <https://cqcl.github.io/tket/pytket/api/predicates.html#pytket.predicates.NoClassicalControlPredicate>`_
* `NoMidMeasurePredicate <https://cqcl.github.io/tket/pytket/api/predicates.html#pytket.predicates.NoMidMeasurePredicatePredicate>`_
* `NoFastFeedforwardPredicate <https://cqcl.github.io/tket/pytket/api/predicates.html#pytket.predicates.NoFastFeedforwardPredicate>`_


This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Bugs, support and feature requests

Please file bugs and feature requests on the Github
[issue tracker](https://github.com/CQCL/pytket-qiskit/issues).

There is also a Slack channel for discussion and support. Click [here](https://tketusers.slack.com/join/shared_invite/zt-18qmsamj9-UqQFVdkRzxnXCcKtcarLRA#/shared-invite/email) to join.

## Development

To install an extension in editable mode, simply change to its subdirectory
within the `modules` directory, and run:

```shell
pip install -e .
```

## Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed
changes on the `develop` branch, and open a PR from your fork. If it passes
tests and is accepted after review, it will be merged in.

### Code style

#### Formatting

All code should be formatted using
[black](https://black.readthedocs.io/en/stable/), with default options. This is
checked on the CI. The CI is currently using version 20.8b1.

#### Type annotation

On the CI, [mypy](https://mypy.readthedocs.io/en/stable/) is used as a static
type checker and all submissions must pass its checks. You should therefore run
`mypy` locally on any changed files before submitting a PR. Because of the way
extension modules embed themselves into the `pytket` namespace this is a little
complicated, but it should be sufficient to run the script `modules/mypy-check`
(passing as a single argument the root directory of the module to test). The
script requires `mypy` 0.800 or above.

#### Linting

We use [pylint](https://pypi.org/project/pylint/) on the CI to check compliance
with a set of style requirements (listed in `.pylintrc`). You should run
`pylint` over any changed files before submitting a PR, to catch any issues.

### Tests

To run the tests for a module:

1. `cd` into that module's `tests` directory;
2. ensure you have installed `pytest`, `hypothesis`, and any modules listed in
the `test-requirements.txt` file (all via `pip`);
3. run `pytest`.

When adding a new feature, please add a test for it. When fixing a bug, please
add a test that demonstrates the fix.
