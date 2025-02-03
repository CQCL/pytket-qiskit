---
file_format: mystnb
---

# pytket-qiskit

IBM's [Qiskit](https://www.ibm.com/quantum/qiskit) is an open-source framework for quantum
computation, ranging from high-level algorithms to low-level circuit
representations, simulation and access to the IBM quantum devices and simulators.

`pytket-qiskit` is an extension to `pytket` that allows `pytket` circuits to be
run on IBM backends and simulators, as well as conversion to and from Qiskit
representations.

`pytket-qiskit` is available for Python 3.10, 3.11 and 3.12, on Linux, MacOS and
Windows. To install, run:

```
pip install pytket-qiskit
```

This will install `pytket` if it isn't already installed, and add new classes
and methods into the `pytket.extensions` namespace.

## Available IBM Backends

```{eval-rst}
.. currentmodule:: pytket.extensions.qiskit
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    IBMQBackend
    IBMQEmulatorBackend
    AerBackend
    AerStateBackend
    AerUnitaryBackend
    AerDensityMatrixBackend

```

An example using the shots-based {py:class}`AerBackend` simulator is shown below.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.qiskit import AerBackend
from pytket import Circuit

backend = AerBackend()
circ = Circuit(2).H(0).CX(0, 1).measure_all()

# Compilation not needed here as both H and CX are supported gates
result = backend.run_circuit(circ, n_shots=1000)
```

This simulator supports a large set of gates and by default has no architectural constraints or quantum noise. However the user can pass in a noise model or custom architecture to more closely model a real quantum device.

The {py:class}`AerBackend` also supports GPU simulation which can be configured as follows.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.qiskit import AerBackend

backend = AerBackend()
backend._qiskit_backend.set_option("device", "GPU")
```

:::{note}
Making use of GPU simulation requires the qiskit-aer-gpu package. This can be installed with the command

```
pip install qiskit-aer-gpu
```
:::

## Access and Credentials

With the exception of the Aer simulators, accessing devices and simulators through the `pytket-qiskit` extension requires an IBM account. An account can be set up here: <https://quantum.ibm.com/>.

Once you have created an account you can obtain an API token which you can use to configure your credentials locally.

In this section we are assuming that you have set the following variables with the corresponding values:

```{code-cell} ipython3
---
tags: [skip-execution]
---
# Replace the placeholders with your actual values

ibm_token = '<your_ibm_token_here>'
hub = '<your_hub_here>'
group = '<your_group_here>'
project = '<your_project_here>'

inst = f"{hub}/{group}/{project}"
```

### Method 1: Using {py:class}`QiskitRuntimeService`

You can use the following qiskit commands to save your IBM credentials
to disk:

```{code-cell} ipython3
---
tags: [skip-execution]
---
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_token, instance=inst)
```

To see which devices you can access, use the {py:meth}`IBMQBackend.available_devices` method. Note that it is possible to pass an optional `instance` argument to this method. This allows you to see which IBM devices are accessible with your credentials.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.qiskit import IBMQBackend

backend = IBMQBackend("ibm_kyiv") # Initialise backend for an IBM device

backendinfo_list = backend.available_devices(instance=inst)
print([backend.device_name for backend in backendinfo_list])
```

For more information, see the documentation for [qiskit-ibm-runtime](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime).

### Method 2: Saving credentials in a local pytket config file

Alternatively, you can store your credentials in local pytket config using the {py:meth}`~pytket.extensions.qiskit.backends.config.set_ibmq_config` method.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.qiskit import set_ibmq_config

set_ibmq_config(ibmq_api_token=ibm_token)
```

After saving your credentials you can access `pytket-qiskit` backend repeatedly without having to re-initialise your credentials.

If you are a member of an IBM hub then you can add this information to {py:meth}`~pytket.extensions.qiskit.backends.config.set_ibmq_config` as well.

```{code-cell} ipython3
---
tags: [skip-execution]
---
from pytket.extensions.qiskit import set_ibmq_config

set_ibmq_config(ibmq_api_token=ibm_token, instance=f"{hub}/{group}/{project}")
```

```{eval-rst}
.. currentmodule:: pytket.extensions.qiskit.backends.config
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    QiskitConfig
    set_ibmq_config
```

## Converting circuits between pytket and qiskit

Users may wish to port quantum circuits between pytket and qiskit. This allows the features of both libraries to be used.
For instance those familiar with qiskit may wish to convert their circuits to pytket and use the available compilation passes to optimise circuits.

```{eval-rst}
.. currentmodule:: pytket.extensions.qiskit

```

```{eval-rst}
.. autosummary::
    :nosignatures:

    qiskit_to_tk
    tk_to_qiskit

```

## Default Compilation

Every {py:class}`~pytket.backends.backend.Backend` in pytket has its own {py:meth}`~pytket.backends.Backend.default_compilation_pass` method. This method applies a sequence of optimisations to a circuit depending on the value of an `optimisation_level` parameter. This default compilation will ensure that the circuit meets all the constraints required to run on the {py:class}`~pytket.backends.backend.Backend`. The passes applied by different levels of optimisation are specified in the table below. Note that optimisation levels 0, 1 and
2 preserve barriers in a circuit, while optimisation level 3 will remove them.

:::{list-table} **Default compilation pass for the IBMQBackend and IBMQEmulatorBackend**
:widths: 25 25 25 25
:header-rows: 1

* - optimisation_level = 0
  - optimisation_level = 1
  - optimisation_level = 2 [1]
  - optimisation_level = 3
* - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
  - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
  - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
  - [DecomposeBoxes](inv:#*.passes.DecomposeBoxes)
* - [AutoRebase [2]](inv:#*.AutoRebase)
  - [SynthesiseTket](inv:#*.SynthesiseTket)
  - [FullPeepholeOptimise](inv:#*.passes.FullPeepholeOptimise)
  - [RemoveBarriers](inv:#*pytket._tket.passes.RemoveBarriers)
* - LightSabre [3]
  - LightSabre [3]
  - LightSabre [3]
  - [AutoRebase [2]](inv:#*.AutoRebase)
* - [AutoRebase [2]](inv:#*.AutoRebase)
  - [SynthesiseTket](inv:#*.SynthesiseTket)
  - [KAKDecomposition(allow_swaps=False)](inv:#*.passes.KAKDecomposition)
  - [GreedyPauliSimp](inv:#*.passes.GreedyPauliSimp)
* - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
  - [AutoRebase [2]](inv:#*.AutoRebase)
  - [CliffordSimp(allow_swaps=False)](inv:#*.passes.CliffordSimp)
  - [AutoRebase [2]](inv:#*.AutoRebase)
* -
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
  - [SynthesiseTket](inv:#*.SynthesiseTket)
  - LightSabre [3]
* -
  -
  - [AutoRebase [2]](inv:#*.AutoRebase)
  - [SynthesiseTket](inv:#*.SynthesiseTket)
* -
  -
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)
  - [AutoRebase [2]](inv:#*.AutoRebase)

* -
  -
  -
  - [RemoveRedundancies](inv:#*.passes.RemoveRedundancies)

:::

- \[1\] If no value is specified then `optimisation_level` defaults to a value of 2.
- \[2\] {py:class}`~pytket.passes.AutoRebase` is a conversion to the gateset supported by the backend. For IBM quantum devices and emulators the supported gate set is either $\{X, SX, Rz, CX\}$, $\{X, SX, Rz, ECR\}$, or $\{X, SX, Rz, CZ\}$. The more idealised Aer simulators have a much broader range of supported gates.
- \[3\] This is imported from qiskit and corresponds to the method in "LightSABRE: A Lightweight and Enhanced SABRE Algorithm", Henry Zou, Matthew Treinish, Kevin Hartman, Alexander Ivrii, Jake Lishman, arXiv:2409.08368.

**Note:** The {py:meth}`~AerBackend.default_compilation_pass` for {py:class}`AerBackend` is the same as above if a {py:class}`NoiseModel` is used. A {py:class}`NoiseModel` implicitly defines connectivity constraints via edge errors. If no {py:class}`NoiseModel` is used then then any passes related to connectivity constraints are omitted from the {py:meth}`~AerBackend.default_compilation_pass` for {py:class}`AerBackend`.

## Noise Modelling

```{eval-rst}
.. currentmodule:: pytket.extensions.qiskit.backends.crosstalk_model
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    CrosstalkParams

```

## Using TKET directly on qiskit circuits

```{eval-rst}
.. currentmodule:: pytket.extensions.qiskit
```

For usage of {py:class}`~tket_backend.TketBackend` see the [qiskit integration notebook example](https://docs.quantinuum.com/tket/user-guide/examples/backends/qiskit_integration.html).

```{eval-rst}
.. autosummary::
    :nosignatures:

    ~tket_backend.TketBackend
    ~tket_pass.TketPass
    ~tket_pass.TketAutoPass
    ~tket_job.TketJob


```

```{eval-rst}
.. toctree::
    api.md
    changelog.md
```

```{eval-rst}
.. toctree::
   :caption: Useful links

   Issue tracker <https://github.com/CQCL/pytket-qiskit/issues>
   PyPi <https://pypi.org/project/pytket-qiskit/>
```
