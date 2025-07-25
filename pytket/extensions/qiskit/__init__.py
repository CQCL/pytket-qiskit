# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# _metadata.py is copied to the folder after installation.
from ._metadata import __extension_name__, __extension_version__
from .backends import (
    AerBackend,
    AerDensityMatrixBackend,
    AerStateBackend,
    AerUnitaryBackend,
    IBMQBackend,
    IBMQEmulatorBackend,
    NoIBMQCredentialsError,
)
from .backends.config import set_ibmq_config
from .qiskit_convert import process_characterisation, qiskit_to_tk, tk_to_qiskit

# from .tket_pass import TketPass
