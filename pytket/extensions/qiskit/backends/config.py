# Copyright 2021-2023 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License atF
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, ClassVar, Dict, Optional, Type
from dataclasses import dataclass
from pytket.config import PytketExtConfig


@dataclass
class QiskitConfig(PytketExtConfig):
    """Holds config parameters for pytket-qiskit."""

    ext_dict_key: ClassVar[str] = "qiskit"

    instance: Optional[str]
    ibmq_api_token: Optional[str]

    @classmethod
    def from_extension_dict(
        cls: Type["QiskitConfig"], ext_dict: Dict[str, Any]
    ) -> "QiskitConfig":
        return cls(
            ext_dict.get("instance", None),
            ext_dict.get("ibmq_api_token", None),
        )


def set_ibmq_config(
    instance: Optional[str] = None,
    ibmq_api_token: Optional[str] = None,
) -> None:
    """Set default values for any of hub, group, project or API token
    for your IBMQ provider. Can be overridden in backend construction."""

    config = QiskitConfig.from_default_config_file()
    if instance is not None:
        config.instance = instance
    if ibmq_api_token is not None:
        config.ibmq_api_token = ibmq_api_token
    config.update_default_config_file()
