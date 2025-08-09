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

from dataclasses import dataclass
from typing import Any, ClassVar

from pytket.config import PytketExtConfig


@dataclass
class QiskitConfig(PytketExtConfig):
    """Holds config parameters for pytket-qiskit."""

    ext_dict_key: ClassVar[str] = "qiskit"

    instance: str | None
    ibmq_api_token: str | None

    @classmethod
    def from_extension_dict(
        cls: type["QiskitConfig"], ext_dict: dict[str, Any]
    ) -> "QiskitConfig":
        return cls(
            ext_dict.get("instance"),
            ext_dict.get("ibmq_api_token"),
        )


def set_ibmq_config(
    instance: str | None = None,
    ibmq_api_token: str | None = None,
) -> None:
    """Set default values for instance or API token for your IBMQ provider. Can be
    overridden in backend construction."""

    config = QiskitConfig.from_default_config_file()
    if instance is not None:
        config.instance = instance
    if ibmq_api_token is not None:
        config.ibmq_api_token = ibmq_api_token
    config.update_default_config_file()
